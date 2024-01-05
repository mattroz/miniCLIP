import sys
sys.path.append(".")

import torch

import torch.nn as nn
import numpy as np

from src.model.text_encoder import TextEncoderV1
from src.model.vision_encoder import VisionEncoderV1
from src.data.tokenizer import Tokenizer


class CLIP(nn.Module):
    def __init__(self, vision_encoder, text_encoder, vision_embed_dim, text_embed_dim, embed_dim, temperature=0.07):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.vision_embed_dim = vision_embed_dim
        self.text_embed_dim = text_embed_dim
        self.embed_dim = embed_dim
        self.temperature = torch.scalar_tensor(temperature)
    
        self.layer_norm = nn.LayerNorm(self.text_embed_dim)
        # layer to project the text embeddings to the same dimension as the vision embeddings
        self.text_projection = nn.Linear(self.text_embed_dim, self.embed_dim, bias=False)
        self.vision_projection = nn.Linear(self.vision_embed_dim, self.embed_dim, bias=False)

    def vision_adapter(self, image):
        x = self.vision_encoder(image)
        x = self.vision_projection(x)
        return x
    
    """
    From the CLIP paper, page 5:
        The text sequence is bracketed with [SOS] and [EOS] tokens and 
        the activations of the highest layer of the transformer at the [EOS] token
        are treated as the feature representation of the text
        which is layer normalized and then linearly projected into
        the multi-modal embedding space.
    """
    def text_adapter(self, text, padding_mask=None, attention_mask=None):
        x = self.text_encoder(text, padding_mask, attention_mask)

        x = self.layer_norm(x)

        # We should pick the activations of the highest token in the sequence, 
        # which is the [EOS] token from the original text.
        eot_token_position = text.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_token_position]
        x = self.text_projection(x)

        return x
    
    def forward(self, image, text, padding_mask=None, attention_mask=None):
        vision_embedding = self.vision_adapter(image)
        text_embedding = self.text_adapter(text, padding_mask=padding_mask, attention_mask=attention_mask)

        vision_embedding_norm = vision_embedding / torch.linalg.norm(vision_embedding, dim=1, ord=2, keepdim=True)
        text_embedding_norm = text_embedding / torch.linalg.norm(text_embedding, dim=1, ord=2, keepdim=True)

        logits = torch.matmul(vision_embedding_norm, text_embedding_norm.t()) * torch.exp(self.temperature)

        return logits
        

if __name__ == "__main__":
    batch_size = 3
    seqlen = 32
    d_model = 512

    tokenizer = Tokenizer(bos_str="<|startoftext|>", eos_str="<|endoftext|>", tokenizer_name="gpt2")

    vision_encoder = VisionEncoderV1(model_name="resnet50", pretrained=False, out_features=d_model)
    text_encoder = TextEncoderV1(vocab_size=tokenizer.vocab_size, d_model=d_model, n_layers=8, n_heads=8, sequence_length=seqlen)

    dummy_text_batch = ['Most of the sheep in the pasture are lying down on the grass.', 
                        'Plate of pasta with lemon and broccoli mixed in it.', 
                        'some parked bicycles and two women on a bench and a book']

    dummy_image = torch.rand(batch_size, 3, 224, 224)
    dummy_text_tokenized, padding_mask = tokenizer.encode_batch(dummy_text_batch, max_length=seqlen)

    clip = CLIP(vision_encoder, text_encoder, text_embed_dim=d_model, vision_embed_dim=d_model, embed_dim=d_model)

    print("n params: ", sum(p.numel() for p in clip.parameters() if p.requires_grad))

    encoded_vision_feat = clip.vision_adapter(dummy_image)
    encoded_text_feat = clip.text_adapter(dummy_text_tokenized, padding_mask=padding_mask)
    logits = clip(dummy_image, dummy_text_tokenized, padding_mask=padding_mask)

    print("Vision features: ", encoded_vision_feat.shape)
    print("Text features: ", encoded_text_feat.shape)
    print("Logits: ", logits.shape)
    print(logits)
