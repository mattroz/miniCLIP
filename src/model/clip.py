import sys
sys.path.append(".")

import torch

import torch.nn as nn
import numpy as np

from src.model.text_encoder import TextEncoderV1
from src.model.vision_encoder import VisionEncoderV1


class CLIP(nn.Module):
    def __init__(self, vision_encoder, text_encoder, embed_dim, temperature=0.07):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.embed_dim = embed_dim
        self.temperature = temperature
    
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        # layer to project the text embeddings to the same dimension as the vision embeddings
        self.projection = nn.Linear(self.embed_dim, self.vision_encoder.out_features, bias=False)

    def vision_adapter(self, image):
        return self.vision_encoder(image)
    
    """
    From the CLIP paper, page 5:
        The text sequence is bracketed with [SOS] and [EOS] tokens and 
        the activations of the highest layer of the transformer at the [EOS] token
        are treated as the feature representation of the text
        which is layer normalized and then linearly projected into
        the multi-modal embedding space.
    """
    def text_adapter(self, text):
        x = self.text_encoder(text)

        x = self.layer_norm(x)

        # We should pick the activations of the highest token in the sequence, 
        # which is the [EOS] token from the original text.
        eot_token_position = text.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_token_position]
        x = self.projection(x)

        return x
        

if __name__ == "__main__":
    vision_encoder = VisionEncoderV1(model_name="resnet50", pretrained=False, out_features=512)
    text_encoder = TextEncoderV1(vocab_size=11, d_model=512, n_layers=8, n_heads=8, sequence_length=16)

    dummy_image = torch.rand(3, 3, 224, 224)
    dummy_text_tokenized = torch.randint(low=0, high=10 , size=(3, 16))

    clip = CLIP(vision_encoder, text_encoder, embed_dim=512)

    encoded_vision_feat = clip.vision_adapter(dummy_image)
    encoded_text_feat = clip.text_adapter(dummy_text_tokenized)

    print("Vision features: ", encoded_vision_feat.shape)
    print("Text features: ", encoded_text_feat.shape)
