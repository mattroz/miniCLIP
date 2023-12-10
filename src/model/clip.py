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


if __name__ == "__main__":
    vision_encoder = VisionEncoderV1(model_name="resnet50", pretrained=False, out_features=512)
    text_encoder = TextEncoderV1(vocab_size=11, d_model=512, n_layers=8, n_heads=8, sequence_length=16)

    dummy_image = torch.rand(1, 3, 224, 224)
    dummy_text_tokenized = torch.randint(low=0, high=10 , size=(1, 16))

    encoded_vision_feat = vision_encoder(dummy_image)
    encoded_text_feat = text_encoder(dummy_text_tokenized)

    print("Vision features: ", encoded_vision_feat.shape)
    print("Text features: ", encoded_text_feat.shape)