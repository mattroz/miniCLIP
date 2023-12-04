import torch

import torch.nn as nn
import numpy as np


class CLIP(nn.Module):
    def __init__(self, vision_encoder, textencoder, embed_dim, temperature=0.07):
        super().__init__()
        