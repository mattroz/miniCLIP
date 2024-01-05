import torch

import torch.nn as nn
import torch.nn.functional as F

"""CLIP paper, Figure 3"""

class ContrastiveCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pairwise_embedding):
        labels = torch.arange(pairwise_embedding.shape[0], device=pairwise_embedding.device)
        loss_vis = F.cross_entropy(pairwise_embedding, labels)
        loss_text = F.cross_entropy(pairwise_embedding.T, labels)
        loss = (loss_vis + loss_text)/2

        return loss