import torch

import torch.nn as nn


def generate_attention_mask(sequence_length: int, device: torch.device):
    mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf')).to(device)


"""
From paper: The text encoder is a Transformer (Vaswani et al., 2017)
with the architecture modifications described in Radford
et al. (2019).

From Radford et al. (2019): We use a Transformer (Vaswani et al., 2017) based archi-
tecture for our LMs. The model largely follows the details
of the OpenAI GPT model (Radford et al., 2018) with a few modifications: 
    1. Layer normalization (Ba et al., 2016)
    was moved to the input of each sub-block, similar to a
    pre-activation residual network (He et al., 2016);
    2. An additional layer normalization was added after the final self-
    attention block.

From Radford et al. (2018): 
    [ x + MMHA(x) ---> LayerNorm ---> x + Feedforward(x) ---> LayerNorm] x 12
"""
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(d_model)
        self.multihead_attn = nn.MultiheadAttention(d_model, n_head)
        self.layernorm_2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def apply_attention(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        attention_mask = attention_mask.to(dtype=x.dtype, device=x.device) if attention_mask is not None else None
        return self.multihead_attn(x, x, x, need_weights=False, attn_mask=attention_mask)[0]

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        identity = x
        x = self.layernorm_1(x)
        x = self.apply_attention(x, attention_mask)
        x = x + identity
        
        identity = x
        x = self.layernorm_2(x)
        x = self.feed_forward(x)
        x = x + identity

        return x


class Transformer(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, sequence_length: int):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.attention_mask = generate_attention_mask(sequence_length, device='cpu')
        self.blocks = nn.ModuleList([ResidualAttentionBlock(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        attention_mask = self.attention_mask if attention_mask is None else attention_mask
        for block in self.blocks:
            x = block(x, attention_mask)
        
        return x
    

if __name__ == "__main__":
    d_model, sequence_length, n_heads = 768, 16, 12
    transformer = Transformer(d_model=d_model, n_layers=12, n_heads=n_heads, sequence_length=sequence_length)
    print("n params: ", sum(p.numel() for p in transformer.parameters() if p.requires_grad))
    
    # Transformer expects (sequence_length, batch_size, d_model) input shape
    x = torch.rand(sequence_length, 8, d_model)
    print(transformer(x).shape)