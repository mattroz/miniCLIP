import sys
sys.path.append(".")

import torch

import torch.nn as nn

from src.model.transformer import Transformer

class TextEncoderV1(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 n_layers: int, 
                 n_heads: int, 
                 sequence_length: int) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)         
        self.positional_encoding = nn.Parameter(torch.empty(sequence_length, d_model))
        self.transformer = Transformer(d_model, n_layers, n_heads, sequence_length)
    
    def forward(self, text: torch.Tensor, padding_mask: torch.Tensor = None, attention_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.token_embedding(text) # [batch_size, sequence_length, d_model]
        x = x + self.positional_encoding

        x = x.permute(1, 0, 2) # [batch, seqlen, d_modle] -> [seqlen, batch, d_model]
        x = self.transformer(x, padding_mask, attention_mask)
        x = x.permute(1, 0, 2) # [seq_len, batch, d_model] -> [batch, seq_len, d_model]

        return x
    

if __name__ == "__main__":
    import tiktoken

    width = 512
    layer= 8
    heads = 8
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    print("Vocab size: ", vocab_size)

    tokenized_text = tokenizer.encode("some random text to tokenize.") + [tokenizer.eot_token]
    tokenized_text = torch.tensor(tokenized_text, dtype=torch.long).unsqueeze(0)
    print("Tokenized text: ", tokenized_text)

    text_encoder = TextEncoderV1(vocab_size, width, layer, heads, sequence_length=tokenized_text.shape[1])

    out = text_encoder(tokenized_text)

    print("After text encoder: ", out.shape)