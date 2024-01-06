import torch
import tiktoken

import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence

class Tokenizer:
    def __init__(self, bos_str: str, eos_str: str, tokenizer_name="gpt2"):
        self.bos_str = bos_str #"<|startoftext|>"
        self.eos_str = eos_str #"<|endoftext|>"

        base_encoder = tiktoken.get_encoding(tokenizer_name)
        self.base_tokenizer = tiktoken.Encoding(
            name=f"{tokenizer_name}_miniclip",
            pat_str=base_encoder._pat_str,
            mergeable_ranks=base_encoder._mergeable_ranks,
            special_tokens={
                **base_encoder._special_tokens,
                self.bos_str: base_encoder.max_token_value,
                self.eos_str: base_encoder.max_token_value + 1
            }
        )

        self.bos_token = self.base_tokenizer.encode_single_token(self.bos_str)
        self.eos_token = self.base_tokenizer.encode_single_token(self.eos_str)
        self.vocab_size = self.base_tokenizer.n_vocab

    def encode(self, text: str) -> torch.Tensor:
        tokenized_text = self.base_tokenizer.encode(text)
        tokenized_text = torch.tensor([self.bos_token] + tokenized_text + [self.eos_token], dtype=torch.long)
        return tokenized_text

    def decode(self, tokenized_text: torch.Tensor, supress_special_tokens=True):
        decoded_text = self.base_tokenizer.decode(tokenized_text.tolist())
        
        if supress_special_tokens:
            decoded_text = decoded_text.replace(self.bos_str, "").replace(self.eos_str, "")
        
        return decoded_text

    def encode_batch(self, batched_text, max_length=None):
        tokenized_text = self.base_tokenizer.encode_batch(batched_text)
        tokenized_text = [torch.tensor([self.bos_token] + _tokenized + [self.eos_token], dtype=torch.long) for _tokenized in tokenized_text]
        padding_mask = [torch.zeros_like(_tokenized, dtype=torch.bool) for _tokenized in tokenized_text]

        tokenized_text = pad_sequence(tokenized_text, batch_first=True, padding_value=self.eos_token)
        padding_mask = pad_sequence(padding_mask, batch_first=True, padding_value=1)

        if max_length:
            tokenized_text = nn.functional.pad(tokenized_text, (0, max_length - tokenized_text.shape[1]), value=self.eos_token)
            padding_mask = nn.functional.pad(padding_mask, (0, max_length - padding_mask.shape[1]), value=1)

        return tokenized_text, padding_mask
    
    def decode_batch(self, batched_tokens, supress_special_tokens=True):
        decoded_batch = self.base_tokenizer.decode_batch(batched_tokens.tolist())
        
        if supress_special_tokens:
            decoded_batch = [decoded_batch[i].replace(self.bos_str, "").replace(self.eos_str, "") for i in range(len(decoded_batch))]
        
        return decoded_batch
    