import pytest
import tiktoken

from itertools import product

from src.data.tokenizer import Tokenizer


@pytest.mark.parametrize("tokenizer_name, sequence_length", product(tiktoken.list_encoding_names(), (32, 64, 128)))
def test_tokenizer(tokenizer_name, sequence_length):
    tokenizer = Tokenizer(bos_str="<|startoftext|>", eos_str="<|endoftext|>", tokenizer_name=tokenizer_name)
    
    text_batch = ['Most of the sheep in the pasture are lying down on the grass. ', 
                  'Plate of pasta with lemon and broccoli mixed in it.', 
                  'some parked bicycles and two women on a bench and a book', 
                  'a red traffic light  and some people walking on a street', 
                  'A cat with hind legs on the edge of the bathtub and front paws on the toilet seat.', 
                  'Elvis and a young lady on a motorcycle with others standing around a store on the corner.']
    encoded_batch, attention_mask = tokenizer.encode_batch(text_batch, max_length=sequence_length)
    decoded_batch = tokenizer.decode_batch(encoded_batch, supress_special_tokens=True)
    
    assert attention_mask.shape == (len(text_batch), sequence_length)
    assert encoded_batch.shape == (len(text_batch), sequence_length)
    assert text_batch == decoded_batch

    assert "Hello world!" == tokenizer.decode(tokenizer.encode("Hello world!"))
    assert "<|startoftext|>Hello world!<|endoftext|>" == tokenizer.decode(tokenizer.encode("Hello world!"), supress_special_tokens=False)
    