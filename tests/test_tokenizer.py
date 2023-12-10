import pytest
import tiktoken

from src.data.tokenizer import Tokenizer


@pytest.mark.parametrize("tokenizer_name", tiktoken.list_encoding_names())
def test_tokenizer(tokenizer_name):
    tokenizer = Tokenizer(bos_str="<|startoftext|>", eos_str="<|endoftext|>", tokenizer_name=tokenizer_name)
    
    text_batch = ['Most of the sheep in the pasture are lying down on the grass. ', 
                  'Plate of pasta with lemon and broccoli mixed in it.', 
                  'some parked bicycles and two women on a bench and a book', 
                  'a red traffic light  and some people walking on a street', 
                  'A cat with hind legs on the edge of the bathtub and front paws on the toilet seat.', 
                  'Elvis and a young lady on a motorcycle with others standing around a store on the corner.']
    encoded_batch = tokenizer.encode_batch(text_batch, max_length=64)
    decoded_batch = tokenizer.decode_batch(encoded_batch, supress_special_tokens=True)
    
    assert text_batch == decoded_batch

    assert "Hello world!" == tokenizer.decode(tokenizer.encode("Hello world!"))
    assert "<|startoftext|>Hello world!<|endoftext|>" == tokenizer.decode(tokenizer.encode("Hello world!"), supress_special_tokens=False)
    