from .clip import CLIP
from .text_encoder import TextEncoderV1
from .vision_encoder import VisionEncoderV1


def make_CLIP(
        vocab_size, 
        vision_encoder_name="resnet50", 
        embed_dim=512,
        sequence_length=16,
        text_encoder_num_layers=8,
        text_encoder_num_heads=8,
        temperature=0.07,
        vision_encoder_pretrained=False
):
    vision_encoder = VisionEncoderV1(model_name=vision_encoder_name, 
                                     out_features=embed_dim, 
                                     pretrained=vision_encoder_pretrained)
    
    text_encoder = TextEncoderV1(vocab_size=vocab_size, 
                                 d_model=embed_dim, 
                                 n_layers=text_encoder_num_layers, 
                                 n_heads=text_encoder_num_heads, 
                                 sequence_length=sequence_length)

    clip = CLIP(vision_encoder, 
                text_encoder, 
                text_embed_dim=embed_dim, 
                vision_embed_dim=embed_dim, 
                embed_dim=embed_dim, 
                temperature=temperature)

    return clip