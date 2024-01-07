import sys
sys.path.append(".")
import argparse
import torch

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from omegaconf import DictConfig
from torchvision import transforms

from src.model import make_CLIP
from src.data.loader import CocoDataset
from src.data.tokenizer import Tokenizer
from src.utils import load_config


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_config", type=Path, required=True, help="Path to config file")
    parser.add_argument("--path_to_ckpt", type=Path, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--n_matricies", type=int, default=5, help="Number of similarity matricies to plot")
    parser.add_argument("--n_pairs", type=int, default=8, help="Number of image-text pairs to plot")
    args = parser.parse_args()

    return args


def initialize_model(config: DictConfig, path_to_checkpoint: Path, vocab_size: int):
    device = torch.device(config.train.device)
    model = make_CLIP(vocab_size=vocab_size, 
                      vision_encoder_name=config.vision_encoder.backbone_name, 
                      vision_encoder_pretrained=config.vision_encoder.pretrained,
                      text_encoder_num_layers=config.text_encoder.num_layers,
                      text_encoder_num_heads=config.text_encoder.num_heads,
                      embed_dim=config.clip_model.embed_dim,
                      sequence_length=config.data.sequence_length,
                      temperature=config.clip_model.temperature).to(device)
    
    state_dict = torch.load(path_to_checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    return model


def save_vision_text_similarities_plots(similarity: np.ndarray, original_images: list, original_captions: list, path_to_save: Path):
    count = len(original_captions)
    plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=0, vmax=1)
    
    plt.yticks(range(count), original_captions, fontsize=15)
    plt.xticks([])
    
    for i, image in enumerate(original_images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=15)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])

    plt.title("Image-text similarities", size=20)
    plt.savefig(path_to_save, bbox_inches="tight")


def prepare_data(config: DictConfig):
    image_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(config.data.image_size, config.data.image_size), 
                          interpolation=transforms.InterpolationMode.BICUBIC,
                          antialias=None),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                             std=(0.26862954, 0.26130258, 0.27577711))
    ])
    
    dataset = CocoDataset(path_to_annotation=config.data.path_to_val_annotation, 
                          path_to_images=config.data.path_to_val_images,
                          transforms=None)
    
    return dataset, image_transforms


def main(args, config):
    device = torch.device(config.train.device)

    tokenizer = Tokenizer(bos_str=config.data.bos_str, 
                          eos_str=config.data.eos_str, 
                          tokenizer_name=config.data.tokenizer_name)
    
    CLIP = initialize_model(config, args.path_to_ckpt, tokenizer.vocab_size)
    CLIP.eval()

    dataset, image_transforms = prepare_data(config)
    
    for sim_matrix_idx in range(args.n_matricies):
        original_images = []
        original_captions = []
        
        samples_indices = np.random.choice(len(dataset), size=args.n_pairs, replace=False)

        for sample_idx in samples_indices:
            original_image, original_caption = dataset[sample_idx]
            
            original_images.append(original_image)
            original_captions.append(original_caption)

        images = torch.stack([image_transforms(image) for image in original_images]).to(device)
        captions, padding_mask = tokenizer.encode_batch(original_captions, max_length=config.data.sequence_length)

        with torch.no_grad():
            images = images.to(device)
            captions = captions.to(device)
            padding_mask = padding_mask.to(device)
            
            vision_embedding = CLIP.vision_adapter(images)
            text_embedding = CLIP.text_adapter(captions, padding_mask=padding_mask)

            vision_embedding /= torch.linalg.norm(vision_embedding, dim=1, ord=2, keepdim=True)
            text_embedding /= torch.linalg.norm(text_embedding, dim=1, ord=2, keepdim=True)
            similarity = torch.matmul(vision_embedding, text_embedding.T)
        
        similarity = similarity.cpu().numpy()

        path_to_save = Path(config.path_to_artifacts, f"similarity_{sim_matrix_idx}.png")
        save_vision_text_similarities_plots(similarity, original_images, original_captions, path_to_save)


if __name__ == "__main__":
    args = read_args()

    config = load_config(args.path_to_config)

    main(args, config)
        