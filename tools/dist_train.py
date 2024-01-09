import os
import sys
sys.path.append(".")
import cv2

import argparse
import torch
import logging

import numpy as np
import torch.nn as nn

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from omegaconf import OmegaConf, DictConfig
from src.utils import load_config, fix_random_seeds, get_logger, merge_configs
from src.data.loader import CocoDataset
from src.data.tokenizer import Tokenizer
from src.model import make_CLIP
from src.loss import ContrastiveCrossEntropy


torch.autograd.set_detect_anomaly(True)


def ddp_setup():
    init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_config", type=Path, required=True, help="Path to config file")
    parser.add_argument("--path_to_log", type=Path, required=True, help="Path to log file")
    args = parser.parse_args()

    return args


def run_training(config: DictConfig):
    ddp_setup()

    rank = int(os.environ["LOCAL_RANK"])

    # logger.info(f"Loading tokenizer {config.data.tokenizer_name}")
    tokenizer = Tokenizer(bos_str=config.data.bos_str, 
                          eos_str=config.data.eos_str, 
                          tokenizer_name=config.data.tokenizer_name)

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=config.data.image_size, 
                          interpolation=transforms.InterpolationMode.BICUBIC,
                          antialias=None),
        transforms.CenterCrop(size=(config.data.image_size, config.data.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                             std=(0.26862954, 0.26130258, 0.27577711))
    ])

    # logger.info("Initializing COCO dataset")
    train_dataset = CocoDataset(path_to_annotation=config.data.path_to_train_annotation, 
                                path_to_images=config.data.path_to_train_images, 
                                transforms=train_transforms)

    val_dataset = CocoDataset(path_to_annotation=config.data.path_to_val_annotation, 
                              path_to_images=config.data.path_to_val_images,
                              transforms=None)

    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=config.train.batch_size, 
                                  shuffle=False,
                                  pin_memory=True,
                                  sampler=ElasticDistributedSampler(train_dataset),
                                  num_workers=config.train.num_workers)
    
    val_dataloader = DataLoader(dataset=val_dataset, 
                                batch_size=config.train.batch_size, 
                                shuffle=False, 
                                num_workers=config.train.num_workers)
    
    # logger.info(f"Train dataset contains {len(train_dataset)} samples")
    # logger.info(f"Val dataset contains {len(val_dataset)} samples")

    # logger.info("Initializing CLIP")
    model = make_CLIP(vocab_size=tokenizer.vocab_size, 
                      vision_encoder_name=config.vision_encoder.backbone_name, 
                      vision_encoder_pretrained=config.vision_encoder.pretrained,
                      text_encoder_num_layers=config.text_encoder.num_layers,
                      text_encoder_num_heads=config.text_encoder.num_heads,
                      embed_dim=config.clip_model.embed_dim,
                      sequence_length=config.data.sequence_length,
                      temperature=config.clip_model.temperature).to(rank)
    model = DDP(model, device_ids=[rank])
    
    if "clip_grad_norm" in config.train:
        # logger.info(f"Clip gradient norm is enabled, clipping at {config.train.clip_grad_norm}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.clip_grad_norm)
    
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config.num_parameters = num_parameters

    # logger.info(f"Model size: {num_parameters} parameters")

    # logger.info(f"Initializing optimizer [AdamW, lr={config.train.learning_rate}, weight_decay={config.train.weight_decay}]")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)

    # logger.info("Initializing loss function")
    criterion = ContrastiveCrossEntropy()

    # logger.info(f"Initializing learning rate scheduler [CosineAnnealingLR, T_max={config.train.num_epochs * len(train_dataloader)}]")
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.num_epochs * len(train_dataloader))

    # logger.info(f"Starting training for {config.train.num_epochs} epochs")

    train_pbar = tqdm(train_dataloader, total=len(train_dataloader), desc="training", position=rank)
    for epoch in range(config.train.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.train.num_epochs}")
        
        train_dataloader.sampler.set_epoch(epoch)
        
        train_loss = []
        model.train()

        for batch_idx, (images, captions) in enumerate(train_pbar):
            captions, padding_mask = tokenizer.encode_batch(captions, max_length=config.data.sequence_length)
            
            images = images.to(rank)
            captions = captions.to(rank)
            padding_mask = padding_mask.to(rank)
            
            optimizer.zero_grad()

            vision_text_logits = model(images, captions, padding_mask=padding_mask)

            loss = criterion(vision_text_logits)

            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            current_lr = optimizer.param_groups[0]["lr"]

            pbar_desc = f"loss: {np.mean(train_loss):.5f} | " + \
                        f"lr: {current_lr:.8f}"             
            train_pbar.set_description(pbar_desc)

        logger.info(f"train loss @ epoch {epoch}: {np.mean(train_loss):.5f}")
        lr_scheduler.step()

        if (epoch + 1) % config.train.validate_every_n_epochs == 0 and rank == 0:
            logger.info(f"Running validation at {epoch + 1} epoch")
            
            val_pbar = tqdm(val_dataloader, total=len(val_dataloader), desc="validating", position=rank)    
            val_loss = []
            model.eval()

            for batch_idx, (images, captions) in enumerate(val_pbar):
                captions, padding_mask = tokenizer.encode_batch(captions, max_length=config.data.sequence_length)
                
                images = images.to(rank)
                captions = captions.to(rank)
                padding_mask = padding_mask.to(rank)
                
                with torch.no_grad():
                    vision_text_logits = model(images, captions, padding_mask=padding_mask)
                    loss = criterion(vision_text_logits)
                
                val_loss.append(loss.item())

                pbar_desc = f"loss: {np.mean(val_loss):.5f}"
                val_pbar.set_description(pbar_desc)

            logger.info(f"val loss @ epoch {epoch}: {np.mean(val_loss):.5f}")

        if (epoch + 1) % config.train.save_every_n_epochs == 0 and rank == 0:
            path_to_save_ckpt = Path(config.path_to_checkpoints, f"{config.experiment_name}_epoch_{epoch + 1}.pth")
            logger.info(f"Saving model to {str(path_to_save_ckpt)}")
            torch.save(model.state_dict(), path_to_save_ckpt)
    
    destroy_process_group()
    

if __name__ == "__main__":
    args = read_args()

    experiment_name = f"{args.path_to_config.stem}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    path_to_experiment_dir = Path(args.path_to_log, experiment_name).resolve()
    path_to_artifacts = Path(path_to_experiment_dir, "artifacts").resolve()
    path_to_checkpoints = Path(path_to_experiment_dir, "checkpoints").resolve()
    path_to_experiment_dir.mkdir(parents=True, exist_ok=True)
    path_to_artifacts.mkdir(parents=True, exist_ok=True)
    path_to_checkpoints.mkdir(parents=True, exist_ok=True)

    path_to_log_file = Path(path_to_experiment_dir, "train.log").resolve()

    config = load_config(args.path_to_config)
    config.experiment_name = experiment_name
    config.path_to_experiment_dir = str(path_to_experiment_dir)
    config.path_to_artifacts = str(path_to_artifacts)
    config.path_to_checkpoints = str(path_to_checkpoints)
    path_to_save_config = Path(config.path_to_experiment_dir, config.experiment_name + ".yaml")

    config = merge_configs(config, args)

    logger = get_logger(path_to_log_file, logger_name="CLIP_training")

    logger.info("Experiment name: %s", config.experiment_name)
    logger.info(f"Config:\n{config}")

    logger.info("Fixing random seeds")
    fix_random_seeds(config.train.random_seed)
    
    logger.info("Running training")
    run_training(config)
    # try:
    #     run_training(config)
    # except Exception as e:
    #     logger.error(e)
    #     logger.info(f"Saving config to {path_to_save_config}")
    #     OmegaConf.save(config, path_to_save_config)
    #     destroy_process_group()
    #     raise e

    logger.info(f"Saving config to {path_to_save_config}")
    OmegaConf.save(config, path_to_save_config)
    
    
