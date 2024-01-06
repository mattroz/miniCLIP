import sys
sys.path.append(".")

import argparse
import torch
import logging

import numpy as np
import torch.nn as nn

from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader

from omegaconf import OmegaConf, DictConfig
from src.utils import load_config, fix_random_seeds, get_logger, merge_configs
from src.data.loader import CocoDataset
from src.data.tokenizer import Tokenizer
from src.model import make_CLIP


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_config", type=Path, required=True, help="Path to config file")
    parser.add_argument("--path_to_log", type=Path, required=True, help="Path to log file")
    args = parser.parse_args()

    return args


def run_training(config: DictConfig):    
    logger.info(f"Loading tokenizer {config.data.tokenizer_name}")
    tokenizer = Tokenizer(bos_str=config.data.bos_str, 
                          eos_str=config.data.eos_str, 
                          tokenizer_name=config.data.tokenizer_name)
    
    logger.warning(f"Just a random warning")

    logger.info("Initializing CLIP")
    model = make_CLIP(vocab_size=tokenizer.vocab_size, 
                      vision_encoder_name=config.vision_encoder.backbone_name, 
                      text_encoder_num_layers=config.text_encoder.num_layers,
                      text_encoder_num_heads=config.text_encoder.num_heads,
                      embed_dim=config.clip_model.embed_dim,
                      sequence_length=config.data.sequence_length,
                      temperature=config.clip_model.temperature)
    
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config.num_parameters = num_parameters
    
    logger.info(f"CLIP initialized with {num_parameters} parameters")
    logger.error(f"End of program")


if __name__ == "__main__":
    args = read_args()

    experiment_name = f"{args.path_to_config.stem}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    path_to_experiment_dir = Path(args.path_to_log, experiment_name).resolve()
    path_to_experiment_dir.mkdir(parents=True, exist_ok=False)

    path_to_log_file = Path(path_to_experiment_dir, "train.log").resolve()

    config = load_config(args.path_to_config)
    config.experiment_name = experiment_name
    config.path_to_experiment_dir = str(path_to_experiment_dir)

    config = merge_configs(config, args)

    logger = get_logger(path_to_log_file, logger_name="CLIP_training")

    logger.info("Experiment name: %s", config.experiment_name)
    logger.info(f"Config:\n{config}")

    logger.info("Fixing random seeds")
    fix_random_seeds(config.train.random_seed)

    logger.info("Running training")
    run_training(config)

    OmegaConf.save(config, Path(config.path_to_experiment_dir, config.experiment_name + ".yaml"))
    
    
