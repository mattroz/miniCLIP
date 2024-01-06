import argparse
import sys
import logging
import yaml
import json

from omegaconf import OmegaConf
from pathlib import Path


class LoggerColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[0m',   # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[1;37;41m',  # White text on red background
        'RESET': '\033[0m'     # Reset to default
    }

    def format(self, record):
        log_message = super(LoggerColoredFormatter, self).format(record)
        log_level = record.levelname
        return f"{self.COLORS.get(log_level, '')}{log_message}{self.COLORS['RESET']}"


def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def load_config(path: Path):
    config = OmegaConf.load(path)
    return config
    

def fix_random_seeds(random_seed: int = 42):
    import torch
    import random
    import numpy as np
    
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def merge_configs(config: OmegaConf, config_to_merge: argparse.Namespace):
    cfg = vars(config_to_merge)
    # Handle PosiPath objects as they are stored in an ugly way in .yaml
    for key in cfg:
        if isinstance(cfg[key], Path):
            cfg[key] = str(cfg[key].resolve())

    config = OmegaConf.merge(config, OmegaConf.create(vars(config_to_merge)))
    return config


def get_logger(path_to_logs: Path, logger_name: str = __name__):
    """Sets logger parameters.

    Args:
        log_fpath (pathlib.Path): Log file path.
    """
    if not path_to_logs.parent.is_dir():
        path_to_logs.parent.mkdir(parents=True, exist_ok=True)

    log_messages_format = "[%(asctime)-19s] [%(name)-30s:%(funcName)-30s:%(lineno)-5s] [%(levelname)-8s] %(message)s"
    log_date_format = "%Y-%m-%d %H:%M:%S"

    file_handler = logging.FileHandler(filename=str(path_to_logs))
    file_formatter = logging.Formatter(fmt=log_messages_format, datefmt=log_date_format)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    colored_formatter = LoggerColoredFormatter(fmt=log_messages_format, datefmt=log_date_format)
    console_handler.setFormatter(colored_formatter)

    handlers = [file_handler, console_handler]
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    logger = logging.getLogger(logger_name)

    return logger