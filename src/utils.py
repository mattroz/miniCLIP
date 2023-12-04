import yaml
import json

from pathlib import Path


def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)