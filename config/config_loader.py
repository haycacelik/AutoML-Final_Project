from pathlib import Path
from typing import TypedDict, Union

import yaml


class Config(TypedDict):
    seed: int
    token_length: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    data_fraction: float
    val_percentage: float
    fraction_layers_to_finetune: float
    classification_head_hidden_dim: int
    classification_head_dropout_rate: float
    classification_head_hidden_layers: int
    output_path: str
    data_path: str

def load_config(config_path: Union[str, Path]) -> dict:
    """Load config file with basic type checking"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: dict) -> dict:
    """Ensure parameters are in bounds"""

    if not 0 <= config['fraction_layers_to_finetune'] <= 1:
        raise ValueError('fraction_layers_to_finetune must be between 0 and 1')

    return config