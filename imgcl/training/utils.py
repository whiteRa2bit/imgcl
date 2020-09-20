import os

import torch

from imgcl.config import CHECKPOINT_DIR


def _get_checkpoint_path(model, config):
    checkpoint_name = model.name
    for key, value in config.items():
        checkpoint_name += f"_{key}_{value}"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.pt")
    return checkpoint_path


def save_checkpoint(model, config):
    checkpoint_path = _get_checkpoint_path(model, config)
    torch.save(model.state_dict(), checkpoint_path)


def load_checkpoint(model, config):
    checkpoint_path = _get_checkpoint_path(model, config)
    model.load_state_dict(torch.load(checkpoint_path))
    return model
