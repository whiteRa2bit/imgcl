import os

import torch

from imgcl.config import CHECKPOINT_DIR


def save_checkpoint(model, config):
    checkpoint_name = "model"
    for key, value in config.items():
        checkpoint_name += f"_{key}_{value}"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.pt")
    torch.save(model.state_dict(), checkpoint_path)
