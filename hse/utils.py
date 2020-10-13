import os

from config import CHECKPOINT_DIR


def get_checkpoint_path(model, config):
    checkpoint_name = model.name
    for key, value in config.items():
        checkpoint_name += f"_{key}_{value}"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.pt")
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    return checkpoint_path
