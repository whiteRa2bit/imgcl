import torch

from imgcl.config import TRAIN_VAL_DIR, TRAIN_VAL_LABELS_PATH
from imgcl.dataset.dataset_pytorch import ImageDataset
from imgcl.nets.baseline import Model
from imgcl.training.trainer import Trainer


def main():
    config = {
        "lr": 3e-4,
        "epochs_num": 100,
        "log_each": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "train_batch_size": 64,
        "val_batch_size": 5000
    }

    dataset = ImageDataset(TRAIN_VAL_DIR, TRAIN_VAL_LABELS_PATH)
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    trainer = Trainer(model, optimizer, dataset, config)
    trainer.train()

if __name__ == '__main__':
    main()
