import torch

from imgcl.config import TRAIN_CONFIG, TRAIN_VAL_DIR, TRAIN_VAL_LABELS_PATH
from imgcl.dataset.dataset_pytorch import ImageDataset
from imgcl.nets.alexnet import Model
from imgcl.training.trainer import Trainer


def main():
    dataset = ImageDataset(TRAIN_VAL_DIR, TRAIN_VAL_LABELS_PATH)
    model = Model(TRAIN_CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG["lr"])
    trainer = Trainer(model, optimizer, dataset, TRAIN_CONFIG)
    trainer.train()


if __name__ == '__main__':
    main()
