import torch

from imgcl.config import TRAIN_CONFIG, TRAIN_VAL_DIR, TRAIN_VAL_LABELS_PATH
from imgcl.dataset import ImageDataset
from imgcl.nets.vgg import Model
from imgcl.trainer import Trainer


def main():
    dataset = ImageDataset(TRAIN_VAL_DIR, TRAIN_VAL_LABELS_PATH)
    model = Model(TRAIN_CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG["lr_init"], weight_decay=TRAIN_CONFIG['l2_coef'])
    trainer = Trainer(model, optimizer, dataset, TRAIN_CONFIG)
    trainer.train()


if __name__ == '__main__':
    main()
