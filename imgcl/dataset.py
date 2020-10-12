import os

import pandas as pd
from skimage import io
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from imgcl.config import IDX_SIZE, ID_COLUMN, LABEL_COLUMN


class ImageDataset(Dataset):
    def __init__(self, data_dir, labels_path=None, is_train=True):
        self.data_dir = data_dir
        self.labels = self._get_labels(labels_path)
        self.is_train = is_train
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0, hue=0),
            # transforms.Resize((96, 96)),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomAffine(10, translate=None, scale=None, shear=None, resample=False, fillcolor=0),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(os.listdir(self.data_dir)) - 1

    def __getitem__(self, idx):
        idx = self._transform_idx(idx)
        img_path = os.path.join(self.data_dir, idx)
        img = io.imread(img_path)

        # if self.is_train:
        img_tensor = self.transforms(img)
        # else:
        #     img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        #     img_tensor = img_tensor / 255 - 0.5
        label = self.labels.get(idx, -1)
        return {"idx": idx, "image": img_tensor.float(), "label": label}

    def _transform_idx(self, idx):
        idx = str(idx)
        dataset = "trainval" if self.labels else "test"  # TODO: (@whiteRa2bit, 2020-09-20) Add to config
        prefix = f"{dataset}_{'0' * (IDX_SIZE - len(idx))}"
        postfix = '.jpg'
        full_idx = prefix + idx + postfix
        return full_idx

    @staticmethod
    def _get_labels(labels_path):
        if labels_path is None:
            labels = {}
        else:
            labels = pd.read_csv(labels_path)
            labels.set_index([ID_COLUMN], inplace=True)
            labels = dict(zip(labels.index, labels[LABEL_COLUMN].values))
        return labels
