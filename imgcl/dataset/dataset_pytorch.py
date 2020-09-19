import os

import pandas as pd
from skimage import io
import torch
from torch.utils.data import Dataset

from imgcl.config import IDX_SIZE, ID_COLUMN, LABEL_COLUMN

class ImageDataset(Dataset):
    def __init__(self, data_dir, labels_path):
        self.data_dir = data_dir
        labels = pd.read_csv(labels_path)
        labels.set_index([ID_COLUMN], inplace=True)
        self.labels = dict(zip(labels.index, labels[LABEL_COLUMN].values))
        
    def __len__(self):
        return len(os.listdir(self.data_dir)) - 1

    def __getitem__(self, idx):
        idx = self._transform_idx(idx)
        img_path = os.path.join(self.data_dir, idx)
        img = io.imread(img_path)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        label= self.labels[idx]
        return {"image": img_tensor.float(), "label": label}
    
    @staticmethod
    def _transform_idx(idx):
        idx = str(idx)
        prefix = f"trainval_{'0' * (IDX_SIZE - len(idx))}"
        postfix = '.jpg'
        full_idx = prefix + idx + postfix
        return full_idx    
