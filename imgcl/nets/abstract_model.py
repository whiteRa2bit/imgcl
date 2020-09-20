import os
from abc import abstractmethod, ABC

import torch.nn as nn


class AbstractModel(ABC, nn.Module):
    def __init__(self, config=None):
        super(AbstractModel, self).__init__()
        self.config = config

    @abstractmethod
    def forward(self, x, debug=False):
        pass

    @property
    @abstractmethod
    def name(self):
        pass
