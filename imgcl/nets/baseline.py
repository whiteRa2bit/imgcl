import torch.nn as nn
import torch.nn.functional as F

from imgcl.nets.abstract_model import AbstractModel


class Model(AbstractModel):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 7 * 7, 496)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(496, 200)

    def forward(self, x, debug=False):
        def _debug():
            if debug:
                print(x.shape)

        x = F.relu(self.conv1(x))
        _debug()
        x = self.pool(x)
        _debug()
        x = self.pool(F.relu(self.conv2(x)))
        _debug()
        x = x.view(-1, 16 * 7 * 7)
        _debug()
        x = F.relu(self.fc1(x))
        _debug()
        x = self.fc2(x)
        _debug()
        return x

    @property
    def name(self):
        return "baseline"
