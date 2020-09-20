import torch.nn as nn
import torch.nn.functional as F

from imgcl.nets.abstract_model import AbstractModel


class Model(AbstractModel):
    def __init__(self, config):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 32, 5)
        self.conv3 = nn.Conv2d(32, 128, 5)

        self.drop = nn.Dropout(config['dropout'])
        self.batchnorm_conv2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 200)

    def forward(self, x, debug=False):
        def _debug():
            if debug:
                print(x.shape)

        _debug()
        x = F.relu(self.conv1(x))
        _debug()
        x = self.pool(x)
        _debug()
        x = F.relu(self.conv2(x))
        _debug()
        x = self.batchnorm_conv2(x)
        _debug()
        x = F.relu(self.conv3(x))
        _debug()
        x = self.pool(x)
        _debug()

        x = x.view(-1, 128 * 5 * 5)
        _debug()
        x = F.relu(self.fc1(x))
        _debug()
        x = F.relu(self.fc2(x))
        _debug()
        x = self.fc3(x)
        _debug()
        return x

    @property
    def name(self):
        return "alexnet"
