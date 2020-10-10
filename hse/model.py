import torch.nn as nn
import torch.nn.functional as F

from abstract_model import AbstractModel


class Model(AbstractModel):
    def __init__(self, config):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(3, 2)
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 128, 3, padding=1)

        self.drop = nn.Dropout(config['dropout'])
        self.batchnorm_conv2 = nn.BatchNorm2d(64)
        self.batchnorm_conv4 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(128 * 4 * 4, 800)
        self.fc2 = nn.Linear(800, 512)
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
        x = F.relu(self.conv4(x))
        _debug()
        x = self.pool(x)
        _debug()
        x = self.batchnorm_conv4(x)
        _debug()
        x = F.relu(self.conv5(x))
        _debug()
        x = self.pool(x)
        _debug()

        x = x.view(-1, 128 * 4 * 4)
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
