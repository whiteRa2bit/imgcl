import torch.nn as nn
import torch.nn.functional as F

from imgcl.nets.abstract_model import AbstractModel


class Model(AbstractModel):
    def __init__(self, config):
        super(Model, self).__init__()
        self.pool = nn.AvgPool2d(3, 2)
        # Conv 1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        # Conv 2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(128)
        # Conv 3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(256)
        # Conv 4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(512)


        # self.classifier = nn.Sequential(
        #     nn.Linear(4096, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(config['dropout']),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(config['dropout']),
        #     nn.Linear(512, 200)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(config['dropout']),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config['dropout']),
            nn.Linear(256, 200)
        )
        

    def forward(self, x, debug=False):
        def _debug():
            if debug:
                print(x.shape)

        # Conv 1
        _debug()
        x = F.relu(self.conv1_1(x))
        _debug()
        x = F.relu(self.conv1_2(x))
        _debug()
        x = self.pool(x)
        x = self.batchnorm1(x)
        _debug()

         # Conv 2
        x = F.relu(self.conv2_1(x))
        _debug()
        x = F.relu(self.conv2_2(x))
        _debug()
        x = self.pool(x)
        x = self.batchnorm2(x)
        _debug()


        # Conv 3
        x = F.relu(self.conv3_1(x))
        _debug()
        x = F.relu(self.conv3_2(x))
        _debug()
        x = F.relu(self.conv3_3(x))
        _debug()
        x = self.pool(x)
        x = self.batchnorm3(x)
        _debug()

        # Conv 4
        x = F.relu(self.conv4_1(x))
        _debug()
        x = F.relu(self.conv4_2(x))
        _debug()
        x = F.relu(self.conv4_3(x))
        _debug()
        x = self.pool(x)
        _debug()

        # x = x.view(-1, 256 * 4 * 4)
        x = x.view(-1, 512)
        _debug()

        # Classifier
        x = self.classifier(x)
        _debug()

        return x

    @property
    def name(self):
        return "vgg"
