import torch
import torch.nn as nn
import torch.nn.functional as F

from imgcl.nets.abstract_model import AbstractModel


class Model(AbstractModel):
    def __init__(self, config):
        super(Model, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Conv 1
        self.conv1_1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        # self.batchnorm1 = nn.BatchNorm2d(64)
        # Conv 2
        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        # self.batchnorm2 = nn.BatchNorm2d(128)
        # Conv 3
        self.conv3_1 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        # self.batchnorm3 = nn.BatchNorm2d(256)
        # Conv 4
        # self.conv4_1 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        # self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        # self.batchnorm4 = nn.BatchNorm2d(512)
        # Conv 5
        # self.conv5_1 = nn.Conv2d(512, 1024, 3, stride=2)
        # self.conv5_2 = nn.Conv2d(1024, 1024, 3, padding=1)
        # self.conv5_3 = nn.Conv2d(1024, 1024, 3, padding=1)
        # self.batchnorm5 = nn.BatchNorm2d(1024)

        self.classifier = nn.Sequential(
            nn.Linear(256, 200),
            # nn.LeakyReLU(inplace=True),
            # nn.Dropout(config['dropout']),
            # nn.Linear(380, 256),
            # nn.LeakyReLU(inplace=True),
            # nn.Dropout(config['dropout']),
            # nn.Linear(256, 200)
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 9, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(config['dropout']),
        #     nn.Linear(1024, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(config['dropout']),
        #     nn.Linear(256, 200))

    def forward(self, x, debug=False):
        def _debug(x):
            if debug:
                print(x.shape)

        # Conv 1
        _debug(x)
        out1_1 = F.leaky_relu(self.conv1_1(x))
        _debug(out1_1)
        out1_2 = F.leaky_relu(self.conv1_2(out1_1))
        _debug(out1_2)
        out1 = self.batchnorm1(out1_1 + out1_2)
        _debug(out1_2)

        # Conv 2
        out2_1 = F.leaky_relu(self.conv2_1(out1))
        _debug(out2_1)
        out2_2 = F.leaky_relu(self.conv2_2(out2_1))
        _debug(out2_2)
        out2 = self.batchnorm2(out2_1 + out2_2)
        _debug(out2)

        # Conv 3
        out3_1 = F.leaky_relu(self.conv3_1(out2))
        _debug(out3_1)
        out3_2 = F.leaky_relu(self.conv3_2(out3_1))
        _debug(out3_2)
        out3_3 = F.leaky_relu(self.conv3_3(out3_2))
        _debug(out3_2)
        out3 = self.batchnorm3(out3_1 + out3_3)
        _debug(out3)

        # Conv 4
        # out4_1 = F.leaky_relu(self.conv4_1(out3))
        # _debug(out4_1)
        # out4_2 = F.leaky_relu(self.conv4_2(out4_1))
        # _debug(out4_2)
        # out4_3 = F.leaky_relu(self.conv4_3(out4_2))
        # _debug(out4_3)
        # out = self.batchnorm4(out4_1 + out4_3)
        # _debug(out)

        out = self.pool(out3)
        _debug(out)
        out = torch.flatten(out, 1)
        _debug(out)

        # Classifier
        out = self.classifier(out)
        _debug(out)

        return out

    @property
    def name(self):
        return "resnet"
