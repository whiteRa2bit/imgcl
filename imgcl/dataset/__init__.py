import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 64, 3)

        self.conv2_drop = nn.Dropout2d(0.3)
        self.drop = nn.Dropout(0.2)

        self.fc1 = nn.Linear(64 * 3 * 3, 400)
        self.fc2 = nn.Linear(400, 200)

    def forward(self, x, debug=False):
        def _debug():
            if debug:
                print(x.shape)

        x = F.relu(self.conv1(x))
        _debug()
        x = self.pool(x)
        _debug()
        x = F.relu(self.conv2(x))
        _debug()
        x = self.pool(x)
        _debug()
        x = self.conv2_drop(x)
        _debug()
        x = F.relu(self.conv3(x))
        _debug()
        x = self.pool(x)
        _debug()
        x = self.conv2_drop(x)
        x = x.view(-1, 64 * 3 * 3)
        _debug()
        x = F.relu(self.fc1(x))
        _debug()
        x = self.drop(x)
        _debug()
        x = self.fc2(x)
        _debug()
        return x
