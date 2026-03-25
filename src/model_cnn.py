import torch
import torch.nn as nn
import torch.nn.functional as F

class CharacterCNN(nn.Module):
    def __init__(self, num_classes):
        super(CharacterCNN, self).__init__()

        # Convolution block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolution block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Convolution block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Fully connected classifier
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Input shape: [batch_size, 1, 28, 28]

        x = self.pool(F.relu(self.conv1(x)))   # -> [batch, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))   # -> [batch, 64, 7, 7]
        x = self.pool(F.relu(self.conv3(x)))   # -> [batch, 128, 3, 3]

        x = torch.flatten(x, 1)                # -> [batch, 128*3*3]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                        # -> [batch, num_classes]

        return x