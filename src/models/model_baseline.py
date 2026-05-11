import torch
import torch.nn as nn

class BaselineLogisticRegression(nn.Module):
    def __init__(self, input_dim=28*28, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)
        x = x.view(x.size(0), -1)  # flatten to (batch_size, 784)
        return self.linear(x)