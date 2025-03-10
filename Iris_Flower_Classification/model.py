import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_layer=4, h1=8, h2=8, output_layer=3):
        super().__init__()
        self.fc1 = nn.Linear(input_layer, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output_layer)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)  # No activation here, since it's output
        return x