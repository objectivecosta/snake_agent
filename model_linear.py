import torch.nn as nn
import torch.nn.functional as F


class SnakeLinearNetwork(nn.Module):
    def __init__(self):
        super(SnakeLinearNetwork, self).__init__()

        # Input size will be:
        # [
        #   is blocked leftwards,
        #   is blocked straight,
        #   is blocked rightwards,

        #   is moving left,
        #   is moving up,
        #   is moving right,
        #   is moving down,

        #   is food left,
        #   is food up,
        #   is food right,
        #   is food down,
        # ]


        # Fully connected layers for combined inputs
        self.fc1 = nn.Linear(11, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        # Ensure input x has a batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
