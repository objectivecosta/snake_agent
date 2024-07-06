import torch.nn as nn
import torch.nn.functional as F


class SnakeNetwork(nn.Module):
    def __init__(self, grid_width, grid_height):
        super(SnakeNetwork, self).__init__()

        # Convolutional layers for multichannel input
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)

        # Fully connected layers for combined inputs
        self.fc1 = nn.Linear(32 * grid_width * grid_height, 384)
        self.fc2 = nn.Linear(384, 4)

    def forward(self, x):
        # Ensure input x has a batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Process the multichannel input through conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Verify the shape is (batch_size, 2, 48w, 48h)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the output to (batch_size, 2 * 48 * 48)

        # Concatenate the conv layer output with valid directions
        # x = torch.cat((x, valid_directions), dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
