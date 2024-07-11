import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SnakeCNNNetwork(nn.Module):
    def __init__(self, grid_width, grid_height):
        super(SnakeCNNNetwork, self).__init__()

        # Convolutional layers for multichannel input
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Fully connected layers for combined inputs

        # Combined inputs will be the CNN results + the linear results.
        self.fc1 = nn.Linear(32 * grid_width * grid_height + 11, 384)
        self.fc2 = nn.Linear(384, 3)

    def forward(self, x, linear_state):
        # Ensure input x has a batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Process the multichannel input through conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Verify the shape is (batch_size, 2, 48w, 48h)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the output to (batch_size, 2 * grid_w * grid_h + 11)

        if linear_state.dim() == 1:
            linear_state = linear_state.unsqueeze(0)

        # Concatenate the conv layer output with valid directions
        x = torch.cat((x, linear_state), dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def write_to_disk(self):
        model_folder_path = './model'
        file_name = Path(__file__).stem
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, "{}.pth".format(file_name))
        torch.save(self.state_dict(), file_name)

    def read_from_disk(self):
        model_folder_path = './model'
        file_name = Path(__file__).stem

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, "{}.pth".format(file_name))

        self.load_state_dict(torch.load(file_name))
        self.eval()
