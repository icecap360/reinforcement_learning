import torch.nn as nn
import torch
import numpy as np

class ConvModel(nn.Module):
    def __init__(self, img_size, n_action):
        super().__init__()
        self.img_size = img_size
        self.n_actions = n_action
        self.n_blocks = 4
        self.blocks = [
            Block(3, 32), # 128
            Block(32, 48), # 64
            Block(48, 64), # 32
            Block(64, 96), # 16
            Block(96, 128), # 8
        ]
        self.flatten = nn.AvgPool2d(8)
        self.last_layer = nn.Sequential(
            nn.Linear(
                    128, self.n_actions
                ),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.blocks(x)
        x = self.flatten(x)
        x = self.last_layer(x)
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.GroupNorm(8),
            nn.MaxPool2d(2, 2)
        )
    def forward(self, x):
        return self.layers(x)
    