import torch.nn as nn
import torch
import numpy as np

class QFunctionConv(nn.Module):
    def __init__(self, img_size, n_action):
        super().__init__()
        self.img_size = img_size
        self.n_actions = n_action
        self.n_blocks = 4
        self.blocks = nn.Sequential(
            Block(3, 32, norm_groups=1), # 128
            Block(32, 48), # 64
            Block(48, 64), # 32
            Block(64, 96), # 16
            Block(96, 128) # 8
        )
        self.flatten = nn.AvgPool2d(8)
        self.last_layer = nn.Sequential(
            nn.Linear(
                    128, self.n_actions
                ),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = torch.moveaxis(x, 3, 1)
        x = self.blocks(x)
        x = self.flatten(x)
        x = self.last_layer(x)
        return x

class MainModel:
    def __init__(self, q_function, discount = 1.0, device='cuda'):
        self.q_function = q_function
        self.discount = discount
        self.loss = nn.MSELoss()
        self.device = device
    
    def calculate_loss(self, replays):
        s_ts = torch.stack([r.s_t for r in replays]).to(self.device)
        r_ts = torch.stack([r.r_t for r in replays]).to(self.device)
        a_ts =  torch.stack([r.a_t for r in replays]).to(self.device)
        with torch.no_grad():
            target = r_ts + self.discount * torch.max(self.q_function(s_ts), 1)
        pred = self.q_function(s_ts)
        loss = self.loss(pred[a_ts], target)
        return loss
    
    def sample_action(self, s_t):
        s_t = torch.tensor(s_t, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            pred = self.q_function(s_t)
        a_t = torch.argmax(pred, 1).squeeze(0).item()
        return a_t 
    def to(self, device):
        self.q_function = self.q_function.to(device)



class Block(nn.Module):
    def __init__(self, in_channels, out_channels, norm_groups=8, down_sample=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.GroupNorm(norm_groups, out_channels),
            nn.MaxPool2d(2, 2)
        )
    def forward(self, x):
        return self.layers(x)
    