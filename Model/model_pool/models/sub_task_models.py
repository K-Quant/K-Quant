import torch
from torch import nn


class regression_layer(nn.Module):
    def __init__(self, seq_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred_layer = nn.Linear(seq_len, 1)

    def forward(self, x):
        return self.pred_layer(x).squeeze()  # shape [B,1]


class ranking_layer(nn.Module):
    def __init__(self, seq_len, num_group, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred_layer = nn.Linear(seq_len, num_group)

    def forward(self, x):
        return self.pred_layer(x)  # shape [B, N]
