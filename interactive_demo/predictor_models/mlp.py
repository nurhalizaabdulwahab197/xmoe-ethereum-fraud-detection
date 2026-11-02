import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Feature-only baseline used in the manuscript clean-protocol comparison."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        hidden_channels = 64
        self.dropout = 0.5
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, out_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # edge_index is accepted for runner compatibility but is intentionally unused.
        del edge_index
        x = self.lin1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)
