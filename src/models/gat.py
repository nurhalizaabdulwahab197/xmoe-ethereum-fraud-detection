import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gat1 = GATConv(input_dim, 32, heads=4, dropout=0.5)
        self.gat2 = GATConv(32 * 4, 32, heads=1, concat=False, dropout=0.5)
        self.fc = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        return self.fc(x)

