"""
gnn_model.py
============
minimal graph convolutional network for connectivity-based scenario
classification (stage 3 of approach b).

architecture choice: deliberately small. with n=9 subjects we cannot afford
expressive models. the goal of stage 3 is to validate whether network-level
features are sufficient for discrimination, not to maximise accuracy.

input per epoch:
  - nodes:   6 rois
  - node features:  per-roi band power summary, length=in_dim
  - edges:   complete graph over the 6 rois, edge weight = wpli value
            for one band (default: theta)
output: 15-class logits (control_action_matched + 14 scenarios)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class MinimalGCN(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 16,
                 n_classes: int = 15, dropout: float = 0.5):
        super().__init__()
        #very small hidden dim. with n=9 we must avoid overfit
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x, edge_index, edge_weight, batch):
        #node-level message passing
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)

        #graph-level pooling
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = MinimalGCN(in_dim=4, hidden=16, n_classes=15, dropout=0.5)
    n = count_parameters(m)
    print(f"MinimalGCN parameters: {n}")
