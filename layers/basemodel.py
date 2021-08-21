import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean, scatter_sum
from utils import init_weights, SSP


class EdgeModel(nn.Module):
    def __init__(self, hidden_dim):
        super(EdgeModel, self).__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(hidden_dim*3, hidden_dim), nn.LayerNorm(hidden_dim), SSP(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), SSP(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), SSP())
        self.reset_parameters()

    def reset_parameters(self):
        self.edge_mlp.apply(init_weights)

    def forward(self, src, dest, edge_attr):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)


class NodeModel(nn.Module):
    def __init__(self, hidden_dim):
        super(NodeModel, self).__init__()
        self.node_mlp = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.LayerNorm(hidden_dim), SSP(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), SSP(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), SSP())
        self.reset_parameters()

    def reset_parameters(self):
        self.node_mlp.apply(init_weights)

    def forward(self, x, edge_index, edge_attr):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        # torch_scatter.scatter_mean(src, index, dim=-1, out=None, dim_size=None, fill_value=0)
        # averages all values from src into out at the indices specified in the index
        out = scatter_sum(edge_attr, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp(out)


class GlobalModel(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(GlobalModel, self).__init__()
        self.global_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.reset_parameters()
        
    def reset_parameters(self):
        self.global_mlp.apply(init_weights)

    def forward(self, x, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = scatter_sum(x, batch, dim=0)
        return self.global_mlp(out)