import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MetaLayer
from torch_geometric.nn import global_mean_pool, global_add_pool

from torch_scatter import scatter_mean
from torch_geometric.utils import to_dense_batch

from layers import EdgeModel, NodeModel, GlobalModel
from utils import init_weights, SSP


class GraphNetwork(nn.Module):
    def __init__(self, M, N, n_atom_feats, n_bond_feats, hidden_dim, device):
        super(GraphNetwork, self).__init__()
        
        self.GN_encoder = Encoder(n_atom_feats, n_bond_feats, hidden_dim)
        self.GN_processor = Processor(M, N, EdgeModel(hidden_dim), NodeModel(hidden_dim))
        self.GN_decoder = Decoder(hidden_dim)
        self.reconstruction = Reconstruction(hidden_dim)

        self.M = M
        self.N = N
        self.device = device
        self.hidden_dim = hidden_dim

    def forward(self, g):
        
        x, edge_attr = self.GN_encoder(atom = g.z, edge_attr = g.edge_attr)
        enc_x = x

        x_list, edge_attr = self.GN_processor(x, g.edge_index, edge_attr)
        out = self.GN_decoder(x_proc = x_list, x_enc = enc_x, batch = g.batch)
        
        x_reconstruction = self.reconstruction(x_list)

        return out, x_reconstruction, x_list


#############################################################################################################################
# Building Blocks
# Encode Process Decode Scheme
#############################################################################################################################

class Encoder(nn.Module):
    def __init__(self, n_atom_feats, n_bond_feats, hidden_dim):
        super(Encoder, self).__init__()

        self.embedder = nn.Embedding(90, hidden_dim)
        self.node_encoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.edge_encoder = nn.Sequential(nn.Linear(n_bond_feats, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))        
        self.reset_parameters()
    
    def reset_parameters(self):
        for item in [self.node_encoder, self.edge_encoder]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
    
    def forward(self, atom, edge_attr):
        
        x = self.embedder(atom)
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        return x, edge_attr


class GNBlock(nn.Module):
    def __init__(self, M, edge_model = None, node_model = None):
        super(GNBlock, self).__init__()
        self.edge_model = nn.ModuleList([edge_model for i in range(M)])
        self.node_model = nn.ModuleList([node_model for i in range(M)])
        self.reset_parameters()
        self.M = M

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr):
        row = edge_index[0]
        col = edge_index[0]

        for i in range(self.M):
            edge_attr_ = self.edge_model[i](x[row], x[col], edge_attr)
            x_ = self.node_model[i](x, edge_index, edge_attr_)
            edge_attr = edge_attr + edge_attr_
            x = x + x_
        
        return x, edge_attr


class Processor(nn.Module):
    def __init__(self, M, N, edge_model, node_model):
        super(Processor, self).__init__()
        self.GN_block = nn.ModuleList([GNBlock(M, edge_model, node_model) for i in range(N)])
        self.N = N

    def forward(self, x, edge_index, edge_attr):
        
        x_list = None
        for i in range(self.N):
            x, edge_attr = self.GN_block[i](x, edge_index, edge_attr)
            if x_list == None:
                x_list = x.reshape(1, x.shape[0], x.shape[1])
            else :
                x_list = torch.cat((x_list, x.reshape(1, x.shape[0], x.shape[1])), dim=0)

        return x_list, edge_attr 


class Decoder(nn.Module):
    def __init__(self, hidden_dim):
        super(Decoder, self).__init__()
        self.proc_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), SSP(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), SSP(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), SSP())
        self.enc_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), SSP(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), SSP(), 
                                    nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), SSP())
        self.proc_W = nn.Linear(hidden_dim, 1)
        self.enc_W = nn.Linear(hidden_dim, 1)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.proc_mlp.apply(init_weights)
        self.enc_mlp.apply(init_weights)
        self.proc_W.apply(init_weights)
        self.enc_W.apply(init_weights)
    
    def forward(self, x_proc, x_enc, batch):

        temp_proc_list, temp_enc_list = None, None
        for i in range(x_proc.shape[0]):
            temp_proc = global_add_pool(self.proc_mlp(x_proc)[i], batch)
            temp_enc = global_add_pool(self.enc_mlp(x_enc), batch)
            if temp_proc_list == None:
                temp_proc_list = temp_proc.reshape(1, len(batch.unique()), -1)
                temp_enc_list = temp_enc.reshape(1, len(batch.unique()), -1)
            else:
                temp_proc_list = torch.cat((temp_proc_list, temp_proc.reshape(1, len(batch.unique()), -1)), dim = 0)
                temp_enc_list = torch.cat((temp_enc_list, temp_enc.reshape(1, len(batch.unique()), -1)), dim = 0)
        
        self.proc_W(temp_proc_list)

        target = (self.proc_W(temp_proc_list) + self.enc_W(temp_enc_list)).reshape(-1, len(batch.unique()))
        
        return target


class Reconstruction(nn.Module):
    def __init__(self, hidden_dim):
        super(Reconstruction, self).__init__()
        self.mlp = nn.Linear(hidden_dim, 3)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.mlp.apply(init_weights)

    def forward(self, x_proc):

        preds = self.mlp(x_proc)
        
        return preds
