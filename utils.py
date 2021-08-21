import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Softplus

import numpy as np
import argparse

from torch_geometric.utils import remove_self_loops

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", '-d', type=int, default=0, help="GPU to use")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of Epochs for training")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size")
    parser.add_argument("--es", type=int, default=100, help="Early Stopping Criteria")
    parser.add_argument('--target', type=int, default= 0, help='Index of target (0~11) for prediction')
    parser.add_argument('--hidden_dim', type=int, default= 512, help='Hidden dimension of the network')
    parser.add_argument('--M', type=int, default= 10, help='Number of Iterations per GNBlock')
    parser.add_argument('--N', type=int, default= 2, help='Number of GNBlocks to use')
    parser.add_argument('--alpha', type=float, default= 0.0, help='weight parameters for autoencoder')
    parser.add_argument('--noise_std', type=float, default= 0.02, help='Gaussian Noise std.')
    
    return parser.parse_args()


def training_config(args):
    train_config = dict()
    for arg in vars(args):
        train_config[arg] = getattr(args,arg)
    return train_config


def get_name(train_config):
    name = ''
    dic = train_config
    config = ["lr", "batch_size", "M", "N", "alpha", "noise_std"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class SSP(Softplus):
    def __init__(self, beta=1, origin=0.5, threshold=20):
        super(SSP, self).__init__(beta, threshold)
        self.origin = origin
        self.sp0 = F.softplus(torch.zeros(1) + self.origin, self.beta, self.threshold).item()

    def forward(self, input):
        return F.softplus(input + self.origin, self.beta, self.threshold) - self.sp0


def rbf(x, mu, radius):

    return ((2/radius)**(1/2))*(torch.sin((mu*np.pi/radius)*x)/x)


def even_samples(n_samples):
    samples = torch.empty(n_samples)

    for i in range(0, n_samples):
        samples[i] = (i + 1)

    return samples


class MyTransform(object):

    def __init__(self, target, n_bond_feats):
        self.target = target
        self.n_bond_feats = n_bond_feats

    def __call__(self, data):
        data.y = data.y[:, self.target]

        radius = 2

        edge_index = (torch.cdist(data.pos, data.pos) < radius).nonzero().T
        edge_index = remove_self_loops(edge_index)[0]

        rbf_means = even_samples(self.n_bond_feats)

        displacements = data.pos[edge_index[0]] - data.pos[edge_index[1]]
        distance = torch.cdist(data.pos, data.pos)[np.asarray(edge_index)].reshape(-1, 1)
        distance = rbf(distance.expand(len(distance), self.n_bond_feats), rbf_means, radius = radius)
        edge_attr = torch.cat((F.normalize(displacements, dim=1), distance), dim = 1)
        
        data.edge_index = edge_index
        data.edge_attr = edge_attr

        return data



class Noise_injection:

    def __init__(self, std = 0.02):
        """
        two simple graph augmentation functions --> "Node feature masking" and "Edge masking"
        Random binary node feature mask following Bernoulli distribution with parameter p_f
        Random binary edge mask following Bernoulli distribution with parameter p_e
        """
        self.std = std
        self.radius = 2

    def _preprocess(self, data, n_bond_feats, device):

        noise = torch.normal(0, self.std, size=(data.pos.shape[0], data.pos.shape[1])).to(device)
        pos = data.pos.clone()
        pos = noise + pos

        # Create Edges based on noisy nodes
        edge_index = (torch.cdist(pos, pos) < self.radius).nonzero().T
        edge_index = edge_index.T[data.batch[edge_index][0] == data.batch[edge_index][1]].T
        edge_index = remove_self_loops(edge_index)[0]

        rbf_means = even_samples(n_bond_feats).to(device)

        displacements = pos[edge_index[0]] - pos[edge_index[1]]
        distance = torch.cdist(pos, pos)[np.asarray(edge_index.cpu())].reshape(-1, 1)
        distance = rbf(distance.expand(len(distance), n_bond_feats), rbf_means, radius = self.radius)
        edge_attr = torch.cat((F.normalize(displacements, dim=1), distance), dim = 1)

        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.noise = noise

        return data

    def __call__(self, data):
        
        return self._preprocess(data)