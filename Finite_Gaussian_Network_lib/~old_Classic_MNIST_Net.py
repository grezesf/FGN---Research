# classical feed forward model with variable number of hidden layers and units per layer for MNIST only

import torch.nn as nn
import torch.nn.functional as F
import torch

class Classic_MNIST_Net(nn.Module):
    
    def __init__(self, hidden_l_nums):
        super(Classic_MNIST_Net, self).__init__()
        
        # the hidden layers
        in_feats=28*28
        # add modules
        self.hidden_layers = nn.ModuleList([])
        for idx, out_feats in enumerate(hidden_l_nums):
            self.hidden_layers.append(nn.Linear(in_feats, out_feats))
            in_feats = out_feats
        
        # final layer
        self.fl = nn.Linear(in_feats, 10)
        
    def forward(self, x):
        # squash the data
        x = x.view(-1, 28*28)
        # for each hidden layer
        for l in self.hidden_layers:
            # apply linear
            x = l(x)
            # apply non-linerarity
            x = torch.tanh(x)
            # not needed: batchnorm
        # final out layer
        x = self.fl(x)
        # softmax
        x = F.log_softmax(x, dim=-1)
        
        return x