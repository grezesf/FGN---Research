# Hybrid Feedforward Neural Network model: only last layer is FGN

import torch.nn as nn
import torch.nn.functional as F
import torch

from FGN_layer import FGN_layer 

class Feedforward_Hybrid_Last_net(nn.Module):
    
    def __init__(self,in_feats, out_feats, hidden_l_nums, drop_p=0.0, noisy_centers=False):
        super(Feedforward_Hybrid_Last_net, self).__init__()
        
        # input dimension
        self.in_feats=in_feats
        # output imension (number of classes)
        self.out_feats=out_feats
        # dropout prob (same throughout network)
        self.drop_p = drop_p
        # should noise be added to the centers during training?
        self.noisy_centers = noisy_centers       
        
        # the hidden layers
        # add modules
        self.hidden_layers = nn.ModuleList([])
        
        # optional input dropout
        if drop_p > 0:
            self.hidden_layers.append(nn.Dropout(p=self.drop_p)) 
            
        # input batchnorm
        self.ib = nn.BatchNorm1d(self.in_feats)
        
        # add in the variable layers
        next_in = self.in_feats
        for idx, next_out in enumerate(hidden_l_nums):
            # linear layer
            self.hidden_layers.append(nn.Linear(next_in, next_out))
            # optional: batchnorm
            self.hidden_layers.append(nn.BatchNorm1d(next_out))
            # optional: dropout layer
            if drop_p > 0:
                self.hidden_layers.append(nn.Dropout(p=self.drop_p))
            # reset feat for next layer
            next_in = next_out
        
        # final layer Finite Gaussia
        self.fl = FGN_layer(next_in, self.out_feats, noisy_centers=self.noisy_centers)
        # final layer batchnorm
#         self.flb = nn.BatchNorm1d(self.out_feats)
        
    def forward(self, x):
        # squash the data
        x = x.view(-1, self.in_feats)
        # input batchnorm
        x = self.ib(x)
        # for each hidden layer
        for layer in self.hidden_layers:
            # apply layer (linear, batchnorm or dropout)
            x = layer(x)
            # if linear, apply non-linerarity
#             if isinstance(layer, nn.Linear):
#                 x = torch.tanh(x)
        # final out layer
        x= self.fl(x)
        # final layer batchnorm
#         x = self.flb(x)
        # NO softmax
#         x = F.log_softmax(x, dim=-1)
        
        return x