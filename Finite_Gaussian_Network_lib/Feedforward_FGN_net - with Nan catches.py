# Feedforward Finite Gaussian Neural Network model

import torch.nn as nn
import torch.nn.functional as F
import torch

from FGN_layer import FGN_layer 

class Feedforward_FGN_net(nn.Module):
    
    def __init__(self,in_feats, out_feats, hidden_l_nums, drop_p=0.0, **kwargs):
        super(Feedforward_FGN_net, self).__init__()
        
        # input dimension
        self.in_feats=in_feats
        # output imension (number of classes)
        self.out_feats=out_feats
        # dropout prob (same throughout network)
        self.drop_p = drop_p
        
        ### kwargs for FGN_layer
#         # ordinal used for the norm (distance to centers) computation
#         # (1=diamond, 2=euclidean, 3->float('inf')=approach manhattan)
#         self.ordinal = ordinal
#         # should noise be added to the centers during training?
#         self.noisy_centers = noisy_centers
#         # should an extra non-linearity be used and if so which one?
#         # can be False or a function on tensors such as torch.tanh()
#         self.non_lin = non_lin
        
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
            # the FGN layer
            self.hidden_layers.append(FGN_layer(next_in, next_out, **kwargs))
            # optional: batchnorm
            self.hidden_layers.append(nn.BatchNorm1d(next_out))
            # optional: dropout layer
            if drop_p > 0:
                self.hidden_layers.append(nn.Dropout(p=self.drop_p))
            # reset feat for next layer
            next_in = next_out

        # final layer, always non-random eval, always non-noisy center (?)
        self.fl = FGN_layer(next_in, self.out_feats, **kwargs)
        # final layer batchnorm
#         self.flb = nn.BatchNorm1d(self.out_feats)
        
    def forward(self, x):
        if (x != x).any(): raise TypeError("x 0 is nan \n {}".format(x))
        # squash the data
        x = x.view(-1, self.in_feats)
        # input batchnorm
        x = self.ib(x)
        if (x != x).any():
            raise TypeError("x 1 is nan \n {}".format(x))
        # for each hidden layer
        for idx, layer in enumerate(self.hidden_layers):
            # apply layer (finite, batchnorm or dropout)
            x = layer(x)
            if (x != x).any(): 
                print("layer id:", idx)
                raise TypeError("x 2 is nan \n {}".format(x))
            # if FGN_layer, apply non-linerarity (only needed to replicate classic net behavior)
#             if isinstance(layer, FGN_layer):
#                 x = torch.tanh(x)
        # final out layer
        x = self.fl(x)
        if (x != x).any():
            print("final layer:", idx)
            raise TypeError("x 3 is nan \n {}".format(x))
        # final layer batchnorm
#         x = self.flb(x)
        # NO softmax
#         x = F.log_softmax(x, dim=-1)
        
        return x
    
    # utility function: sets random eval for hidden layers
    def set_random_eval(self, b):
        
        # set master eval
        self.random_eval = b
        
        # set for all hidden layers
        for layer in self.hidden_layers:
            if isinstance(layer, FGN_layer):
                layer.random_eval = b
        
        # return nothing
        return None
