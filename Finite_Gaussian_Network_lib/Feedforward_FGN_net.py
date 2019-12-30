# Feedforward Finite Gaussian Neural Network model

import torch.nn as nn
import torch.nn.functional as F
import torch

from FGN_layer import FGN_layer 

class Feedforward_FGN_net(nn.Module):
    
    def __init__(self, in_feats, out_feats, hidden_layer_sizes, drop_p=0.0, **kwargs):
        super(Feedforward_FGN_net, self).__init__()
        
        # input dimension
        self.in_feats=in_feats
        # output imension (number of classes)
        self.out_feats=out_feats
        # dropout prob (same throughout network)
        self.drop_p = drop_p
        # the number of neurons for each hidden layer (can be empty)
#         hidden_layer_sizes
        
        ### kwargs for FGN_layer
#         # covar_type: one of 'sphere', 'diag' or 'full'
#         # ordinal: used for the norm (distance to centers) computation
#         # (1=diamond, 2=euclidean, 3->float('inf')=approach manhattan) (only for sphere covar)
#         # noisy_centers: should noise be added to the centers during training?
#         # non_lin: should an extra non-linearity be used and if so which one?
#         # can be False or a function on tensors such as torch.tanh()
        
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
        for idx, next_out in enumerate(hidden_layer_sizes):
            # the FGN layer
            self.hidden_layers.append(FGN_layer(next_in, next_out, **kwargs))
            # optional: batchnorm
            self.hidden_layers.append(nn.BatchNorm1d(next_out))
            # optional: dropout layer
            if drop_p > 0:
                self.hidden_layers.append(nn.Dropout(p=self.drop_p))
            # reset feat for next layer
            next_in = next_out

        # final layer, should always be non-random eval
        # should never have non-lin since passed to softmax
        kwargs.pop('non_lin', 'None')
        self.fl = FGN_layer(next_in, self.out_feats, non_lin=False, **kwargs)
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
        
        # notice this doesn't change final layer
        
        # return nothing
        return None
    
    def set_first_layer_centers(self, data_loader):
        # based on data (points in space, variance)
        # sets the centers. sigmas unchanged
        # works best with random sampler, assumes nothing in between input and first layer
        # returns nothing
        
        # number of neurons in first FGN layer
        for l in self.hidden_layers:
            if isinstance(l, FGN_layer):
                first_layer_size = l.out_features
                break
        # size of data loader batches
        batch_size = data_loader.batch_size
        count = 0
        
        with torch.no_grad():
            while count<first_layer_size:
                # load next batch samples
                batch_samples = next(iter(data_loader))[0]
                # number of samples from the batch to load
                num_to_load = min(count+batch_size,first_layer_size)-count
                l.centers[count:count+num_to_load] = torch.nn.Parameter(batch_samples[:num_to_load],  requires_grad=True)

                count += batch_size
        
        return None