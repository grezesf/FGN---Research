# define the FGN layer class to dev

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class FGN_layer(nn.Module):
    r""" Applies a Finite Gaussian Neuron layer to the incoming data
    
    Args:
    
    Shape:
    
    Attributes:
    
    Examples:
        
        >>> l=FGN_layer(20,30)
    
    """
    def __init__(self, in_features, out_features):
        super(FGN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # learnable parameters
        # regular NN weights (transposed at the start, see order of Tensor(dims))
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad = True)
        # centers of FGNs
        self.centers = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad = True)
        # size of FGNs
        self.sigmas = nn.Parameter(torch.Tensor(out_features,), requires_grad = True)   
        
        # parameter init call
        self.reset_parameters()
    
    # parameter init definition
    def reset_parameters(self):
        s = np.sqrt(self.in_features)
        # regular NN init
        self.weights.data.uniform_(-s, s)
        # centers init, assuming data normalized to mean 0 var 1
        self.centers.data.normal_()
        # size init, to be researched further
#         s = np.sqrt(self.in_features)
#         s = self.in_features
        s = np.log2(self.in_features)
        self.sigmas.data.uniform_(s, s)
        
    def forward(self, input):
        
        # linear part is the same as normal NNs
        biases = -torch.sum(torch.mul(self.weights, self.centers), dim=-1)
        l = F.linear(input, self.weights, bias=biases)
        # optional, apply tanh here
        # l = torch.tanh(l)

        # gaussian component
        # unsqueeze the inputs to allow broadcasting
        # compute distance to centers
        g = (input.unsqueeze(1)-self.centers)**2
        g = g.sum(dim=2)

        # for future, use any norm instead?
#         g = torch.norm(input.unsqueeze(1)-self.centers, p=1, dim=2)

        # apply sigma
        eps = 1e-3 #minimum sigma
        g = -g/(torch.clamp(self.sigmas, min=eps)**2 )
#         g = -g/self.sigmas**2
        # apply exponential
        g = torch.exp(g)

        # combine gaussian with linear
        res = l*g
        # optional, flatten res
        # res = F.tanh(res)

        return res
    