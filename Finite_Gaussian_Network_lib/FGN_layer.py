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
    def __init__(self, in_features, out_features, ordinal=2, noisy_centers=False, train_centers=True, random_eval=False):
        super(FGN_layer, self).__init__()
        # input dimension
        self.in_features = in_features
        # output dimension
        self.out_features = out_features
        # ordinal used for the norm (distance to centers) computation
        # (1=diamond, 2=euclidean, 3->float('inf')=approach manhattan)
        self.ordinal = ordinal
        # should noise be added to the centers during training?
        self.noisy_centers = noisy_centers
        # should the centers be trained (if not, will be on zero origin)
        self.train_centers = train_centers
        # noise scale for noisy centers
        self.scale = max(1e-6, np.sqrt(self.in_features)/1000.0)
        # should the eval be random far from the neuron center?
        self.random_eval = random_eval

        
        # learnable parameters
        # regular NN weights (transposed at the start, see order of Tensor(dims))
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        # centers of FGNs
        self.centers = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=self.train_centers)
        # size of FGNs
        self.sigmas = nn.Parameter(torch.Tensor(out_features,), requires_grad=True)
        
        # parameter init call
        self.reset_parameters()
    
    # parameter init definition
    def reset_parameters(self):
        s = np.sqrt(self.in_features)
        # regular NN init
        self.weights.data.uniform_(-s, s)
        # centers init, assuming data normalized to mean 0 var 1
        if self.train_centers:
            s = np.sqrt(self.in_features)
            self.centers.data.normal_(std=0.1)
        else:
            self.centers.data.uniform_(-0,0)
        # size init, to be researched further
        s = np.sqrt(self.in_features)
#         s = self.in_features
#         s = np.log2(self.in_features)
        self.sigmas.data.uniform_(s, s)
        
    def forward(self, input):
        
        ### linear part is the same as normal NNs
        biases = -torch.sum(torch.mul(self.weights, self.centers), dim=-1)
        l = F.linear(input, self.weights, bias=biases)
        # optional, apply tanh here
#         l = torch.tanh(l)

#         ### gaussian component

        # unsqueeze the inputs to allow broadcasting
        # distance to centers
        g = input.unsqueeze(1)-self.centers
        # add noise to centers if required and if in training
        if (self.noisy_centers and self.training):
            c_noise = torch.Tensor(np.random.normal(scale=self.scale, size=self.centers.size()))
            # send to proper device
            c_noise = c_noise.to(next(self.parameters()).device)
            g = g+c_noise
#       # square for euclidean dist
#         g = g**2
        # apply abs if ordinal is odd
        if self.ordinal%2 == 1:
            g = torch.abs(g)
        # raise to ordinal power
        g = g.pow(self.ordinal)
        # sum along axis
        g = g.sum(dim=2)
        # to make it identical to the torch.norm computation below add .pow(1.0/ord), but not really needed
#         g = g.pow(1.0/self.ordinal)

#         # for future, use any norm instead? (adds .pow(1/p) to computation, might be slower, but also might be numpy optimized)
#         ordinal = self.ordinal 
#             g = torch.norm(input.unsqueeze(1)-self.centers, p=ordinal, dim=2)
#         g = g**p # only needed to be identical to above

        # apply sigma
        eps = 1e-3 #minimum sigma
        g = -g/(torch.clamp(self.sigmas, min=eps)**2 )
#         g = -g/self.sigmas**2
        # apply exponential
        g = torch.exp(g)
        # if in eval mode and random eval applied, replace zeros (ie activity far from center) with random [-max,max]
        if (self.random_eval and not self.training):
#             print("hit random spot", self.out_features)
            max_v = torch.max(torch.abs(g)).item()
            # indexes to replace (where the activity is very close to zero)
            zero_inds = (g<=1e-32).float().to(next(self.parameters()).device)
            # indexes to not replace
            nzero_inds = (g>1e-32).float().to(next(self.parameters()).device)
            random_noise = torch.FloatTensor(g.shape).uniform_(-max_v, max_v).to(next(self.parameters()).device)
            g = torch.mul(g, nzero_inds) + torch.mul(random_noise, zero_inds)
        

        ### combine gaussian with linear
        res = l*g
        # optional, flatten res with extra non-linearity
        # res = F.tanh(res)

        return res
    