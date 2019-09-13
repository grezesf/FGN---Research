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
    def __init__(self, in_features, out_features, 
                 covar_type='diag', ordinal=2.0, noisy_centers=False, non_lin=False, 
                 train_centers=True, **kwargs):
        super(FGN_layer, self).__init__()
        # input dimension
        self.in_features = in_features
        # output dimension
        self.out_features = out_features
        # covariance type for the gaussian: one of 'sphere', 'diag', 'full'
        self.covar_type = covar_type
        # ordinal used for the norm (distance to centers) computation 
        # (1=diamond, 2=euclidean, 3->float('inf')=approach manhattan)
        # right now only for 'sphere' covar type
        self.ordinal = ordinal
        # should an extra non-linearity be used and if so which one?
        # can be False or a function on tensors such as torch.tanh()
        self.non_lin = non_lin
        # should noise be added to the centers during training?
        self.noisy_centers = noisy_centers
        # should the centers be trained (if not, will be on zero origin)
        self.train_centers = train_centers
        # noise scale for noisy centers
        self.scale = max(1e-6, np.sqrt(self.in_features)/1000.0)
        # random_eval is False by default
        self.random_eval = False

        
        # learnable parameters and related
        # regular NN weights (transposed at the start, see order of Tensor(dims))
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        # centers of FGNs
        self.centers = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=self.train_centers)
        # size/range of FGNs
        if covar_type == 'sphere':
            self.sigmas = nn.Parameter(torch.Tensor(out_features,), requires_grad=True)
#             self.inv_covar = nn.Parameter(torch.Tensor(out_features,), requires_grad=True)
        elif covar_type == 'diag':
            self.sigmas = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
#             self.inv_covar = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        elif covar_type == 'full':
            self.sigmas = nn.Parameter(torch.Tensor(out_features, in_features, in_features,), requires_grad=True)
#             self.inv_covar = nn.Parameter(torch.Tensor(out_features, in_features, in_features,), requires_grad=True)
        else:
            # error
            raise TypeError("covar_type not one of ['sphere', 'diag', 'full']")
        # minimum sigma
        self.eps = 1e-8 
        # inverse covariance, which will be actually used
        self.inv_covar = nn.Parameter(0.0*torch.Tensor(self.sigmas), requires_grad=True)

        # parameter init call
        self.reset_parameters()
    
    # parameter init definition
    def reset_parameters(self):
        s = np.sqrt(self.in_features)
        # regular NN init
        self.weights.data.uniform_(-s, s)
        # centers init, assuming data normalized to mean 0 var 1 (not necessarily   true after first layer)
        if self.train_centers:
            s = np.sqrt(self.in_features)
            self.centers.data.normal_(std=0.1)
        else:
            self.centers.data.uniform_(-0,0)
        # sigmas init, to be researched further
#         s = np.sqrt(self.in_features)
        s = self.in_features
#         s = np.log2(self.in_features)

        if self.covar_type in ['sphere', 'diag']:
            self.sigmas.data.uniform_(s-0.5, s+0.5)
            self.inv_covar.data.copy_(1.0/self.sigmas)
        elif self.covar_type == 'full':
            # self.covar_type == 'full'
            # start with a diag cov matrix, actually  spherical since all cov are the same sigmas
            # and add small amount of noise
            r_sigsmas = torch.abs(torch.randn(self.in_features))
            self.sigmas.data.copy_(s*r_sigsmas*torch.eye(self.in_features) + 0.00*torch.randn(self.in_features,self.in_features))
            # ensure invertible using only O(N^2) instead of O(~N^3) with A=B*B' method
#             self.sigmas.data.copy_(0.5*(self.sigmas+self.sigmas.transpose(1,2)))
#             self.sigmas.data.copy_(self.sigmas + (2.0*torch.eye(self.in_features).expand_as(self.sigmas)))
            self.inv_covar.data.copy_((1.0/s)*(1.0/r_sigsmas)*torch.eye(self.in_features))
            
        
    def forward(self, input):
       
        ### linear part is the same as normal NNs
        biases = -torch.sum(torch.mul(self.weights, self.centers), dim=-1)
        l = F.linear(input, self.weights, bias=biases)
        # optional, apply tanh here
#         l = torch.tanh(l)

        ### gaussian component

        # unsqueeze the inputs to allow broadcasting
        # distance to centers
        g = input.unsqueeze(1)-self.centers
        # add noise to centers if required and if in training
        if (self.noisy_centers and self.training):
            c_noise = torch.Tensor(np.random.normal(scale=self.scale, size=self.centers.size()))
            # send to proper device
            c_noise = c_noise.to(next(self.parameters()).device)
            g = g+c_noise
            
        # spherical gaussian
        if self.covar_type == 'sphere':
           
            # if the ordinal is smaller than one, prevent zero distance to center or grad will be none
            if self.ordinal < 1.0:
                g = torch.abs(g)+1e-32
            else:
                g = torch.abs(g)
                
            # raise to ordinal power
            g = g.pow(self.ordinal)
            # sum along axis
            g = g.sum(dim=2)
    #         # to make it identical to the torch.norm computation below add .pow(1.0/ord), but not really needed
    #         g = g.pow(1.0/self.ordinal)
    #         if (g != g).any(): raise TypeError("g 4 is nan \n {}".format(g))

            # apply sigma(s) // inv_cov
            g = g*(self.inv_covar**2)
#             g = g*torch.abs(self.inv_covar)

         
        # diagonal covariance gaussian
        elif self.covar_type == 'diag':
            # black magic - worked it out from [batch_size, num_neuron, input_dim] -> [batch_size, num_neurons]
            g = torch.einsum('zij,zij->zi', g*(self.inv_covar**2), g)
#             g = torch.einsum('zij,zij->zi', g*torch.abs(self.inv_covar), g)

         
        # full diagonal covariance gaussian
        else:
            # black magic - worked it out from [batch_size, num_neuron, input_dim] -> [batch_size, num_neurons]
            # keep in mind inv_covar is actually half of the cov matrix here
            g = torch.einsum('xzi,zij,zkj,xzk->xz', g, self.inv_covar, self.inv_covar, g)
        
        # apply exponential
        g = torch.exp(-g)
            
        # if in eval mode and random eval applied, replace zeros (ie activity far from center) with random [-max,max]
        if (self.random_eval and not self.training):
            max_v = torch.max(torch.abs(g)).item()
            # indexes to replace (where the activity is very close to zero)
            zero_inds = (g<=1e-32).float().to(next(self.parameters()).device)
            # indexes to not replace
            nzero_inds = (g>1e-32).float().to(next(self.parameters()).device)
            random_noise = torch.FloatTensor(g.shape).uniform_(-max_v, max_v).to(next(self.parameters()).device)
            g = torch.mul(g, nzero_inds) + torch.mul(random_noise, zero_inds)
        
        # optional, apply tanh to linear
        if self.non_lin != False:
            l = self.non_lin(l)

        ### combine gaussian with linear
#         print(l.shape, g.shape)
        res = l*g
        # optional, flatten res with extra non-linearity
#         res = F.tanh(res)
#         res = torch.clamp(res, min=-1.0, max=1.0)

        return res
    