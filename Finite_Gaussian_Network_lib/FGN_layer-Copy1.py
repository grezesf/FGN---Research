# a special ca

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np 
import math

class FGN_layer(nn.Module):
    r""" Applies a Finite Gaussian Neuron layer to the incoming data
    
    Args:
    
    Shape:
    
    Attributes:
    
    Examples:
        
        >>> l=FGN_layer(20,30)
    
    """
    def __init__(self, in_features, out_features, 
                 covar_type='diag', ordinal=2.0, non_lin=False, free_biases=True,
                 **kwargs):
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
        # should an extra non-linearity be used (currently always tanh)?
        # can be False or  True (TODO: any function on tensors)
        self.non_lin = non_lin
        # should centers of gaussian define the linear layer biases?
        # ie should the zero-line of the linear part go through the center?
        self.free_biases = free_biases
       
        # random_eval is False at creation. Can be manually changed later
        self.random_eval = False

        
        # learnable parameters and related
        # regular NN weights (transposed at the start, see order of Tensor(dims))
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.biases = nn.Parameter(torch.Tensor(out_features,), requires_grad=True)
        
        # centers of FGNs
        self.centers = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        # size/range of FGNs
        # inverse covariance will actually be used
        if covar_type == 'sphere':
#             self.sigmas = nn.Parameter(torch.Tensor(out_features,), requires_grad=True)
            self.inv_covars = nn.Parameter(torch.Tensor(out_features,), requires_grad=True)
        elif covar_type == 'diag':
#             self.sigmas = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
            self.inv_covars = nn.Parameter(torch.Tensor(out_features, in_features,), requires_grad=True)
        elif covar_type == 'full':
#             self.sigmas = nn.Parameter(torch.Tensor(out_features, in_features, in_features,), requires_grad=True)
            self.inv_covars = nn.Parameter(torch.Tensor(out_features, in_features, in_features,), requires_grad=True)
        else:
            # error
            raise TypeError("covar_type not one of ['sphere', 'diag', 'full']")
        # minimum sigma
        self.eps = 1e-8 

        # parameter init call
        self.reset_parameters()
    
    # parameter init definition
    def reset_parameters(self):
        # regular NN init
#         s = np.sqrt(self.in_features)
#         self.weights.data.uniform_(-s, s)
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))

#         self.biases.data.uniform_(-s, s)
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.biases, -bound, bound)

        # centers init, assuming data normalized to mean 0 var 1 (not necessarily true after first layer)
        self.centers.data.normal_(std=0.1)

        # sigmas init, to be researched further
#         s = np.sqrt(self.in_features)
        s = self.in_features
#         s = np.log2(self.in_features)
        if self.covar_type in ['sphere', 'diag']:
#             self.sigmas.data.uniform_(s-0.5, s+0.5)
            self.inv_covars.data.uniform_(1.0/(s+0.5), 1.0/(s-0.5))
        elif self.covar_type == 'full':
            # start with a diag cov matrix, actually  spherical since all cov are the same sigmas
            # and add small amount of noise
            r_sigmas = torch.abs(torch.randn(self.in_features))
#             self.sigmas.data.copy_(s*r_sigmas*torch.eye(self.in_features) + 0.00*torch.randn(self.in_features,self.in_features))
            # ensure invertible using only O(N^2) instead of O(~N^3) with A=B*B' method
#             self.sigmas.data.copy_(0.5*(self.sigmas+self.sigmas.transpose(1,2)))
#             self.sigmas.data.copy_(self.sigmas + (2.0*torch.eye(self.in_features).expand_as(self.sigmas)))
            self.inv_covars.data.copy_((1.0/s)*(1.0/r_sigmas)*torch.eye(self.in_features))
    
    def update_biases_from_centers(self):
        # computes the biases of linear layers based on centers of gaussians
        # this function is called by forward if self.free_biases is False
        # it might be used later elsewhere.
        self.biases = torch.nn.Parameter(-torch.sum(torch.mul(self.weights, self.centers), dim=-1))
        
    def forward(self, inputs):
        ### linear part is the same as normal NNs
        if not self.free_biases:
            # compute biases from centers to keep gaussians on the zero line
            self.update_biases_from_centers()
        l = F.linear(inputs, self.weights, bias=self.biases)
        # optional, apply tanh here to have same activity as classical net
        if self.non_lin != False:
            l = torch.tanh(l)

        ### gaussian component
        # computation for sphere and diag are the same?
        if self.covar_type=='sphere':
            centers_list = [c for c in self.centers]
            inv_covars_list = [i for i in self.inv_covars]
            # distance to centers,
            # with inv_covar applied,
            # with ordinal applied,
            # and small value added to prevent problems when ord<1
            g_list = [torch.sum( torch.pow( torch.abs(inputs-center) + 1e-32*float(self.ordinal<1.0), self.ordinal), dim=1) 
                      for (center, inv_covar) in zip(centers_list, inv_covars_list)]
            g = torch.stack(g_list)
            g = torch.t(g)
            g = g*(torch.clamp(self.inv_covars, max=1.0/self.eps)**2)
        
        elif self.covar_type=='diag':
            centers_list = [c for c in self.centers]
            inv_covars_list = [i for i in self.inv_covars]
            # distance to centers,
            # with inv_covar applied,
            # with ordinal applied,
            # and small value added to prevent problems when ord<1
            g_list = [torch.sum( torch.pow( torch.abs(inputs-center) * 
                                          torch.pow(inv_covar**2 + 1e-32*float(1.0/self.ordinal<1.0), 1./self.ordinal) 
                                          + 1e-32*float(self.ordinal<1.0), 
                                          self.ordinal),
                                dim=1) 
                      for (center, inv_covar) in zip(centers_list, inv_covars_list)]
            g = torch.stack(g_list)
            g = torch.t(g)
            
            #old
#         # unsqueeze the inputs to allow broadcasting
#         # distance to centers
#         g = inputs.unsqueeze(1)-self.centers

#         # add noise to distances if required and if in training
#         if (self.noisy_centers and self.training):
#             c_noise = torch.Tensor(np.random.normal(scale=self.scale, size=self.centers.size()))
#             # send to proper device
#             c_noise = c_noise.to(next(self.parameters()).device)
#             g = g+c_noise
                    
#         # spherical gaussian
#         if self.covar_type == 'sphere':

# #             # if the ordinal is smaller than one, prevent zero distance to center or grad will be none
# #             if self.ordinal < 1.0:
# #                 g = torch.abs(g)+1e-32
# #             else:
# #                 g = torch.abs(g)
                
# #             # raise to ordinal power
# #             g = g.pow(self.ordinal)
# #             # sum along axis
# #             g = g.sum(dim=2)
# #     #         # to make it identical to the torch.norm computation below add .pow(1.0/ord), but not really needed
# #     #         g = g.pow(1.0/self.ordinal)
# #     #         if (g != g).any(): raise TypeError("g 4 is nan \n {}".format(g))

#             # apply sigma(s) // inv_cov
#             g = g*(self.inv_covar**2)
# #             g = g*torch.abs(self.inv_covar)

         
#         # diagonal covariance gaussian
#         elif self.covar_type == 'diag':
# #             black magic - worked it out from [batch_size, num_neuron, input_dim] -> [batch_size, num_neurons]
# #             g = torch.einsum('zij,zij->zi', g*(torch.abs(self.inv_covar)**2), g)
# #             g = torch.einsum('zij,zij->zi', g*torch.abs(self.inv_covar), g)
#             # apply sigma(s) // inv_cov
#             g = g*(self.inv_covar**2)
         
        # full diagonal covariance gaussian
        else:
            # unsqueeze the inputs to allow broadcasting
            # distance to centers
            g = inputs.unsqueeze(1)-self.centers
            
            # black magic - worked it out from [batch_size, num_neuron, input_dim] -> [batch_size, num_neurons]
            # keep in mind inv_covar is actually half of the cov matrix here
            g = torch.einsum('xzi,zij,zkj,xzk->xz', g, self.inv_covars, self.inv_covars, g)
        
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
            # if the models uses a non-linearity that clamps the outputs to [-1,1]
            # better to set the activity to be ouside the range
            # TODO: make a more generic guarantee
            max_v = 10000
            random_noise = torch.FloatTensor(g.shape).uniform_(max_v, max_v).to(next(self.parameters()).device)
            g = torch.mul(g, nzero_inds) + torch.mul(random_noise, zero_inds)
        
#         # optional, apply tanh to linear
#         if self.non_lin != False:
#             l = torch.tanh(l)

        ### combine gaussian with linear
#         print(l.shape, g.shape)
        res = l*g
        # optional, flatten res with extra non-linearity
#         res = F.tanh(res)
#         res = torch.clamp(res, min=-1.0, max=1.0)

        return res
    