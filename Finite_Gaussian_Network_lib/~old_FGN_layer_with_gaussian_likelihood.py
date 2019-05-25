# define the FGN layer class to dev

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class FGN_layer(nn.Module):
    r""" Applies a Finite Gaussian Neuron layer to the incoming data
    also includes computation of the likelihood (negative, log) of the gaussians over the data 
    (expensive to compute)
    
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
        self.sigs = nn.Parameter(torch.Tensor(out_features,), requires_grad = True)
        # importance of each gaussian for likelihoods
        self.pis = nn.Parameter(torch.Tensor(out_features,), requires_grad = True)
   
        
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
        self.sigs.data.uniform_(s, s)
        # PIs init, start at 1/n each
        self.pis.data.fill_(1.0/self.out_features)
        
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

        # for future, use any norm?
#         g = torch.norm(input.unsqueeze(1)-self.centers, p=1, dim=2)

        # apply sigma
        eps = 1e-3 #minimum sigma
        g = -g/(torch.clamp(self.sigs, min=eps)**2 )
#         g = -g/(self.sigs**2)
        # apply exponential
        g = torch.exp(g)

        # combine gaussian with linear
        res = l*g
        # optional, flatten res
        # res = F.tanh(res)

        # negative likelihoods computation for each data point
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Likelihood_function
        neg_likelihoods = input.unsqueeze(1)
        neg_likelihoods = neg_likelihoods - self.centers
        neg_likelihoods = neg_likelihoods**2
        neg_likelihoods = torch.sum(neg_likelihoods, dim=-1)
        neg_likelihoods = neg_likelihoods/(self.sigs**2)
        # add ln(det(SIG)) = 2k*log(sig)
        neg_likelihoods = neg_likelihoods + 2*self.in_features*torch.log(self.sigs)
        # at this stage, all are ~ -ln N(sample|gaussian) for each gaussian in layer
        # multiply by the PIs, constrained to sum to 1 and be >0
        # we use softmax for this, meaning that the underlying PIs are not constrained.
        pis_normalized = F.softmax(self.pis, dim=-1)
        neg_likelihoods = neg_likelihoods*pis_normalized
        # average them up
#         neg_likelihoods = torch.sum(neg_likelihoods, dim=-1)/neg_likelihoods.shape[0]
        neg_likelihoods = torch.mean(neg_likelihoods)
        
        return res, neg_likelihoods
    