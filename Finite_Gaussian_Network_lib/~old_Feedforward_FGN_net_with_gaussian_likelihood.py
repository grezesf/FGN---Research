# Feedforward Finite Gaussian Neural Network model with neg-log likelhood computation over the gaussians


### NOT UP TO DATE


import torch.nn as nn
import torch.nn.functional as F
import torch

from FGN_layer import FGN_layer 

class Feedforward_FGN_net(nn.Module):
    
    def __init__(self,in_feats, out_feats, hidden_l_nums, drop_p=0.0):
        super(Feedforward_FGN_net, self).__init__()
        
        # input dimension
        self.in_feats=in_feats
        # output imension (number of classes)
        self.out_feats=out_feats
        
        # the hidden layers    
        # input dropout
        self.id = nn.Dropout(p=drop_p) 
        # add modules
        next_in = self.in_feats
        self.hidden_layers = nn.ModuleList([])
        for idx, next_out in enumerate(hidden_l_nums):
            # the FGN layer
            self.hidden_layers.append(FGN_layer(next_in, next_out))
            # optional: batchnorm
            self.hidden_layers.append(nn.BatchNorm1d(next_out))
            # optional: dropout layer
            if drop_p > 0:
                self.hidden_layers.append(nn.Dropout(p=drop_p))
            # reset feat for next layer
            next_in = next_out

        # final layer
        self.fl = FGN_layer(next_in, self.out_feats)
        
    def forward(self, x):
        # squash the image
        x = x.view(-1, self.in_feats)
        # input dropout
        x = self.id(x)
        # for each hidden layer
        for layer in self.hidden_layers:
            # apply finite gaussian layer
            x,l = layer(x)
            # add to other likelihoods
            try:
                ls = torch.cat((ls,l), dim=0)
            except:
                ls=l
            # not needed for FGN non-linerarity
#             x = torch.tanh(x)
            # not needed: batchnorm
        # final out layer
        x,l = self.fl(x)
        # add to other likelihoods
        try:
            ls = torch.cat((ls,l), dim=0)
        except:
            ls=l
        # softmax
        x = F.log_softmax(x, dim=-1)
        
        return x,ls