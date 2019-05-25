# NLL loss function for FGNets, with some weight regularizer
# to be used by th.training()

import torch.nn.functional as F
import torch

def NLL_loss(model, output, target):
    
    ### parameters
    # model: the model used to compute the loss
    # output: the output of the model
    # target: the desired output
    
    # normal NLL loss
    nll_loss = F.nll_loss(output, target.long())

    # sum of sigma^2 for regularizer 
    for p in model.named_parameters():
        if 'sigs' in p[0]:
            try:
                sig_loss += torch.sum(p[1]**2)
            except:
                sig_loss = torch.sum(p[1]**2)
        
    # regulizer parameter
    t_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lmbda = 1.0/(len(target)+t_params)

    # combine 
    loss = nll_loss + lmbda*sig_loss
    
    return loss