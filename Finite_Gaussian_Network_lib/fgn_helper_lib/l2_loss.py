import torch

def l2_loss(model):
    # average of linear layer weights^2 in the model, used for regularization
    
    # init on right device (not tested on shared GPU models)
    count = 0
    l2 = torch.tensor([0.0], device=next(model.parameters()).device)
    
    for p in model.named_parameters():
        if ('weight' in p[0]) or ('bias' in p[0]):
            l2 += torch.sum(p[1]**2)
            count += len(p[1])
        
    # average per weight
    l2 = l2/count
    
    return l2