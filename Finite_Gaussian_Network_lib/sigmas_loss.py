import torch

def sigmas_loss(model):
    # average of sigma^2 in the model, used for regularization
    
    # init on right device (not tested shared GPU models)
    count = 0
    sig_loss = torch.tensor([0.0], device=next(model.parameters()).device)
    for p in model.named_parameters():
        if 'sigmas' in p[0]:
            sig_loss += torch.sum(p[1]**2)
            count += len(p[1])

    # average per sigma
    sig_loss = sig_loss/count
    
    return sig_loss