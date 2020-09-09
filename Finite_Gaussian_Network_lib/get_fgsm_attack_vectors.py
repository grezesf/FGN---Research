import torch
from torch.autograd import Variable

def get_fgsm_attack_vectors(model, samples, loss_func):
    # given a model and some samples
    # returns the untargeted attack vectors given by FGSM for this loss function
    
    # get model device 
    device = next(model.parameters()).device
    
    samples_as_var = Variable(samples, requires_grad=True)
    
    # set model to eval mode
    model.eval()

    # compute model output 
    cur_out = model(samples_as_var.to(device))
    _, pred_classes = torch.max(cur_out, dim=1, keepdim=False)
        
    # compute loss
    cur_losses = loss_func(cur_out, pred_classes)
    
    # apply backprob
    cur_losses.backward()
    
    
    # the adversarial directions
    fgsm_dirs = torch.sign(samples_as_var.grad.data)
    
    return fgsm_dirs