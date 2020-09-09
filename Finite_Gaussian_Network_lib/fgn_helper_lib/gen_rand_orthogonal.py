import torch

def gen_rand_orthogonal(t):
    # given a pytorch tensor, returns a random orthogonal vector of same norm
    # https://stackoverflow.com/questions/33658620/generating-two-orthogonal-vectors-that-are-orthogonal-to-a-particular-direction
    # suffers from numerical instability
    
    # normalize t
    t2 = t/torch.norm(t, p=2)
    rand_orthogonal =  torch.rand_like(t2.flatten())
    rand_orthogonal -= t2.flatten().dot(rand_orthogonal) * t2.flatten()
    # normalize
    rand_orthogonal /= torch.norm(rand_orthogonal, p=2)
    # reshape
    rand_orthogonal = rand_orthogonal.reshape_as(t2)
    
#     make same norm as t
    rand_orthogonal *= torch.norm(t, p=2) 
    
    return(rand_orthogonal)