import torch

def get_rand_orthogonal_set(t, mode='canonical'):
    # given a tensor t, returns a (random or canonical) set orthognal vectors to t
    # (probably not uniformly sampled)
    # no check if the random init is actually invertible
    
    # normalize t
    t2 = t.flatten() / t.norm()
    
    if mode=='random':
        r = torch.stack([torch.rand_like(t2) for _ in range(len(t2)-1)])
    if mode=='canonical':
        r = torch.eye(len(t2)-1,len(t2))
    
    A = torch.cat((t2.unsqueeze(0),r)).T
    
    Q,R = torch.qr(A)
    
    # reshape back to orignal, renorm back to original
    r = torch.stack([t.norm()* q.reshape_as(t) for q in Q.T])
    
    # the first elem is either t or -t, just replace it
    r[0] = t
    return r
