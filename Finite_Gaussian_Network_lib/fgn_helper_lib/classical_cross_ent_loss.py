import torch.nn.functional as F

from .l2_loss import l2_loss

# cross-entropy loss function to be used by Feedforward_Classic_net
# includes l2_loss

def classical_cross_ent_loss(model, output, target, lmbda_l2=1e-4):
    cent_loss = F.cross_entropy(output, target.long())
    l2 = l2_loss(model)
    return cent_loss + lmbda_l2*l2