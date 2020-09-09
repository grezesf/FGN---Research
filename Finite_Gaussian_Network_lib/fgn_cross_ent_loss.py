
import torch.nn.functional as F

from .fgn_helper_lib.l2_loss import l2_loss
from .sigmas_loss import sigmas_loss

# cross-entropy loss function to be used by Feedforward_FGN_net
# includes l2_loss and sigmas reduction loss

def fgn_cross_ent_loss(model, output, target, lmbda_l2=1e-4, lmbda_sigs=1e-3):
    
    # normal Cent loss
    cent_loss = F.cross_entropy(output, target.long())
#     cent_loss = F.binary_cross_entropy(output, target.long())
    # normal l2 loss
    l2 = l2_loss(model)
    #sigma loss
    sig_loss = sigmas_loss(model)
    
    return cent_loss + lmbda_l2*l2 + lmbda_sigs*sig_loss