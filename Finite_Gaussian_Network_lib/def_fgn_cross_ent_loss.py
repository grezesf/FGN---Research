from fgn_cross_ent_loss import fgn_cross_ent_loss

# defines a loss func based on a given lambda for the l2 loss for Feedforward_FGN_net

def def_fgn_cross_ent_loss(lmbda_l2, lmbda_sigs):

    return (lambda model, output, target: fgn_cross_ent_loss(model, output, target, lmbda_l2=lmbda_l2, lmbda_sigs=lmbda_sigs))