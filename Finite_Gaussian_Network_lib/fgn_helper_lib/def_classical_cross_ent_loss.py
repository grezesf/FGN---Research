from .classical_cross_ent_loss import classical_cross_ent_loss

# defines a loss func based on a given lambda for the l2 loss

def def_classical_cross_ent_loss(lmbda_l2=0.0, **kwargs):

    return (lambda model, output, target: classical_cross_ent_loss(model, output, target, lmbda_l2=lmbda_l2))