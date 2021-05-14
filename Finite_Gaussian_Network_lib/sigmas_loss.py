import torch
import numpy as np

def sigmas_loss(model, covar_type=None):
    # average of sigma^2 in the model, used for regularization
    # actually relies on the inv_covar value AND NAME IN THE NAMED_PARAMETERS!
    
    # init on right device (not tested shared GPU models)
    count = 0
    sig_loss = torch.tensor([0.0], device=next(model.parameters()).device)
    
    if covar_type is None:
        # try to guess the model type
        covar_type = model.covar_type
    
    for p in model.named_parameters():
#         # old code when using sigmas and not inverse
#         if 'sigmas' in p[0]:
#             sig_loss += torch.sum(p[1]**2)

        # check param
        if 'inv_covar' in p[0]:
            # spherical gaussian same as diagonal cov matrix
            if covar_type in ['sphere', 'diag']:
                sig_loss += torch.sum(torch.abs(1.0/(p[1]**2)))
                    
#                 sig_loss -= torch.sum(torch.abs(p[1])**2)
#                 sig_loss -= torch.sum(torch.abs(p[1]))
#                 sig_loss = -sig_loss
#                 sig_loss = 1./sig_loss

            # full covariance matrix
            elif covar_type == 'full':
                # compute trace of inv_covar, and maximize it 
                # note that this is done by minimizing the opposite, thus we have negative loss values 
                sig_loss -= torch.einsum('zii->', torch.abs(p[1]))
                # above seems to lead to nans, instead use inverse and small lmda_sigmas
#                 sig_loss += 1/torch.einsum('zii->', torch.abs(p[1]))
            
            elif covar_type == 'chol':
                
                # in theory maximizing the trace of the half inv_covar is good enough
                sig_loss -= torch.einsum('zii->', torch.abs(p[1]))
                
                # minimize trace of cov matrix , optimize=['einsum_path', (0,)] is not a param for torch but would help a little
                # tr(A=PDP-1)=tr(D); tr(A-1)=Tr(D-1)  https://math.stackexchange.com/questions/391128/trace-of-an-inverse-matrix
                # does this mean minimizing the trace of A is also minimizing the trace of A^-1?
#                 sig_loss += torch.einsum('zii->', p[1]*(p[1].transpose(1,2)) )
                # also keep in mind with covar_type='half_full', inv_covar is actually half of the real underlying covariance matrix
#                 sig_loss += torch.einsum('zik,zik->', p[1], p[1])
                # doesn't seem to work, use computation below (slower but shouldn't matter too much)

#                 # full inverse computation
#                 # multiply the half covar matrixes together
#                 sigmas = torch.einsum('zij, zkj -> zik', p[1],p[1])
#                 sigmas = sigmas.inverse()
#                 # trace of inverse
#                 sig_loss += torch.einsum('zii->', sigmas)
            
            else:
                # should never hppen
                raise Exception('Something went wrong with covar_type')

            # count of neurons*params for averaging?
            count += np.prod(p[1].shape)
            
    
    # average per sigma
    sig_loss = sig_loss/float(count)
    
#     print("final",sig_loss)
    
    return sig_loss