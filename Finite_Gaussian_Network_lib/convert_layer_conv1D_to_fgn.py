import torch

def convert_layer_conv1D_to_fgn(classic_layer, fgn_layer,  init_factor=25000):
    # changes the weights of the fgn_layer to match the behavior of the classic_layer
    # the two layers MUST have been created with the same parameters
    
    # some size checks
    assert(fgn_layer.Conv1d.weight.shape==classic_layer.weight.shape)
    assert(fgn_layer.Conv1d.bias.shape==classic_layer.bias.shape)

    # convert params
    fgn_layer.Conv1d.weight = classic_layer.weight
    fgn_layer.Conv1d.bias = classic_layer.bias
    fgn_layer.centers = torch.nn.Parameter(torch.stack([(-b/torch.dot(w.flatten(),w.flatten()))*w.flatten()
                                                        for (b,w) in zip(classic_layer.bias,
                                                                         classic_layer.weight)]).reshape(fgn_layer.centers.shape)
                                          )
    
    # ensure large sigma, enough to mimic behavior of classic_layer
    fgn_layer.inv_covars = torch.nn.Parameter(fgn_layer.inv_covars/init_factor)
    
    # returns nothing
