from collections import OrderedDict 
import torch
import numpy as np

def convert_classic_to_fgn(classic_model, fgn_model, init_factor=10.0, verbose=False):
    # given two similar models, one Feedforward_Classic_net, one Feedforward_FGN_net
    # built with the same hidden_l_nums, drop_p
    # init_factor is used to determine the size of the sigma initialization
   
    
    # get covar_type
    covar_type = fgn_model.fl.covar_type
    
    # create the state dict to be loaded
    new_state_dict = build_ordered_dict_for_fgn(classic_model, covar_type, new_state_dict=OrderedDict(), init_factor=init_factor, verbose=verbose)
    
    if verbose :
        print()
        print('Classic', classic_model.state_dict().keys())
        print('FGN', fgn_model.state_dict().keys())
        print('NEW', new_state_dict.keys())
    
    fgn_model.load_state_dict(new_state_dict)
    
    #return nothing

def build_ordered_dict_for_fgn(classic_model, covar_type, init_factor, new_state_dict=OrderedDict(), path='', verbose=False):
    
    ###
    # classic_model: a pytorch module (with a statedict) to be used as base
    # new_state_dict: OrderedDict, will be return. Passed as variable to recursive accumulator
    # path1: used by the recursion, added to the name of the parameter
    ###
    
    if verbose: print(new_state_dict.keys())
    
    # for each module in the classic model
    for (name, param) in classic_model.named_children():
        
        if path != '':
            cur_path = path+'.'+name
        else:
            cur_path = name
            
        # if it's a moduleList, recurse
        if isinstance(param, torch.nn.modules.container.ModuleList):
            new_state_dict = build_ordered_dict_for_fgn(param, covar_type, init_factor, new_state_dict, path=cur_path)
        # if it's a linear layer, convert
        elif isinstance(param, torch.nn.modules.linear.Linear):
            converted = convert_state_dict_lin2FGN(lin_state_dict=param.state_dict(), covar_type=covar_type, path=cur_path, init_factor=init_factor)
            new_state_dict.update(converted)
        # if it's neither, just put it in the state dict (batchnorm for ex)
        else:
            for key in param.state_dict():
                new_state_dict.update({cur_path+'.'+key:param.state_dict()[key]})
    
    return new_state_dict

def convert_state_dict_lin2FGN(lin_state_dict, covar_type, path, init_factor):
    # given the state dict of a nn.Linear layer,
    # returns a state dict for FGNlayer
    
    fgn_state_dict = OrderedDict()
    
    weights = list(lin_state_dict.values())[0]
    bias = list(lin_state_dict.values())[1]
    
    new_centers =  torch.Tensor([(-b/np.dot(x,x))*x for x,b in zip(weights.cpu().detach().numpy(), bias.cpu().detach().numpy())]) 
    
    fgn_state_dict[path+'.'+'weights'] = weights
    fgn_state_dict[path+'.'+'centers'] = new_centers
    fgn_state_dict[path+'.'+'biases'] = bias
    # the number of neurons
    out_features = weights.shape[0]
    in_features = weights.shape[1]
    # default inv_covar (large sigma = small inv_covar for large neuron radius of activity)
    default_sigma = init_factor*out_features # the number of neurons
    default_inv_covar = 1.0/default_sigma
    if covar_type == 'sphere':
        fgn_state_dict[path+'.inv_covars'] = torch.Tensor(out_features,).fill_(default_inv_covar)
    if covar_type == 'diag':
        fgn_state_dict[path+'.inv_covars'] = torch.Tensor(out_features, in_features).fill_(default_inv_covar)
    if covar_type == 'full':
        fgn_state_dict[path+'.inv_covars'] =  default_inv_covar*torch.eye(in_features, in_features,).expand(out_features,in_features, in_features)
    if covar_type == 'chol':
        fgn_state_dict[path+'.inv_covars'] =  default_inv_covar*torch.eye(in_features, in_features,).expand(out_features,in_features, in_features)
        
    return fgn_state_dict