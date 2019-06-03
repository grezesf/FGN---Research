from collections import OrderedDict 
import torch
import numpy as np

def convert_classic_to_fgn(classic_model, fgn_model):
    # given two similar models, one Feedforward_Classic_net, one Feedforward_FGN_net
    # built with the same hidden_l_nums
    new_state_dict = build_ordered_dict_for_fgn(classic_model)
    
    fgn_model.load_state_dict(new_state_dict)
    
    #return nothing

def build_ordered_dict_for_fgn(classic_model, new_state_dict=OrderedDict(), path=''):
    
    ###
    # classic_model: a pytorch module (with a statedict) to be used as base
    # new_state_dict: OrderedDict, will be return. Passed as variable to recursive accumulater
    # path1: used by the recursion, added to the name of the parameter
    ###
    
    # for each module in the classic model
    for (name, param) in classic_model.named_children():
        
        if path != '':
            cur_path = path+'.'+name
        else:
            cur_path = name
            
        # if it's a moduleList, recurse
        if isinstance(param, torch.nn.modules.container.ModuleList):
            new_state_dict = build_ordered_dict_for_fgn(param, new_state_dict, cur_path)
        # if it's a linear layer, convert
        elif isinstance(param, torch.nn.modules.linear.Linear):
            converted = convert_state_dict_lin2FGN(param.state_dict(), cur_path)
            new_state_dict.update(converted)
        else:
            for key in param.state_dict():
                new_state_dict.update({cur_path+'.'+key:param.state_dict()[key]})
    
    return new_state_dict

def convert_state_dict_lin2FGN(lin_state_dict, path, default_sigma=1000.0):
    # given the state dict of a nn.Linear layer,
    # returns a state dict for FGNlayer
    
    fgn_state_dict = OrderedDict()
    
    weights = lin_state_dict.values()[0]
    bias = lin_state_dict.values()[1]
    
    new_centers =  torch.Tensor([(-b/np.dot(x,x))*x for x,b in zip(weights.cpu().detach().numpy(), bias.cpu().detach().numpy())]) 
    
    fgn_state_dict[path+'.'+'weights'] = weights
    fgn_state_dict[path+'.'+'centers'] = new_centers
    # default  sigmas (usually large)
    fgn_state_dict[path+'.sigmas'] = torch.Tensor([default_sigma for _ in range(len(weights))])
    
    return fgn_state_dict