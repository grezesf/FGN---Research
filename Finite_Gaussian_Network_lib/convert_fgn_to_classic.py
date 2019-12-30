from collections import OrderedDict 
import torch

def convert_fgn_to_classic(fgn_model, classic_model):
    # given two similar models, one Feedforward_Classic_net, one Feedforward_FGN_net
    # built with the same hidden_l_nums, drop_p
    
    # create the state dict to be loaded
    new_state_dict = build_ordered_dict_for_classic(fgn_model)
    
    classic_model.load_state_dict(new_state_dict)
    
    #return nothing
    
    
def build_ordered_dict_for_classic(fgn_model, new_state_dict=OrderedDict(), path=''):
    
    ###
    # fgn_model: a pytorch module (with a statedict) to be used as base
    # new_state_dict: OrderedDict, will be return. Passed as variable to recursive accumulator
    # path1: used by the recursion, added to the name of the parameter
    ###
    
    # for each module in the classic model
    for (name, param) in fgn_model.named_children():
        
        if path != '':
            cur_path = path+'.'+name
        else:
            cur_path = name
            
        # if it's a moduleList, recurse
        if isinstance(param, torch.nn.modules.container.ModuleList):
            new_state_dict = build_ordered_dict_for_classic(param, new_state_dict, path=cur_path)
        # if it's a FGN_layer, convert
        elif isinstance(param, fgnl.FGN_layer):
            converted = convert_state_dict_FGN2lin(param.state_dict(), covar_type=covar_type, path=cur_path)
            new_state_dict.update(converted)
        # if it's neither, just put it in the state dict (batchnorm for ex)
        else:
            for key in param.state_dict():
                new_state_dict.update({cur_path+'.'+key:param.state_dict()[key]})
    
    return new_state_dict