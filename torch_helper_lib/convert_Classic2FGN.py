from collections import OrderedDict 
import torch
import numpy as np

# function that will convert state dicts 
def convert_state_dict_lin2FGN(lin_state_dict, fgn_state_dict):
    # given the state dict of a nn.Linear layer, and the fgn_state_dict (for the names)
    # returns a state dict for FGNlayer
    
    weights = lin_state_dict.values()[0]
    bias = lin_state_dict.values()[1]
    
    new_centers =  torch.Tensor([(-b/np.dot(x,x))*x for x,b in zip(weights.cpu().detach().numpy(), bias.cpu().detach().numpy())]) 
    
    fgn_state_dict[fgn_state_dict.keys()[0]] = weights
    fgn_state_dict[fgn_state_dict.keys()[1]] = new_centers
    fgn_state_dict[fgn_state_dict.keys()[2]] = fgn_state_dict[fgn_state_dict.keys()[2]]
    
    return fgn_state_dict

def build_lin_layer_state_dicts(model_state_dict):
    # given a full model state dict, builds a list of state_dicts to be passed later to convert_...
    
    res_list = []
    next_sd = OrderedDict()
    # go through the model
    for key in model_state_dict:
        #check if weights
        if 'weight' in key:
            # add to next 
            next_sd.update({key: model_state_dict[key]})
        if 'bias' in key:
            # add to next
            next_sd.update({key: model_state_dict[key]})
            # bias is the last one per layer, update ordered dict
            # add to res_list
            res_list.append(next_sd)
            # reset next
            next_sd = OrderedDict()
    
    return res_list
    
def build_fgn_layer_state_dicts(model_state_dict):
    # given a full fgn model state dict, builds a list of state_dicts to be passed later to convert_...
    
    res_list = []
    next_sd = OrderedDict()
    lin_state_dict = OrderedDict()
    # go through the model
    for key in model_state_dict:
        if 'weights' in key:
            # add to next 
            next_sd.update({key: model_state_dict[key]})
        if 'centers' in key:
            # add to next
            next_sd.update({key: model_state_dict[key]})
        if 'sigs' in key:
            # add to next
            next_sd.update({key: model_state_dict[key]})
            # sigs is the last one per layer, update ordered dict
            # add to res_list
            res_list.append(next_sd)
            # reset next
            next_sd = OrderedDict()
        # old, PIs for likelihod version of FGNets
#         if 'pis' in key:
#             # add to next
#             next_sd.update({key: model_state_dict[key]})
#             # pis is the last one per layer, update ordered dict
#             # add to res_list
#             res_list.append(next_sd)
#             # reset next
#             next_sd = OrderedDict()
    
    return res_list

def convert_Classic2FGN(classic_model, fgn_model):
    # given a Classic_MNIST_Net and a Feedforward_FGN_net of the same sizes
    # converts the weights from the classic net to the fgn net
    # both networks should have identical performance (or very close) after conversion
    
    classic_list = build_lin_layer_state_dicts(classic_model.state_dict())
    fgn_list = build_fgn_layer_state_dicts(fgn_model.state_dict())

#     print("classic",classic_list)
#     print("fgn",fgn_list)
    
    new_state_dict = OrderedDict()
    for c, f in zip (classic_list, fgn_list):
        new_state_dict.update(convert_state_dict_lin2FGN(c,f))
        
    fgn_model.load_state_dict(new_state_dict)
    
    # return nothing
    