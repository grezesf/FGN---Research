import fgn_helper_lib as fgnh
from FGN_layer import FGN_layer
import torch

def adjust_sigma_pred_func(fgn_model, dataloader, pred_func, verbose):
    
    ###
    # adjusts the sigmas of the given fgn model so that the pred accuracy over the dataset is max
    ###
    
    # best pred acc yet
    fgn_test_res = fgnh.test(fgn_model, dataloader, 
                             (lambda model, output, target:torch.tensor(0)), verbose=verbose, 
                             pred_func=pred_func)
    best_pred = fgn_test_res['test_accuracy']
    # best sigma multiplier yet
    best_sig_mult = 1.0
    # lower bound for sigma mult
    lower_bound = 0.0
    # uper bound for sigma
    upper_bound = float('Inf')
    
    # max number of values to test
    max_iter = 25
    
    # first double sigmas until performance decreases
    for ite in range(max_iter):
        # new val to test
        cur_sig_mult = 2.0*best_sig_mult
        if verbose: print(ite, "testing", cur_sig_mult)
        
        # apply multiplier
        # given an fgn model, multiplies all the sigmas by a value
        for p in fgn_model.modules():
            if isinstance(p, FGN_layer):
                p.sigmas = torch.nn.Parameter(p.sigmas*cur_sig_mult)
    
        # test
        fgn_test_res = fgnh.test(fgn_model, dataloader, 
                             (lambda model, output, target:torch.tensor(0)), verbose=verbose, 
                             pred_func=pred_func)
        cur_pred = fgn_test_res['test_accuracy']
        
        # reset sigmas 
        for p in fgn_model.modules():
            if isinstance(p, FGN_layer):
                p.sigmas = torch.nn.Parameter(p.sigmas/cur_sig_mult)
                
        if cur_pred > best_pred:
            if verbose: print("new best during doubling")
            # new best
            best_pred = cur_pred
            best_sig_mult = cur_sig_mult
            # increase lower bound
            lower_bound = cur_sig_mult
        else:
            # new upper bound
            upper_bound = cur_sig_mult
            # and exit loop
            break
            
            
    # next half sigmas until performance decreases
    for ite in range(max_iter):
        # new val to test
        cur_sig_mult = 0.5*best_sig_mult
        if verbose: print(ite, "testing", cur_sig_mult)
        
        # apply multiplier
        # given an fgn model, multiplies all the sigmas by a value
        for p in fgn_model.modules():
            if isinstance(p, FGN_layer):
                p.sigmas = torch.nn.Parameter(p.sigmas*cur_sig_mult)
    
        # test
        fgn_test_res = fgnh.test(fgn_model, dataloader, 
                             (lambda model, output, target:torch.tensor(0)), verbose=verbose, 
                             pred_func=pred_func)
        cur_pred = fgn_test_res['test_accuracy']
        
        # reset sigmas 
        for p in fgn_model.modules():
            if isinstance(p, FGN_layer):
                p.sigmas = torch.nn.Parameter(p.sigmas/cur_sig_mult)
                
        if cur_pred >= (1.0-1e-3)*best_pred:
            if verbose: print("new best during halfing")
            # new best
            best_pred = cur_pred
            best_sig_mult = cur_sig_mult
            # new upper bound
            upper_bound = cur_sig_mult
        else:
            # increase lower bound
            lower_bound = cur_sig_mult
            # and exit loop
            break
    
    # now that we have a real bounds, search by dichotomie
    for ite in range(max_iter):

        # new val to test
        cur_sig_mult = 0.5*(upper_bound+lower_bound)
        if verbose: print(ite, "testing", cur_sig_mult)
        
        # apply multiplier
        # given an fgn model, multiplies all the sigmas by a value
        for p in fgn_model.modules():
            if isinstance(p, FGN_layer):
                p.sigmas = torch.nn.Parameter(p.sigmas*cur_sig_mult)
    
        # test
        fgn_test_res = fgnh.test(fgn_model, dataloader, 
                             (lambda model, output, target:torch.tensor(0)), verbose=verbose, 
                             pred_func=pred_func)
        cur_pred = fgn_test_res['test_accuracy']
        
        # reset sigmas 
        for p in fgn_model.modules():
            if isinstance(p, FGN_layer):
                p.sigmas = torch.nn.Parameter(p.sigmas/cur_sig_mult)
                
        if cur_pred >= (5.0-1e-3)*best_pred:
            if verbose: print("new best during dicho")
            # new low bound
            if cur_sig_mult > best_sig_mult:
                lower_bound = cur_sig_mult
            # new upper bound
            else:
                upper_bound = cur_sig_mult
            # new best
            best_pred = cur_pred
            best_sig_mult = cur_sig_mult
                
        else:
            # new low bound
            if cur_sig_mult < best_sig_mult:
                lower_bound = cur_sig_mult
            # new upper bound
            else:
                upper_bound = cur_sig_mult
            
    # apply best mult
    if verbose: print("best multiplier:", best_sig_mult)
    for p in fgn_model.modules():
        if isinstance(p, FGN_layer):
            p.sigmas = torch.nn.Parameter(p.sigmas*best_sig_mult)
            
    return None