import torch
import numpy as np
import copy
import fgn_helper_lib as fgnh
from FGSM_attack_targeted import FGSM_attack_targeted
from FGSM_attack_untargeted import FGSM_attack_untargeted

def full_FGSM_attack(model, dataloader,
                     num_attacks=None, attack_params=None, **kwargs):
    # for a given model and dataset, runs #num_attacks untargeted attacks.
    
   # model: a pytorch model that outputs a raw vector for a N-class prediction
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model is not a pytorch module")
    
    # dataloader: a pytorch dataloader
    if not isinstance(dataloader, torch.utils.data.dataloader.DataLoader):
        raise TypeError("dataloader is not a pytorch dataloader")
    # force dataloader sampler to be random
    if not isinstance(dataloader.sampler, torch.utils.data.sampler.RandomSampler):
        raise TypeError("dataloader sampler is not random")

    # num_attacks: if None, will attack every point in dataloader
    # otherwise will attack the given number, chosen randomly (this is why RandomSampler is required)
    if num_attacks==None:
        num_attacks = len(dataloader.dataset)
        
    #targeted: boolean, wether to do a targeted or untargeted attack
        
    # attack_params: dictionary with the parameters for FGSM attack
    # make some relatively arbitrary choices as defaults
    try: targeted = attack_params['targeted']
    except: targeted = False
    try: data_classes = attack_params['data_classes']
    except: data_classes = fgnh.get_dataloader_classes(dataloader)
    try: data_bounds = attack_params['data_bounds']
    except: data_bounds = fgnh.get_dataloader_bounds(dataloader)
    try: max_noise = attack_params['max_noise']
    except: max_noise = (max(data_bounds)-min(data_bounds))/10.0
    try: loss_func = attack_params['loss_func']
    except: loss_func = torch.nn.CrossEntropyLoss()
    try: step_size = attack_params['step_size']
    except: step_size = max_noise/5.0
    try: steps = attack_params['steps']
    except: steps = 5
    try: confidence = attack_params['confidence']
    except: confidence_req = 0.5
    
    
    ### kwargs    
    # verbose: boolean, used to print training stats
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False

    # attack model
    attack_count = 0
    
    # values to return
    successful_attack_count = 0
    confidence_dist = []
    steps_dist = []
    adv_class_dist = [] 
    
    # load a batch
    for batch, classes in dataloader:
        # check if enough attacks
        if attack_count>=num_attacks:
            # exit  for batch, classes in dataloader loop
            break
        
        # traverse the batch 
        for data_point, point_class in zip(batch, classes):
            
            # check if enough attacks
            if attack_count>=num_attacks:
                # exit for data_point, point_class in zip(batch, classes):
                break
                           
            # perform attack
            if targeted:
                # pick random class to attack different than original pred
                orig_class = fgnh.get_class_from_pred(model, data_point, **kwargs)
                choices = copy.deepcopy(data_classes)
                choices.remove(orig_class)
                target_class = np.random.choice(choices)
                if verbose: print("Attack model with target", target_class)
                adv_data, adv_noise, attack_results = FGSM_attack_targeted(model, data_point, target_class, 
                                                                    max_noise, loss_func, step_size,
                                                                    data_bounds, steps, confidence_req,
                                                                    **kwargs)
            else: # untargeted
                if verbose: print("Untargeted attack of model")
                adv_data, adv_noise, attack_results = FGSM_attack_untargeted(model, data_point, 
                                                                    max_noise, loss_func, step_size,
                                                                    data_bounds, steps, confidence_req,
                                                                   **kwargs)
            
            
            # saved desired results
            # attack successful?
            if (attack_results['confidence']>=confidence_req):
                successful_attack_count+=1
            # final_confidence distribution
            confidence_dist.append(attack_results['confidence'])
            # number of steps
            steps_dist.append(attack_results['steps'])
            # class of the adversarial data
            adv_class = fgnh.get_class_from_pred(model, adv_data, **kwargs)
            adv_class_dist.append(adv_class)
                        
            # increment attack count
            attack_count+=1
            if verbose: print("Attack count:", attack_count)
        
            # go to next in batch
        # go to next batch
        
    # return results
    results_dict = {'confidence_dist':confidence_dist, 'steps_dist':steps_dist, 'successful_attack_count':successful_attack_count, 'adv_class_dist':adv_class_dist}
    
    return results_dict

