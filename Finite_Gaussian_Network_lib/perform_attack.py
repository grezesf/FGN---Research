from tqdm import tqdm
import itertools
import numpy as np
import pickle

def perform_attack(attack_func,
                   f_model,
                   dataloader,
                   save_file,
                   start=0, stop=None):
    # iterates over dataloader 
    # attack is the attack function AFTER being defined: ex LinfPGD_attack=foolbox.attacks.LinfPGD()
    # (so it's the output foolbox.attacks.LinfPGD(), not foolbox.attacks.LinfPGD itself)
    # ensure the dataloader iterator returns (inputs, labels) 
    # start stop are for itertools
    
    # defines results to return, shape is (epsilons, sample, (sample shape))=(18,32xbatches,1,8000)
    num_epsilons = 18 # hardcoded for now
    data_shape = (1,8000) # next(iter(dataloader))[0].shape[1:] # this could be expensive, hardcoded for now
    # create empty lists of the right shape
    results = {'adv_raw':np.array([]).reshape((num_epsilons, 0, *(data_shape))),
               'adv_clipped':np.array([]).reshape((num_epsilons, 0, *(data_shape))),
               'adv_success':np.array([]).reshape((num_epsilons, 0))}
    
    # iterate over loader
    for inputs, labels in tqdm(itertools.islice(dataloader, start, stop)):
        
        # attack
        adv_raw, adv_clipped, adv_success = attack_func(f_model = f_model, 
                                                        inputs = inputs, 
                                                        labels =labels
                                                       )
        # compile with results
        results['adv_raw'] = np.concatenate([results['adv_raw'],
                                             np.array([x.cpu().numpy() for x in adv_raw])],
                                            axis=1)
        results['adv_clipped'] = np.concatenate([results['adv_clipped'],
                                                 np.array([x.cpu().numpy() for x in adv_clipped])],
                                                axis=1)
        results['adv_success'] = np.concatenate([results['adv_success'],
                                                 np.array([x.cpu().numpy() for x in adv_success])],
                                                axis=1)
    
    
        # save results
        # save files separately (can be as big as 11GB)
        for adv_name in ['adv_raw', 'adv_clipped', 'adv_success']:
            with open(save_file+'_{}_start_{}_stop_{}.pickle'.format(adv_name, start, stop), 'wb') as f:
                pickle.dump(results[adv_name], f, protocol=4)

    # return nothing
    return()