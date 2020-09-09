
# coding: utf-8

# In[ ]:


# Exp-00-Convert_Classic_to_FGN
# take a classic network, convert it to FGN, and train for 1 epoch
# stdbuf -o 0 python ../Exp-00-Convert_Classic_to_FGN.py 2>&1 | tee log_Exp-00_2020-01-09-17:56_convert_log.txt


# In[ ]:


from __future__ import print_function


# In[ ]:


import numpy as np
import time
import os
import sys
import GPUtil
import pickle
import re

import torch

sys.path.append('/home/felix/Research/Adversarial Research/FGN---Research/')
import Finite_Gaussian_Network_lib as fgnl
import Finite_Gaussian_Network_lib.fgn_helper_lib as fgnh


# In[ ]:


# Preliminaries
# saved in :
base_dir =  '/home/felix/Research/Adversarial Research/FGN---Research/Experiments/Exp-00/'
# folder with the classic networks
exp_dir = '/2020_01_07_at_23:28:46/'


# In[ ]:


# load shared configs
shared_config_fixed_parameters_dic = np.load(base_dir+'/shared_config_fixed_parameters_dic.pckl',allow_pickle='TRUE')


# In[ ]:


batch_size = shared_config_fixed_parameters_dic['batch_size']
optimizer_name = shared_config_fixed_parameters_dic['optimizer_name']
lmbda_l2 = shared_config_fixed_parameters_dic['lmbda_l2']
opt_lr = shared_config_fixed_parameters_dic['opt_lr']
# num_epochs = shared_config_fixed_parameters_dic['num_epochs'] #not used


# In[ ]:


# load list of fgn configs
fgn_config_experiments_list = np.load(base_dir+'/fgn_config_experiments_list.pckl',allow_pickle='TRUE')
# same for all FGNs
non_lin = True
free_biases = True


# In[ ]:


# load data
(mnist_train_loader, mnist_val_loader, mnist_test_loader) = fgnh.mnist_dataloaders(batch_size=batch_size)


# In[ ]:


classic_model_list = []
# for every trained classic model

for path, dir,files in os.walk(base_dir+exp_dir):
    for file in files:
        if 'trained_classic_model_full.pth' in file:
            
#             print('adding {}/{}'.format(path, file))
            classic_model_list.append((path,file))


# In[ ]:


for x in classic_model_list:
    print(x)


# In[ ]:
        
# for every trained classic model

for path, file in classic_model_list:
    print()
    print('Converting {}/{}'.format(path, file))
    
    # create subfolder
    try:
        os.mkdir(path+'/converted FGNs')
    except Exception as e:
        print(e)

    # load config
    with open(path+'/config.txt','r+') as c:
        for l in c.readlines():
            if 'hidden_layer_sizes' in l:
                if l == 'hidden_layer_sizes []\n':
                    hidden_layer_sizes = []
                else:
                    hidden_layer_sizes = [int(x.strip(' ,][]')) for x in l[20:].split()]
            if 'lmbda_l2' in l : lmbda_l2 = float(l[8:])
            if 'drop_p' in l : drop_p = float(l[6:])

    # load classic model
    classic_model = torch.load(path+'/trained_classic_model_full.pth')

    # for each 
    for fgn_config in fgn_config_experiments_list:
        print()
        
        # extract params
        covar_type = fgn_config['covar_type']
        ordinal = fgn_config['ordinal']
        lmbda_sigma = fgn_config['lmbda_sigma']
        
        # loss for fgn uses lmbda_sigma
        loss  = fgnl.def_fgn_cross_ent_loss(lmbda_l2=lmbda_l2, lmbda_sigs=lmbda_sigma)

        # def model name
        model_name = 'converted_fgn_hl{}_dp{}_{}_ord{}_ls{}'.format(hidden_layer_sizes, 
                                                                   drop_p, 
                                                                   covar_type, 
                                                                   ordinal,
                                                                   lmbda_sigma)

        print('Current model name: {}'.format(model_name))

        # create folder
        model_folder = path+'/converted FGNs/'+model_name
        try:
            os.mkdir(model_folder)
        except Exception as e:
            print(e)
            print('Folder already exist')
            
        # check if trained pth file already exists

        if not os.path.exists(model_folder+'/trained_converted_fgn_model_full.pth'):
            # create fgn model
            fgn_model = fgnl.Feedforward_FGN_net(in_feats=28*28, 
                                             out_feats=10, 
                                             hidden_layer_sizes=hidden_layer_sizes,
                                             drop_p = drop_p, 
                                             covar_type=covar_type, 
                                             ordinal=ordinal,
                                             non_lin=non_lin,
                                             free_biases=free_biases)

#             print('classic MODEL')
#             print(classic_model)
#             print(classic_model.state_dict().keys())    
#             print('FGN MODEL')
#             print(fgn_model)
#             print(fgn_model.state_dict().keys()) 

            # convert 
            fgnl.convert_classic_to_fgn(classic_model, fgn_model)

            # optimize every params that require grad
            if optimizer_name=='Adam':
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, fgn_model.parameters()), lr=opt_lr)

            # attempt to sent to GPU, else train over CPU
            model_sent_to_device = False
            sleep_time = 30
            while not model_sent_to_device and sleep_time<4800:
                # get free device
                device = torch.device('cuda')
                try:
                    device_id = GPUtil.getFirstAvailable(order='memory', maxLoad=1.0, maxMemory=0.8, verbose=False)[0]
                    # send to least used GPU
                    print('Using GPU:', device_id)
                    with torch.cuda.device(device_id):
                        fgn_model.to(device)
                        model_sent_to_device=True
                        
                except Exception as e:
                    print(e)
                    sleep_time = 1.66*sleep_time
                    print('GPU error. Wait {}s and continue'.format(sleep_time))
                    time.sleep(sleep_time)

            if not model_sent_to_device:
                print('Failed to send to GPU, using CPU')
                device = torch.device('cpu')
                fgn_model.to(device)

            # save config
            with open(model_folder+'/config.txt', 'w') as text_file:
                text_file.write('batch_size {}\n'.format(str(batch_size)))
                text_file.write('num_epochs {}\n'.format(1))
                text_file.write('lmbda_l2 {}\n'.format(str(lmbda_l2)))
                text_file.write('optimizer_name {}\n'.format(optimizer_name))
                text_file.write('opt_lr {}\n'.format(opt_lr))
                text_file.write('hidden_layer_sizes {}\n'.format(hidden_layer_sizes))
                text_file.write('drop_p {}\n'.format(drop_p))
                text_file.write('covar_type {}\n'.format(covar_type))
                text_file.write('ordinal {}\n'.format(ordinal))
                text_file.write('lmbda_sigma {}\n'.format(lmbda_sigma))
                text_file.write('free_biases {}\n'.format(free_biases))
                text_file.write('non_lin {}\n'.format(non_lin))

            # train 1 epoch
            training_done = False
            print('Training')
            try:
                train_res = fgnh.train(model=fgn_model, train_loader=mnist_train_loader, loss_func=loss, 
                                       optimizer=optimizer, epochs=1, save_hist=0, 
                                       pred_func=fgnh.cross_ent_pred_accuracy, test_loader=mnist_val_loader, 
                                       verbose=True)
                training_done=True
            except Exception as e:
                print(e)
                print('Training failed')

            # training finished, if succesful
            if training_done:
                # save model, histories
                # save trained model
                print('Saving trained model {} in {}'.format(model_name, model_folder))

                # save model entirely
                torch.save(fgn_model,model_folder+'/trained_converted_fgn_model_full.pth')

                # save model weights
                torch.save(fgn_model.state_dict(), model_folder+'/trained_converted_fgn_model_state_dict.pth')

                # save training histories as pickle
                # WHOLE HISTORIES are too large, only save loss and accuracy
                with open(model_folder+'/train_histories.pckl','wb') as pickle_file:
                    pickle.dump(train_res, pickle_file)

            # clean up GPU
            del fgn_model
            torch.cuda.empty_cache()
        else:
            print('trained_converted_fgn_model_full.pth already exists, skipping.')
            #end of FGN training
    
    # moving to the next classic model
    del classic_model
    torch.cuda.empty_cache()
    
# In[ ]:
print("SCRIPT END")