
# coding: utf-8

# In[ ]:


# Exp-00
# train FGN feedforward networks
# command to run 
# stdbuf -o 0 python Exp-00-Train_FGN.py 2>&1 | tee Exp-00-log-2019-09-13-00:56_FGN_train.txt


# In[ ]:


from __future__ import print_function


# In[ ]:


import numpy as np
import time
import os
import sys
import GPUtil
import pickle

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
save_dir = base_dir+'/'+exp_dir+'/FGNs/'
# create FGN subfolder
try:
    os.mkdir(save_dir)
except Exception as e:
    print(e)


# In[ ]:
num_iter = 5


# In[ ]:
shared_config_fixed_parameters_dic = np.load(base_dir+'/shared_config_fixed_parameters_dic.pckl',allow_pickle='TRUE')
print("Shared Config")
print("{}".format(shared_config_fixed_parameters_dic))


# In[ ]:


# extract params
batch_size = shared_config_fixed_parameters_dic['batch_size']
optimizer_name = shared_config_fixed_parameters_dic['optimizer_name']
lmbda_l2 = shared_config_fixed_parameters_dic['lmbda_l2']
opt_lr = shared_config_fixed_parameters_dic['opt_lr']
num_epochs = shared_config_fixed_parameters_dic['num_epochs']


# In[ ]:


# load list of all classic configs
shared_config_experiments_list = np.load(base_dir+'/shared_config_experiments_list.pckl',allow_pickle='TRUE')


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
#  do 5 iterations
for ite in range(num_iter):
    # for each classic config
    for classic_config in shared_config_experiments_list:
        for fgn_config in fgn_config_experiments_list:
        
            # extract config values:
            hidden_layer_sizes = classic_config['hidden_layer_sizes']
            drop_p = classic_config['drop_p']
            covar_type = fgn_config['covar_type']
            ordinal = fgn_config['ordinal']
            lmbda_sigma = fgn_config['lmbda_sigma']

            # def model name
            model_name = 'fgn_hl{}_dp{}_{}_ord{}_ls{}_ite{}'.format(hidden_layer_sizes, 
                                                                       drop_p, 
                                                                       covar_type, 
                                                                       ordinal,
                                                                       lmbda_sigma,
                                                                       ite)

            print('Current model name: {}'.format(model_name))
            
            # create folder
            model_folder = save_dir+'/'+model_name+'/'
            try:
                os.mkdir(model_folder)
            except Exception as e:
                print(e)
            
            # check if trained pth file already exists
            if not os.path.exists(model_folder+'/trained_fgn_model_full.pth'):


                # create fgn network
                model = fgnl.Feedforward_FGN_net(in_feats=28*28, 
                                                 out_feats=10, 
                                                 hidden_layer_sizes=hidden_layer_sizes,
                                                 drop_p = drop_p, 
                                                 covar_type=covar_type, 
                                                 ordinal=ordinal,
                                                 non_lin=non_lin,
                                                 free_biases=free_biases
                                                 )

                # for FGNs, init with data centers for first layer
                model.set_first_layer_centers(mnist_test_loader)

                # loss for fgn uses lmbda_sigma
                loss  = fgnl.def_fgn_cross_ent_loss(lmbda_l2=lmbda_l2, lmbda_sigs=lmbda_sigma)

#                 print('MODEL')
#                 print(model)

                # optimize every params that require grad
                if optimizer_name=='Adam':
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt_lr)

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
                            model.to(device)
                            model_sent_to_device=True

                    except Exception as e:
                        print(e)
                        sleep_time = 1.66*sleep_time
                        print('GPU error. Wait {}s and continue'.format(sleep_time))
                        time.sleep(sleep_time)

                if not model_sent_to_device:
                    print('Failed to send to GPU, using CPU')
                    device = torch.device('cpu')
                    model.to(device)


                # save model pre_training
                print('Saving initial (before training) model {} in {}'.format(model_name, save_dir))
                # save model entirely
                torch.save(model, model_folder + '/init_fgn_model_full.pth')
                # save model weights
                torch.save(model.state_dict(), model_folder + '/init_fgn_model_state_dict.pth')
                # save config
                with open(model_folder+'/config.txt', 'w') as text_file:
                    text_file.write('batch_size {}\n'.format(str(batch_size)))
                    text_file.write('num_epochs {}\n'.format(str(num_epochs)))
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

                # train model
                training_done = False
                print('Training')
                try:
                    train_res = fgnh.train(model=model, train_loader=mnist_train_loader, loss_func=loss, 
                                           optimizer=optimizer, epochs=num_epochs, save_hist=0, 
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
                    torch.save(model,model_folder+'/trained_fgn_model_full.pth')

                    # save model weights
                    torch.save(model.state_dict(), model_folder+'/trained_fgn_model_state_dict.pth')

                    # save training histories as pickle
                    # WHOLE HISTORIES are too large, only save loss and accuracy
                    with open(model_folder+'/train_histories.pckl','wb') as pickle_file:
                        pickle.dump(train_res, pickle_file)

                # clean up GPU
                del model
                torch.cuda.empty_cache()
            
            else:
                print('trained_fgn_model_full.pth already exists')

# In[ ]:


print("SCRIPT END")

