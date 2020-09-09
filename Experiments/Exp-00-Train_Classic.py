
# coding: utf-8

# In[1]:


# Exp-00
# train classic feedforward networks
# command to run 
# stdbuf -o 0 python Exp-00-run.py 2>&1 | tee Exp-00-log-2019-09-13-00:56.txt


# In[2]:


from __future__ import print_function


# In[3]:


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


# In[4]:


# Preliminaries
# saved in :
save_dir =  "/home/felix/Research/Adversarial Research/FGN---Research/Experiments/Exp-00/"


# In[5]:


# create timestamped folder
timestamp = time.strftime("%Y_%m_%d_at_%H:%M:%S")
print(timestamp)
try:
    os.mkdir(save_dir + '/' + timestamp)
except e:
    pass


# In[6]:


num_iter = 5


# In[7]:


# load shared configs
shared_config_fixed_parameters_dic = np.load(save_dir+'/shared_config_fixed_parameters_dic.pckl',allow_pickle='TRUE')


# In[8]:


shared_config_fixed_parameters_dic = np.load(save_dir+'/shared_config_fixed_parameters_dic.pckl',allow_pickle='TRUE')
print("Shared Config")
print("{}".format(shared_config_fixed_parameters_dic))


# In[9]:


# extract params
batch_size = shared_config_fixed_parameters_dic['batch_size']
optimizer_name = shared_config_fixed_parameters_dic['optimizer_name']
lmbda_l2 = shared_config_fixed_parameters_dic['lmbda_l2']
opt_lr = shared_config_fixed_parameters_dic['opt_lr']
num_epochs = shared_config_fixed_parameters_dic['num_epochs']


# In[10]:


# load list of all configs
shared_config_experiments_list = np.load(save_dir+'/shared_config_experiments_list.pckl',allow_pickle='TRUE')


# In[11]:


# load data
(mnist_train_loader, mnist_val_loader, mnist_test_loader) = fgnh.mnist_dataloaders(batch_size=batch_size)


# In[12]:


#  do 5 iterations
for ite in range(num_iter):
    # for each config
    for config in shared_config_experiments_list:
        
        # extract config values:
        hidden_layer_sizes = config['hidden_layer_sizes']
        drop_p = config['drop_p']
        
        # def model name
        model_name = 'classic_hl{}_dp{}_ite{}'.format(hidden_layer_sizes, drop_p, ite)
        
        print("Current model name: {}".format(model_name))
        
        # create folder
        model_folder = save_dir+'/'+timestamp+'/'+model_name+'/'
        os.mkdir(model_folder)

        # create classic network
        model = fgnl.Feedforward_Classic_net(in_feats=28*28, 
                                             out_feats=10, 
                                             hidden_layer_sizes=hidden_layer_sizes, 
                                             drop_p=drop_p)
        loss  = fgnh.def_classical_cross_ent_loss(lmbda_l2=lmbda_l2)
        
        print("MODEL")
        print(model)
        
        # optimize every params that require grad
        if optimizer_name=='Adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt_lr)
        
        # attempt to sent to GPU, else train over CPU
        model_sent_to_device = False
        sleep_time = 30
        while not model_sent_to_device and sleep_time<4800:
            # get free device
            device = torch.device('cuda')
            device_id = GPUtil.getFirstAvailable(order='memory', maxLoad=1.0, maxMemory=0.8, verbose=False)[0]

            # send to least used GPU
            print("Using GPU:", device_id)
            with torch.cuda.device(device_id):
                # send to device
                try:
                    model.to(device)
                    model_sent_to_device=True
                except Exception as e:
                    print(e)
                    sleep_time = 1.66*sleep_time
                    print("Not enough ram. Wait {}s and continue").format(sleep_time)
                    time.sleep(sleep_time)
        
        if not model_sent_to_device:
            print("Failed to send to GPU, using CPU")
            device = torch.device('cpu')
            model.to(device)
            
        
        # save model pre_training
        print("Saving initial (before training) model {} in {}".format(model_name, save_dir))
        # save model entirely
        torch.save(model, model_folder + "/init_classic_model_full.pth")
        # save model weights
        torch.save(model.state_dict(), model_folder + "/init_classic_model_state_dict.pth")
        # save config
        with open(model_folder+"/config.txt", "w") as text_file:
            text_file.write("batch_size {}\n".format(str(batch_size)))
            text_file.write("num_epochs {}\n".format(str(num_epochs)))
            text_file.write("lmbda_l2 {}\n".format(str(lmbda_l2)))
            text_file.write("optimizer_name {}\n".format(optimizer_name))
            text_file.write("opt_lr {}\n".format(opt_lr))
            text_file.write("hidden_layer_sizes {}\n".format(hidden_layer_sizes))
            text_file.write("drop_p {}\n".format(drop_p))

        
        # train model
        training_done = False
        print("Training")
        try:
            train_res = fgnh.train(model=model, train_loader=mnist_train_loader, loss_func=loss, 
                                   optimizer=optimizer, epochs=num_epochs, save_hist=0, 
                                   pred_func=fgnh.cross_ent_pred_accuracy, test_loader=mnist_val_loader, 
                                   verbose=True)
            training_done=True
        except Exception as e:
            print(e)
            print("Training failed")
        
        # training finished, if succesful
        if training_done:
            # save model, histories
            # save trained model
            print("Saving trained model {} in {}".format(model_name, model_folder))

            # save model entirely
            torch.save(model,model_folder+"/trained_classic_model_full.pth")

            # save model weights
            torch.save(model.state_dict(), model_folder+"/trained_classic_model_state_dict.pth")

            # save training histories as pickle
            # WHOLE HISTORIES are too large, only save loss and accuracy
            with open(model_folder+"/train_histories.pckl","wb") as pickle_file:
                pickle.dump(train_res, pickle_file)
        
        # clean up GPU
        del model
        torch.cuda.empty_cache()

# In[13]:
print("SCRIPT END")