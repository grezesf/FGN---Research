
# coding: utf-8
from __future__ import print_function

# In[1]:

# In[2]:

# In[3]:


import torch
import time
import itertools
from datetime import datetime
import GPUtil

import sys
sys.path.append('/home/felix/Research/Adversarial Research/FGN---Research/')
import Finite_Gaussian_Network_lib as fgnl
import Finite_Gaussian_Network_lib.fgn_helper_lib as fgnh


# In[4]:


# where to save results

save_dir = "/home/felix/Research/Adversarial Research/FGN---Research/Experiments/Exp-00_train_models/"

# fixed experiment parameters
num_iter = 3
batch_size = 32
(mnist_train_loader, mnist_val_loader, mnist_test_loader) = fgnh.mnist_dataloaders(batch_size=batch_size)
in_feats = 28*28
out_feats = 10
num_epochs = 5
drop_p = 0.2
lmbda_l2 = (4.0*0.1/len(mnist_train_loader.dataset))
device = torch.device('cuda')
optimizer = 'Adam'
lr = 0.001


with open(save_dir+"shared_parameters.txt", "w") as text_file:
    text_file.write("batch_size {}\n".format(str(batch_size)))
    text_file.write("num_epochs {}\n".format(str(num_epochs)))
    text_file.write("drop_p {}\n".format(str(drop_p)))
    text_file.write("lmbda_l2 {}\n".format(str(lmbda_l2)))
    text_file.write("optimizer {}\n".format(optimizer))
    text_file.write("lr {}\n".format(lr))


# In[5]:


# parameters to explore
# width of the network
hidden_layer_sizes_to_try = [16, 64, 256, 1024]
# depth of the network
number_of_hidden_layers_to_try = [0, 1, 2, 3]
# covariance type
covar_types_to_try = ['sphere', 'diag']
# various loss sigmas to try times lmbda_l2
lmbda_sigma_to_try = [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]


# In[6]:


# obviously try both classic and FGN
network_types_to_try = ['classic', 'fgn']


# In[7]:


# list with a bunch of dicts which represent the kwargs for an experiment
exp_params_to_try = []


# define the width and depth of network to try
hidden_layer_params_to_try = []
# add the network with no hidden layers
hidden_layer_params_to_try.append([])

for (num_layers, layer_sizes) in itertools.product(number_of_hidden_layers_to_try[1:], hidden_layer_sizes_to_try):
    hidden_layer_params_to_try.append([layer_sizes for _ in range(num_layers)])

hidden_layer_params_to_try = list(itertools.product(hidden_layer_params_to_try, network_types_to_try))


fgn_params_to_try = list(itertools.product(lmbda_sigma_to_try, covar_types_to_try))

# In[8]:
hidden_layer_params_to_try.reverse()

# define all the experiments to run
for (ite, exp_p) in itertools.product(range(num_iter),hidden_layer_params_to_try):
    hidden_layer_sizes, network_type = exp_p

    if network_type == 'classic':
            kwargs = {'hidden_layer_sizes':hidden_layer_sizes,
                      'network_type':network_type,
                      'ite':ite                
            }
            # add to exp to try
            exp_params_to_try.append(kwargs)

    elif network_type == 'fgn':
        for (lmbda_sigs, covar_type) in fgn_params_to_try:
            kwargs = {'hidden_layer_sizes':hidden_layer_sizes,
                      'network_type':network_type,
                      'ite':ite,
                      'lmbda_sigs':lmbda_sigs,
                      'covar_type':covar_type
            }
            # add to exp to try
            exp_params_to_try.append(kwargs)

    else:
        # error
        print("Error, wrong network type")


# In[9]:

# In[10]:


def define_model_loss_name_from_kwargs(**kwargs):
    
    # given a bunch of kwargs that define an experiment to run, creates and returns the mode, loss and name
    
    # list of used kwargs 
    # for both network definitions
    network_type = kwargs['network_type']
    in_feats = kwargs['in_feats']
    out_feats = kwargs['out_feats']
    hidden_layer_sizes = kwargs['hidden_layer_sizes']
    lmbda_l2 = kwargs['lmbda_l2']
    
    # for fgns
    if network_type=='fgn':
        lmbda_sigs = kwargs['lmbda_sigs']*lmbda_l2
        covar_type = kwargs['covar_type']
    
    # used by both
    timestamp = kwargs['timestamp']
    ite = kwargs['ite']

    if network_type=='classic':
        model = fgnl.Feedforward_Classic_net(in_feats=in_feats, out_feats=out_feats, hidden_layer_sizes=hidden_layer_sizes)
        loss  = fgnh.def_classical_cross_ent_loss(lmbda_l2=lmbda_l2)
        name = "_".join((str(timestamp), str(hidden_layer_sizes), network_type, str(ite)))

    elif network_type == 'fgn':
        model = fgnl.Feedforward_FGN_net(in_feats=in_feats, out_feats=out_feats, hidden_layer_sizes=hidden_layer_sizes, 
                                         covar_type=covar_type)
        loss = fgnl.def_fgn_cross_ent_loss(lmbda_l2=lmbda_l2, lmbda_sigs=lmbda_sigs*lmbda_l2)
        name = "_".join((str(timestamp), str(hidden_layer_sizes), network_type, covar_type, 'lsig{:.4E}'.format(lmbda_sigs), str(ite)))

    
    return model, loss, name


# In[11]:


for kwargs in exp_params_to_try:
    print(kwargs)
    timestamp = datetime.now()
    print(str(timestamp))

    # define model from kwargs
    model, loss, model_name = define_model_loss_name_from_kwargs(in_feats=28*28, out_feats=10, timestamp=timestamp, 
                                                                 lmbda_l2=lmbda_l2, **kwargs)
    print("Model name:", model_name)
    
    # save parameters
    with open(save_dir+model_name+"_parameters.txt", "w") as text_file:
        for key in kwargs.keys():
            if key != 'ite':
                text_file.write("{} {}\n".format(key, kwargs[key]))

    # attempt to sent to GPU
    model_sent_to_device = False
    while not model_sent_to_device:
        # get free device
        device_id = GPUtil.getFirstAvailable(order='memory', maxLoad=1.0, maxMemory=0.8, verbose=False)[0]

        # send to least used GPU
        print("Using GPU:", device_id)
        with torch.cuda.device(device_id):
            # send to device
            try:
                model.to(device)
                model_sent_to_device=True
            except:
                print("Not enough ram. Wait 30s and continue")
                time.sleep(30)
    
    # optimize every params that require grad
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=0.001)
    
    print(model)

    # train model
    try:
        pass
        # save results pre-training
        print("Saving initial (before training) model {} in {}".format(model_name, save_dir))

        # save model entirely
        torch.save(model, save_dir+model_name+"_init_weights_full.pth")

        # save model weights
        torch.save(model.state_dict(), save_dir+model_name+"_init_weights_state_dict.pth")
        
        print("Training")
        train_res = fgnh.train(model=model, train_loader=mnist_train_loader, loss_func=loss, 
                               optimizer=optimizer, epochs=num_epochs, save_hist=2, 
                               pred_func=fgnh.cross_ent_pred_accuracy, test_loader=mnist_val_loader, 
                               verbose=True) 
        
        # save trained model
        print("Saving trained model {} in {}".format(model_name, save_dir))

        # save model entirely
        torch.save(model, save_dir+model_name+"_trained_weights_full.pth")

        # save model weights
        torch.save(model.state_dict(), save_dir+model_name+"_trained_weights_state_dict.pth")
        
        # save training histories
        with open(save_dir+model_name+"_training_history.txt", "w") as text_file:
            for key in train_res.keys():
                text_file.write("{} {}\n".format(key, train_res[key]))

        
    except:
        print("Training failed. Moving on to next exp" )
        
    # clean up GPU space?
    torch.cuda.empty_cache()

print("Exp-00 Done!")
