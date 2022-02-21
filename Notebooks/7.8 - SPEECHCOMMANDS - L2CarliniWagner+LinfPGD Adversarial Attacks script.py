### script to run the 7.7 experiment from the command line

# attack the SPEECHCOMMAND models

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# manually set cuda device
# torch.cuda.set_device(1)
# device = 'cpu'
print(device)


import sys
sys.path.append('/home/felix/Research/Adversarial Research/FGN---Research/')
import Finite_Gaussian_Network_lib as fgnl
import Finite_Gaussian_Network_lib.fgn_helper_lib as fgnh

# load dataset
batch_size = 32
batchsize_for_val = 128
(train_loader, val_loader, test_loader) = fgnh.SpeechCommands_Dataloaders(resample_rate = 8000,
                                                                          batch_size = batch_size,
                                                                          batchsize_for_val = batchsize_for_val,
                                                                          num_workers=5, 
                                                                          pin_memory=True)


# define model classes

## classic model
class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2).squeeze()
    
# FGN model    
class FGN_M5(nn.Module):
    
    # changes:
    # nn.Conv1d -> fgnl.FGN_Conv1d
    # added g to conv inputs and outputs
    # make sure you pass g through the same pooling steps as x
    
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.fgn_conv1 = fgnl.FGN_Conv1d(in_channels=n_input, out_channels=n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.fgn_conv2 = fgnl.FGN_Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.fgn_conv3 = fgnl.FGN_Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.fgn_conv4 = fgnl.FGN_Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)
        
    def forward(self, x):
        x, g = self.fgn_conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        g = self.pool1(g)
        x, g = self.fgn_conv2(x, g)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        g = self.pool2(g)
        x, g = self.fgn_conv3(x ,g)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        g = self.pool3(g)
        x, _ = self.fgn_conv4(x, g)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2).squeeze()


# In[10]:


# pretrained models paths
save_path = '/home/felix/Research/Adversarial Research/FGN---Research/Experiments/sample_SPEECHCOMMANDS_models/'

classic_model_name= 'sample_classic_model_SPEECHCOMMANDS'
fgn_model_name = 'sample_FGN_model_SPEECHCOMMANDS'


# define and load the models
# classic model
classic_model = M5()
classic_model.load_state_dict(torch.load(save_path+classic_model_name+'_state_dict.pth'))
classic_model.to(device)

# fgn model trained from scratch
fgn_model_from_scratch = FGN_M5()
fgn_model_from_scratch.load_state_dict(torch.load(save_path+fgn_model_name+'_state_dict.pth'))
fgn_model_from_scratch.to(device)

# converted fgn model (no retraining)
fgn_model_converted_no_retraining = FGN_M5()
fgn_model_converted_no_retraining.load_state_dict(torch.load(save_path+'sample_FGN_converted_model_SPEECHCOMMANDS'+'_state_dict.pth'))
fgn_model_converted_no_retraining.to(device)

# converted and retrained 1 epoch fgn model
fgn_model_converted_fast_retraining = FGN_M5()
fgn_model_converted_fast_retraining.load_state_dict(torch.load(save_path+'sample_FGN_converted_fast_retrained_model_SPEECHCOMMANDS'+'_state_dict.pth'))
fgn_model_converted_fast_retraining.to(device)

# converted and retrained 21 epoch fgn model
fgn_model_converted_long_retraining = FGN_M5()
fgn_model_converted_long_retraining.load_state_dict(torch.load(save_path+'sample_FGN_converted_long_retrained_model_SPEECHCOMMANDS'+'_state_dict.pth'))
fgn_model_converted_long_retraining.to(device)


# set all models to eval mode
classic_model.eval()
fgn_model_from_scratch.eval()
fgn_model_converted_no_retraining.eval()
fgn_model_converted_fast_retraining.eval()
fgn_model_converted_long_retraining.eval()

### Start Attacking the models
import foolbox
import numpy as np

# set model bounds and preprocessing

# precomputed bounds min and max input values
min_bound = -1.3844940662384033
max_bound = 1.3773366212844849

bounds = (min_bound, max_bound)
# preprocessing - I think these would be used in similar way to pytorch preprocessing
# but possible passed to whatever architecture is used (torch, tensorflow, other) 
# in my case the dataloaders already normalizes the data
preprocessing = dict(mean=0, std=1)

# ready the models for foolbox
classic_f_model = foolbox.PyTorchModel(classic_model, bounds=bounds,
                                       preprocessing=preprocessing, device=device)

fgn_f_model_from_scratch = foolbox.PyTorchModel(fgn_model_from_scratch, bounds=bounds,
                                       preprocessing=preprocessing, device=device)

fgn_f_model_converted_no_retraining = foolbox.PyTorchModel(fgn_model_converted_no_retraining, bounds=bounds,
                                       preprocessing=preprocessing, device=device)

fgn_f_model_converted_fast_retraining = foolbox.PyTorchModel(fgn_model_converted_fast_retraining, bounds=bounds,
                                       preprocessing=preprocessing, device=device)

fgn_f_model_converted_long_retraining = foolbox.PyTorchModel(fgn_model_converted_long_retraining, bounds=bounds,
                                       preprocessing=preprocessing, device=device)

# attack params to explore
epsilons = torch.tensor([(max_bound-min_bound)*x 
            for x in 
            [0.0,
             1/256,
             3/512,
             1/128,
             3/256,
             1/64,
             3/128,
             1/32,
             3/64,
             1/16,
             3/32,
             1/8,
             3/16,
             1/4,
             3/8,
             1/2,
             3/4,
             1.0,] ], device=device)

print('epsilons: {}'.format(epsilons))


### Now, perform the attacks on the models, saving the results 


# In[30]:


### attack parameters
L2CarliniWagner_attack=foolbox.attacks.L2CarliniWagnerAttack()
LinfPGD_attack=foolbox.attacks.LinfPGD()


# targetted vs untargetted
from foolbox.criteria import Misclassification

# define the entire attack function using epsilons, criterion,
def L2CarliniWagner_attack_func(f_model, inputs, labels):
    device = f_model.device
    inputs = inputs.to(device)
    criterions = Misclassification(labels.to(device))
    return L2CarliniWagner_attack(model=f_model, inputs=inputs, criterion=criterions, epsilons=epsilons)

def LinfPGD_attack_func(f_model, inputs, labels):
    device = f_model.device
    inputs = inputs.to(device)
    criterions = Misclassification(labels.to(device))
    return LinfPGD_attack(model=f_model, inputs=inputs, criterion=criterions, epsilons=epsilons)


# In[31]:


# name for the models we are attacking
models_to_attack = {'classic_f_model':classic_f_model, 
                    'fgn_f_model_from_scratch':fgn_f_model_from_scratch, 
                    'fgn_f_model_converted_no_retraining':fgn_f_model_converted_no_retraining,
                    'fgn_f_model_converted_fast_retraining':fgn_f_model_converted_fast_retraining,
                    'fgn_f_model_converted_long_retraining':fgn_f_model_converted_long_retraining
                   }
# names of funcs for attacks
attacks_to_perform = {'L2CarliniWagner':L2CarliniWagner_attack_func,
                     'LinfPGD':LinfPGD_attack_func}

from time import time
import os

# timestamp
timestamp = time()
# save_folder = '/home/felix/Research/Adversarial Research/FGN---Research/Experiments/adversarial_attacks_results/{}/'.format(timestamp)
# print('creating save folder: {}'.format(save_folder)) 
# os.makedirs(save_folder)
save_folder = '/home/felix/Research/Adversarial Research/FGN---Research/Experiments/adversarial_attacks_results/1637624970.2306073/'
start = 48
stop = None

for attack_name, attack in attacks_to_perform.items():
    print('Performing attack:', attack_name)
    for model_name, f_model in models_to_attack.items():
        if not os.path.exists(save_folder+'{}_{}_{}_start_{}_stop_{}.pickle'.format(attack_name,
                                                                                    model_name,
                                                                                    'adv_raw', 
                                                                                    start,
                                                                                    stop)):
            print('Attacking', model_name)

            # do attack and save res
            fgnl.perform_attack(attack,
                                f_model,
                                val_loader,
                                save_file=save_folder+'{}_{}'.format(attack_name, model_name),
                                start=start,
                                stop=stop)
            
        else:
            print('skipping')
            print(attack_name, model_name)

            
    print()
print('EXP DONE')