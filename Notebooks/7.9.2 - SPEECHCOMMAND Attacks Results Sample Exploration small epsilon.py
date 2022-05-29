# analysis of the Carlini-Wagner and PGD attacks on SPEECHCOMMANDS

### 0 - prelims

import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle

import sys
sys.path.append('/home/felix/Research/Adversarial Research/FGN---Research/')
import Finite_Gaussian_Network_lib as fgnl
import Finite_Gaussian_Network_lib.fgn_helper_lib as fgnh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# manually set cuda device
torch.cuda.set_device(1)
device = 'cpu'
print(device)

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

# load all the models
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
model_name_pairs = {'Classic': classic_model,
                    'FGN_from_scratch': fgn_model_from_scratch,
                    'FGN_converted (no retraining)': fgn_model_converted_no_retraining,
                    'FGN_converted (fast retraining)': fgn_model_converted_fast_retraining,
                    'FGN_converted (long retraining)': fgn_model_converted_long_retraining
                   }

# attack results paths
attack_results_dir = '/scratch/felix/FGN---Results/cw_pgd_results_small_epsilon'
attack_results_dir_list = sorted(glob.glob(attack_results_dir+'/*', recursive = True))
all_attack_results = attack_results_dir_list

# load SPEECHCOMMANDS data
labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

# load dataset
batch_size = 32
batchsize_for_val = 128
(train_loader, val_loader, test_loader) = fgnh.SpeechCommands_Dataloaders(resample_rate = 8000,
                                                                          batch_size = batch_size,
                                                                          batchsize_for_val = batchsize_for_val,
                                                                          num_workers=5, 
                                                                          pin_memory=True)

val_set = val_loader.dataset # note these are not resampled

# necessary re-sampling
from torchaudio import transforms
transform = transforms.Resample(orig_freq=16000, new_freq=8000)

### values to iterate over
# max distance allowed based on epsilon
# precomputed bounds min and max input values
min_bound = -1.3844940662384033
max_bound = 1.3773366212844849
epsilons = [(max_bound-min_bound)*x*(1./256.) 
            for x in 
            [1./512,
             3./1024,
             1./256,
             3./512,
             1./128,
             3./256,
             1./64,
             3./128,
             1./32,
             3./64,
             1./16,
             3./32,
             1./8,
             3./16,
             1./4,
             3./8,
             1./2,
             3./4,
            ] ]

attack_names = ['L2CarliniWagner', 'LinfPGD']
model_name_to_path_name = {'Classic': 'classic_f_model',
                           'FGN_from_scratch': 'fgn_f_model_from_scratch',
                           'FGN_converted (no retraining)': 'fgn_f_model_converted_no_retraining',
                           'FGN_converted (fast retraining)': 'fgn_f_model_converted_fast_retraining',
                           'FGN_converted (long retraining)': 'fgn_f_model_converted_long_retraining'
                          }

batch_names_indices_pairs = [('start_0_stop_16', 0, 2048),
                             ('start_16_stop_32', 2048, 4096), 
                             ('start_32_stop_48', 4096, 6144),
                             ('start_48_stop_None', 6144, 9981)]

### dicts to build 
conf_wrongs = {a:{m:[0 for _ in range(len(epsilons))] 
                 for m in model_name_to_path_name.keys()}
              for a in attack_names}
conf_rights = {a:{m:[0 for _ in range(len(epsilons))] 
                 for m in model_name_to_path_name.keys()}
              for a in attack_names}
low_confs = {a:{m:[0 for _ in range(len(epsilons))] 
                 for m in model_name_to_path_name.keys()}
              for a in attack_names}
orig_preds_labels = {m:[] for m in model_name_to_path_name.keys()}
adv_confidences = {a:{m:[[] for _ in range(len(epsilons))] 
                 for m in model_name_to_path_name.keys()}
              for a in attack_names}


### main loop
for batch_name, start_idx, end_idx in batch_names_indices_pairs:
    print('working on batch', batch_name)
    print('start end indices', start_idx, end_idx)

    # original label and waveforms
    batch_orig_labels = []
    batch_orig_samples = []
    for val_set_index in range(start_idx, end_idx):
        waveform, _, label, _, _ = val_set[val_set_index]
        batch_orig_labels.append(label)
        batch_orig_samples.append(transform(waveform))
    batch_orig_samples = pad_sequence(batch_orig_samples).to(device)
        
    for model_name, model_path in list(model_name_to_path_name.items()):
        print('\t working on model', model_name)
        model = model_name_pairs[model_name]
        
        # original evals
        with torch.no_grad():
            batch_orig_model_preds = model(batch_orig_samples)
        batch_orig_preds_softmax = F.softmax(batch_orig_model_preds, dim=-1)
        batch_orig_preds_confidences, batch_orig_preds_labels_index = torch.max(batch_orig_preds_softmax, dim=-1)
        batch_orig_preds_labels = [labels[l] for l in batch_orig_preds_labels_index.numpy()]
        orig_preds_labels[model_name] +=  batch_orig_preds_labels
        
        # attacks
        for attack_name in attack_names:
            print('\t\t working on attack', attack_name)

            # load a batch of adv
            path_to_load = [p for p in all_attack_results 
                            if (attack_name in p 
                                and model_path in p
                                and batch_name in p
                                and 'adv_clipped' in p
                               )]
            assert(len(path_to_load)==1)
            path_to_load=path_to_load[0]
            print('\t\t loading', path_to_load)
            with open(path_to_load, 'rb') as f:
                batch_adv_samples = pickle.load(f)
            
            for eps_ind, eps in list(enumerate(epsilons)):
                print('\t\t\t working on epsilon', eps, len(batch_adv_samples[eps_ind]))

                # don't recompute for epsilon==0
                if eps==0.0:
                    batch_adv_preds_labels = batch_orig_preds_labels
                    batch_adv_preds_confidences = batch_orig_preds_confidences
                else:
                    # eval
                    with torch.no_grad():
                        batch_adv_model_preds = model(torch.tensor(batch_adv_samples[eps_ind]).to(device).float())
                    batch_adv_preds_softmax = F.softmax(batch_adv_model_preds, dim=-1)
                    batch_adv_preds_confidences, batch_adv_preds_labels = torch.max(batch_adv_preds_softmax, dim=-1)
                    batch_adv_preds_labels = [labels[l] for l in batch_adv_preds_labels.numpy()]
                    batch_adv_preds_confidences = batch_adv_preds_confidences.numpy()

                # tally up
                confidently_right = 0
                confidently_wrong = 0
                originaly_wrong = 0
                low_confidence = 0
                assert(len(batch_orig_labels)==len(batch_orig_preds_labels))
                assert(len(batch_adv_preds_labels)==len(batch_adv_preds_confidences))
                assert(len(batch_orig_labels)==len(batch_adv_preds_confidences))
                for (orig_label, 
                     orig_pred_label, 
                     adv_pred_label, 
                     adv_pred_conf) in zip(batch_orig_labels,
                                           batch_orig_preds_labels,
                                           batch_adv_preds_labels,
                                           batch_adv_preds_confidences):
                    # only look at cases where original model was correct
                    if orig_label == orig_pred_label:
                        if adv_pred_conf>=0.5:
                            if orig_label==adv_pred_label:
                                confidently_right+=1
                            else:
                                confidently_wrong+=1
                        else:
                            low_confidence+=1
                    else:
                        originaly_wrong+=1

                print('\t\t\t confidently right:', confidently_right)
                print('\t\t\t confidently wrong:', confidently_wrong)
                print('\t\t\t originaly wrong:', originaly_wrong)
                print('\t\t\t low confidence:', low_confidence)

                conf_wrongs[attack_name][model_name][eps_ind] += confidently_wrong
                conf_rights[attack_name][model_name][eps_ind] += confidently_right
                low_confs[attack_name][model_name][eps_ind] += low_confidence
                adv_confidences[attack_name][model_name][eps_ind]+= batch_adv_preds_confidences.tolist()
                
                
                
#  save
import json
save_results_dir = attack_results_dir
with open(save_results_dir+'/conf_wrongs.json', 'w', encoding ='utf8') as f:
    json.dump(conf_wrongs, f, ensure_ascii = True)
    
with open(save_results_dir+'/conf_rights.json', 'w', encoding ='utf8') as f:
    json.dump(conf_rights, f, ensure_ascii = True)

with open(save_results_dir+'/low_confs.json', 'w', encoding ='utf8') as f:
    json.dump(low_confs, f, ensure_ascii = True)

with open(save_results_dir+'/adv_confidences.json', 'w', encoding ='utf8') as f:
    json.dump(adv_confidences, f, ensure_ascii = True)