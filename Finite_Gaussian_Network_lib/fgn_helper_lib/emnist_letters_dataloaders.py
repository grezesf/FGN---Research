import torch
from torchvision import datasets, transforms
import numpy as np 

def emnist_letters_dataloaders(batch_size=32, batch_size_for_val=None, emnist_path='/home/data/emnist'):
    # customized EMNIST letters dataset and dataloaders declaration
    # transforms does both the conversion from 0-255 to 0-1
    # and normalizes by the mean and std

    # returns train (), validation () and test () dataloaders
    # should be more or less balanced by classes
    
    # values from mnist train data
    train_mean = 0.130961298942565917968750000000
    train_std = 0.308494895696640014648437500000
    
    original_emnist_train_loader = torch.utils.data.DataLoader(
        datasets.EMNIST(emnist_path,
                        split='letters',
                        train=True,
                        download=False, 
                        transform=transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((train_mean,), (train_std,))]),
                       ), 
        batch_size=batch_size,
        shuffle=True)
    
    # train/val split
    cutoff = int( np.shape(original_emnist_train_loader.dataset.data.numpy())[0]*9/10)
    # split into train/validation
    train_ds =  torch.transpose(original_emnist_train_loader.dataset.data[:cutoff, :, :],1,2)
    valid_ds =  torch.transpose(original_emnist_train_loader.dataset.data[cutoff:, :, :],1,2)

    train_dst =  original_emnist_train_loader.dataset.targets.data[:cutoff]
    valid_dst =  original_emnist_train_loader.dataset.targets.data[cutoff:]
    
    # apply transforms
    # keep in mind you should only use the mean+std of data you've 'seen', the train data
    train_ds = train_ds.float()/255
    train_ds = transforms.Normalize(mean=(train_mean,), std=(train_std,))(train_ds)
    valid_ds = valid_ds.float()/255
    valid_ds = transforms.Normalize(mean=(train_mean,), std=(train_std,))(valid_ds)
    
    # recombine into dataloaders
    emnist_train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_ds, train_dst),
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=0,
                                                      pin_memory=False)
    
    # batch_size for val and test should be much larger
    if batch_size_for_val==None: batch_size_for_val = 100*batch_size
    emnist_val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_ds, valid_dst),
                                                    batch_size=batch_size_for_val,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    pin_memory=False)

    ## Don't use test set until paper
    original_emnist_test_loader = torch.utils.data.DataLoader(
        datasets.EMNIST(emnist_path,
                        split='letters',
                        train=False,
                        download=False, 
                        transform=transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((train_mean,), (train_std,))]),
                       ), 
        batch_size=batch_size_for_val,
        shuffle=True)
    
    # apply transforms
    test_ds = torch.transpose(original_emnist_test_loader.dataset.data,1,2)
    test_dst = original_emnist_test_loader.dataset.targets.data
    test_ds = test_ds.float()/255
    test_ds = transforms.Normalize(mean=(train_mean,), std=(train_std,))(test_ds)
    
    emnist_test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_ds, test_dst),
        batch_size=batch_size_for_val,
        shuffle=True,
        num_workers=0,
        pin_memory=False)
    
    return(emnist_train_loader, emnist_val_loader, emnist_test_loader)