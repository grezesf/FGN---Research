import numpy as np
import torch
from torchvision import datasets, transforms

def mnist_random_dataloader(num_samples=1000, batch_size=1):

    # create random 28*28 images
    # pre-normalization means and std from train set
    train_mean = 0.130961298942565917968750000000
    train_std = 0.308494895696640014648437500000
    
    # Random images dataset
    x_rand = np.random.randint(low=0, high=255+1, size=(num_samples,28,28) )
    # convert to [0,1] and apply same transforms as for MNIST training
    x_rand = x_rand/255
    x_rand = x_rand-train_mean
    x_rand = x_rand/train_std
    

    x_rand = torch.Tensor(x_rand)
    # note: because we normalize with the train set mean and std, this random data is NOT normalized to (0,1)
    # this is expected. minpix and maxpix should be the same though
    rand_dataset = torch.utils.data.TensorDataset(x_rand)# create your dataset
    rand_dataloader = torch.utils.data.DataLoader(rand_dataset, 
                                                  batch_size=batch_size, 
                                                  ) # create your dataloader
    
    return rand_dataloader