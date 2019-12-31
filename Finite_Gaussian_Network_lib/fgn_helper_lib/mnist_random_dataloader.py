import numpy as np
import torch
from torchvision import datasets, transforms

def mnist_random_dataloader(num_samples=1000, batch_size=1):

    # create random 28*28 images

    # Random images dataset
    x_rand = np.random.randint(low=0, high=255, size=(num_samples,28,28) )
    # convert to [0,1] and apply same transforms as for MNIST training
    x_rand = x_rand/255.0
    
    # mean and std of MNIST train set
    train_mean = -0.0000061691193877777550369501113891601562500000000
    train_std = 0.999999344348907470703125000000

    x_rand = torch.Tensor(x_rand)
    x_rand = transforms.Normalize(mean=(train_mean,), std=(train_std,))(x_rand)
    rand_dataset = torch.utils.data.TensorDataset(x_rand)# create your dataset
    rand_dataloader = torch.utils.data.DataLoader(rand_dataset, 
                                                  batch_size=batch_size, 
                                                  ) # create your dataloader
    
    return rand_dataloader