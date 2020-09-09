import numpy as np
import torch
from torchvision import datasets, transforms

from .mnist_dataloaders import mnist_dataloaders


def mnist_random_shuffled_dataloader(num_samples=1000, batch_size=1):
    # returns random 28*28 images (like MNIST)
    # creates them by shuffling the pixels around from MNIST training samples
    
    
    # get mnist train samples
    mnist_train_loader, _, _ = mnist_dataloaders(batch_size=batch_size)
    
    # get data
    data = mnist_train_loader.dataset.tensors[0].numpy()
    
    # applying shuffle 
    data = [shuffle_sample(x) for x in data]
    
    while len(data) < num_samples:
        data.extend([shuffle_sample(x) for x in data])
    
    # discard excess and convert to tensor
    data = torch.tensor(data[:num_samples])
    
    # recombine into dataloaders
    random_shuffled_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, ),
                                                     batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    
    return random_shuffled_dataloader


def shuffle_sample(x):
    # helper function, shuffles a single image
    x = x.flatten()
    np.random.shuffle(x)
    x.resize((28,28))
    return x
