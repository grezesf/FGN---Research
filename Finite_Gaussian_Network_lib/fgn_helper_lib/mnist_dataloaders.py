import torch
from torchvision import datasets, transforms

def mnist_dataloaders(batch_size=32, mnist_path='/home/felix/Research/Adversarial Research/MNIST-dataset'):
    # customized MNIST dataset and dataloaders declaration
    # transforms does both the conversion from 0-255 to 0-1
    # and normalizes by the mean and std
    
    # returns train (50K), validation (10K) and test ] (10K) dataloaders
    # should be more or less balanced by classes
    
    original_mnist_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(mnist_path, train=True, download=False, 
                       transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
                ])), 
            batch_size=batch_size, shuffle=True)

    # split into train/validation
    train_ds =  original_mnist_train_loader.dataset.data[:50000, :, :]
    valid_ds =  original_mnist_train_loader.dataset.data[50000:, :, :]

    train_dst =  original_mnist_train_loader.dataset.targets.data[:50000]
    valid_dst =  original_mnist_train_loader.dataset.targets.data[50000:]

    # apply transforms
    # keep in mind you should only use the mean+std of data you've 'seen', the train data
    train_ds=train_ds.float()/255.0
    train_mean = train_ds.mean()
    train_std = train_ds.std()
    train_ds = transforms.Normalize(mean=(train_mean,), std=(train_std,))(train_ds)
    valid_ds=valid_ds.float()/255.0
    valid_ds = transforms.Normalize(mean=(train_mean,), std=(train_std,))(valid_ds)

    # recombine into dataloaders
    mnist_train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_ds, train_dst),
                                                     batch_size=batch_size, shuffle=True, )
    mnist_val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_ds, valid_dst),
                                                     batch_size=batch_size, shuffle=True)

    ## Don't use test set until paper
    mnist_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../MNIST-dataset', train=False, download=False, 
                       transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((train_mean,), (train_std,))])
                      ), 
            batch_size=batch_size, shuffle=True)
    
    
    return(mnist_train_loader, mnist_val_loader, mnist_test_loader)