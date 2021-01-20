import torch
from torchvision import datasets, transforms

def mnist_dataloaders(batch_size=32, batch_size_for_val=None, mnist_path='/home/data/mnist'):
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
    # notes: as is the data is PIL images, and transforms is only applied when batches are called.
    # so original_mnist_train_loader.dataset.data[0] (raw data) is a PIL image (dtype=torch.uint8)
    # but next(iter(original_mnist_train_loader))[0] (after transforms) is a tensor (dtype=torch.float32)
    # ToTensor() essentially divides by 255 in this case
    
    # split into train/validation
    train_ds =  original_mnist_train_loader.dataset.data[:50000, :, :]
    valid_ds =  original_mnist_train_loader.dataset.data[50000:, :, :]

    train_dst =  original_mnist_train_loader.dataset.targets.data[:50000]
    valid_dst =  original_mnist_train_loader.dataset.targets.data[50000:]

    # apply transforms
    # keep in mind you should only use the mean+std of data you've 'seen', the train data
    train_ds = train_ds.float()/255.0
    train_mean = train_ds.mean()
    train_std = train_ds.std()
    train_ds = transforms.Normalize(mean=(train_mean,), std=(train_std,))(train_ds)
    valid_ds = valid_ds.float()/255.0
    valid_ds = transforms.Normalize(mean=(train_mean,), std=(train_std,))(valid_ds)

    # recombine into dataloaders
    mnist_train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_ds, train_dst),
                                                     batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    
    # batch_size for val and test should be much larger
    if batch_size_for_val==None:
        batch_size_for_val = 100*batch_size
    mnist_val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_ds, valid_dst),
                                                     batch_size=batch_size_for_val, shuffle=True, num_workers=0, pin_memory=False)

    ## Don't use test set until paper
    mnist_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(mnist_path, train=False, download=False, 
                       transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((train_mean,), (train_std,))])
                      ), 
            batch_size=batch_size_for_val, shuffle=True)
    
#     # values used for other things
#     pre-normalization means and std
#     train_mean = 0.130961298942565917968750000000
#     train_std = 0.308494895696640014648437500000
#     # minimum/maximum pixel value post normalization, from train dataset
#     min_pix = -0.4242129623889923095703125
#     max_pix =  2.8214867115020751953125000
#     # mean and std of train set post normalization
#     train_mean = -0.0000061691193877777550369501113891601562500000000
#     train_std = 0.999999344348907470703125000000
    
    return(mnist_train_loader, mnist_val_loader, mnist_test_loader)