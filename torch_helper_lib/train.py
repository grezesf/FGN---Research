# a pytorch model training function

import torch
import numpy as np
from AverageMeter import AverageMeter

def train(model, device, train_loader, loss_func, optimizer, epochs, save_hist=0, **kwargs):
    # trains a model over train_loader data for epochs using optimizer and loss_func
    # returns the accuracy over training
    
    # if save_hist = 0, returns only train loss and accuracy history
    # if save_hist = 1, also returns the history of each trainable param of the model for each epoch
    # if save_hist = 2, also logs the history at each update (after each batch of data)
    
    # used kwargs:
    # verbose: boolean, used to print training stats
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    # pred_func: function, used to compute number of correct predictions    
    pred_func = kwargs['pred_func'] if 'pred_func' in kwargs else None
    
    # model: a Pytorch module
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model is not pytorch module")
    # device: a pytorch device 
    if not isinstance(device, torch.device):
        raise TypeError("device is not pytorch device")
    # train_loader: a pytorch data loader
    if not isinstance(train_loader, torch.utils.data.dataloader.DataLoader):
        raise TypeError("train_loader is not pytorch dataloader")
    # loss_func: a pytorch loss function (can be any function)
    if not callable(loss_func):
        raise TypeError("loss_func is not a function")
    # optimizer: a pytorch optimizer  
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer is not a pytorch optimizer")

    
    # objects to return
    train_loss_hist = []
    rolling_losses = AverageMeter()
    train_acc_hist = []
    # histories to save (the trainable params)
    histories = {}
    for (name,param) in filter(lambda (_,p): p.requires_grad, model.named_parameters() ):
        histories[name] = np.expand_dims(param.cpu().detach().numpy(), axis=0)

    for epoch in range(0, epochs):

        # reset loss and acc
        rolling_losses.reset()
        correct = 0

        # set model to trainable mode
        model.train()
        # load a batch
        for batch_idx, (data, target) in enumerate(train_loader):
            # load batch data, targets to device
            data, target = data.to(device), target.to(device)
            # reset optimizer gradients
            optimizer.zero_grad()
            # compute predictions
            output = model(data)
            # compute loss
            loss = loss_func(model=model, output=output, target=target)

            # update rolling average loss
            rolling_losses.update(loss.item(), data.size(0) )
            # update predictions
            if pred_func is not None:
                correct += pred_func(output=output, target=target)

            # propagate gradients and store them
            loss.backward()
            # apply stored gradients to parameters
            optimizer.step()

            # update histories        
            if save_hist==2:
                for (name,param) in filter(lambda (_,p): p.requires_grad, model.named_parameters() ):
                    histories[name] = np.append(histories[name], np.expand_dims(param.cpu().detach().numpy(), axis=0), axis=0)

            #end of batch

        # update histories        
        if save_hist==1:
            for (name,param) in filter(lambda (_,p): p.requires_grad, model.named_parameters() ):
                histories[name] = np.append(histories[name], np.expand_dims(param.cpu().detach().numpy(), axis=0), axis=0)
        
        train_loss_hist.append(rolling_losses.avg)
        if pred_func is not None:
            acc = 100. * correct / len(train_loader.dataset)
            train_acc_hist.append(acc)

        # print epoch stats
        if verbose:
            print('Train set: Average loss: {:.4f}'.format(
                rolling_losses.avg))
        if pred_func is not None:
            print('Train set: Accuracy: {}/{} ({:.0f}%)'.format(
                correct, len(train_loader.dataset),acc))

    # return desired tuple
    ret = (train_loss_hist,)
    if pred_func is not None:
        ret += (train_acc_hist,)
    if save_hist in [1,2]:
        ret += (histories,)
        
    return(ret)
               
               
               