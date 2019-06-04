# a pytorch model testing function

import torch
import numpy as np
from AverageMeter import AverageMeter

def test(model, test_loader, loss_func, **kwargs):
    
    # tests a pytorch model, using device, over test_loader data, using loss_func
    # returns the loss, and accuracy if applicable
    
    ### used kwargs:
    # verbose: boolean, used to print training stats
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    # device: a pytorch device, used to tell where to run the model
    if 'device' in kwargs:
        # check device type
        if not isinstance(kwargs['device'], torch.device):
            raise TypeError("device is not pytorch device")
        # give warning
        if verbose:
            print("Warning: device specified. This might change model location (cuda<->cpu)")
        device = kwargs['device']
        change_device = True
    else:
        # get device from module (will run into probs if on multiple gpus
        device = next(model.parameters()).device
        change_device = False
    # pred_func: function, used to compute number of correct predictions    
    pred_func = kwargs['pred_func'] if 'pred_func' in kwargs else None
    

    ### type checks
    # model: a Pytorch module
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model is not pytorch module")
    # test_loader: a pytorch data loader
    if not isinstance(test_loader, torch.utils.data.dataloader.DataLoader):
        raise TypeError("test_loader is not pytorch dataloader")
    # loss_func: a pytorch loss function (can be any function)
    if not callable(loss_func):
        raise TypeError("loss_func is not a function")
    
    # send model to device
    if change_device:
        model.to(device)
    
    # set model to eval mode
    model.eval()
    
    # values to return
    test_loss = 0
    correct = 0
    rolling_losses = AverageMeter()

    # start testing (no grad computation)
    with torch.no_grad():
        for data, target in test_loader:
            # send data to devices
            data, target = data.to(device), target.to(device)
            # compute outputs
            output = model(data)
            # compute and add loss
            test_loss = loss_func(model=model, output=output, target=target)
            # update rolling average loss
            rolling_losses.update(test_loss.item(), data.size()[0] )
            # update predictions
            if pred_func is not None:
                correct += pred_func(output=output, target=target)
            
    # average loss 
    test_loss = rolling_losses.avg
    # accuracy
    if pred_func is not None:
        acc = 100. * correct / len(test_loader.dataset)
        
    if verbose:
        if pred_func is not None:
            pred_string = ', Accuracy: {}/{} ({:.0f}%)'.format(correct, len(test_loader.dataset), acc)
        else:
            pred_string = ''
        print('Test set - Average loss: {:.4f}'.format(test_loss) + pred_string)
            
    # dict to return
    ret = {'test_loss': test_loss}
    if pred_func is not None:
        ret['test_accuracy'] = acc

    return(ret)