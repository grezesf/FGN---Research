# a pytorch model training function

import torch
import numpy as np
from AverageMeter import AverageMeter
from test import test


def train(model, train_loader, loss_func, optimizer, epochs, save_hist=0, **kwargs):
    
    # trains a model, using device, over train_loader data for epochs using optimizer and loss_func
    # returns the history of the loss over training
    
    # if save_hist = 0, returns only train loss, and accuracy history if applicable
    # if save_hist = 1, also returns the history of each trainable param of the model for each epoch
    # if save_hist = 2, also logs the history at each update (after each batch of data)
    
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
            print("Warning: device specified. This might change model and optimizer location (cuda<->cpu)")
        device = kwargs['device']
        change_device = True
    else:
        # get device from module (will run into probs if on multiple gpus)
        device = next(model.parameters()).device
        change_device = False
    # pred_func: function, used to compute number of correct predictions    
    pred_func = kwargs['pred_func'] if 'pred_func' in kwargs else None
    if pred_func is not None and not callable(pred_func):
        raise TypeError("pred_func is not a function")
    # test_loader: pytorch data loader for test loss and acc
    test_loader = kwargs['test_loader'] if 'test_loader' in kwargs else None
    if test_loader is not None and not isinstance(test_loader, torch.utils.data.dataloader.DataLoader):
        raise TypeError("test_loader is not pytorch dataloader")
    
    ### type checks
    # model: a Pytorch module
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model is not pytorch module")
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
    test_lost_hist = []
    test_acc_hist = []
    # histories to save (the trainable params)
    histories = {}
    for (name,param) in filter(lambda (_,p): p.requires_grad, model.named_parameters() ):
        histories[name] = np.expand_dims(param.cpu().detach().numpy(), axis=0)


    for epoch in range(0, epochs):
        
        # reset loss and acc
        rolling_losses.reset()
        correct = 0
        
         
        # send model to device
        if change_device:
            model.to(device)
            optimizer.load_state_dict(optimizer.state_dict())
        
        # set model to trainable mode
        model.train()
        
        
        # load a batch
        for batch_idx, (data, target) in enumerate(train_loader):
            
#             # convert to same type as model (assumes all parameters have the same type)
#             data.type(model.parameters().next().type())
#             target.type(model.parameters().next().type())
            
            for n,p in model.named_parameters():
                if (p != p).any():
                    print("epoch {}, batch {}, layer {}".format(epoch,batch_idx,n)) 
                    raise TypeError("p 0 is nan \n {}".format(p.grad.data))
            
            # send batch data, targets to device
            data, target = data.to(device), target.to(device)
                                  
            # reset optimizer gradients
            optimizer.zero_grad()
                      
            # compute predictions
            output = model(data)
            if (output != output).any():
                print("epoch {}, batch {}".format(epoch,batch_ix)) 
                raise TypeError("output 0 is nan \n {}".format(output))
                
            # compute loss
            loss = loss_func(model=model, output=output, target=target)
            if (loss != loss).any():
                    print("epoch {}, batch {}, layer {}".format(epoch,batch_idx,n)) 
                    raise TypeError("loss 0 is nan \n {}".format(p.grad.data))

            # update rolling average loss
            rolling_losses.update(loss.item(), data.size()[0] )
            # update predictions
            if pred_func is not None:
                correct += pred_func(output=output, target=target)

            # propagate gradients and store them
            loss.backward()
            for n,p in model.named_parameters():
                if (p.grad is not None) and (p.grad.data != p.grad.data).any():
                    print("epoch {}, batch {}, layer {}".format(epoch,batch_idx,n)) 
                    raise TypeError("p.grad.data 1 is nan \n {}".format(p.grad.data))
               
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
            if pred_func is not None:
                pred_string = ', Accuracy: {}/{} ({:.0f}%)'.format(correct, len(train_loader.dataset),acc)
            else:
                pred_string = ''
            print('Epoch {} Train set - Average loss: {:.4f}'.format(epoch, rolling_losses.avg) + pred_string)
            
        # test data if applicable
        if test_loader is not None:
            if change_device:
                test_res = test(model, test_loader, loss_func, verbose=verbose, pred_func=pred_func, device=device)
            else:
                test_res = test(model, test_loader, loss_func, verbose=verbose, pred_func=pred_func)
            test_lost_hist.append(test_res['test_loss'])
            if pred_func is not None:
                test_acc_hist.append(test_res['test_accuracy'])

    # return desired dictionary
    ret = {'train_loss_hist':train_loss_hist}
    if pred_func is not None:
        ret['train_acc_hist'] = train_acc_hist
    if test_loader is not None:
        ret['test_loss_hist'] = test_lost_hist
        if pred_func is not None:
            ret['test_acc_hist'] = test_acc_hist
    if save_hist in [1,2]:
        ret['histories'] = histories
        
    return(ret)
               
               
               