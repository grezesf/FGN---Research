import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_2D_heatmap(model, title="Set Title", scale=1.0, plot_mode='first', show_data=False):
    
    # scale of the heat maps
    X1 = np.arange(-scale, scale*1.001, scale/100.0)
    X1s, X2s = np.meshgrid(X1,X1)
    heatmap_inputs = np.reshape(list(zip(X1s.flatten(),X2s.flatten())),(-1,2))
    heatmap_inputs = torch.Tensor(heatmap_inputs)
    
    # get model device 
    device = next(model.parameters()).device
    
    # compute predictions
    model.eval()
    with torch.no_grad(): heatmap_preds = model(heatmap_inputs.to(device))
    heatmap_preds = heatmap_preds.cpu().detach().numpy()
    # apply softmax for probs, nans are replaced by 1.0 
    heatmap_preds_softmax = np.nan_to_num(np.array([np.exp(x)/sum(np.exp(x)) for x in heatmap_preds]), nan=1.0)

    # get levels 
    levels = np.arange(-0.0, 1.0+0.001, 10**(-2))
#     ticks = levels[::5]
#     levels=20
    ticks=np.arange(-1,1.1, 0.2)
    
    # number of classes in pred
    num_classes = len(heatmap_preds_softmax[0])
    
    # plot stacked or single
    if plot_mode=='first':
        # only plot the first class
        plt.contourf(X1s, X2s, np.reshape(heatmap_preds_softmax[:,0], np.shape(X1s) ),levels=levels, cmap= mpl.cm.RdYlBu_r)
    
    elif plot_mode=='stacked':
        # plot stacked
        for i in range(num_classes):
            plt.contourf(X1s, X2s, np.reshape(heatmap_preds_softmax[:,i], np.shape(X1s) ),levels=levels, cmap= mpl.cm.RdYlBu_r, alpha=1.0/num_classes)

    elif plot_mode=='full':
        # print all the classes
        print_title=True
        for i in range(num_classes):
            plt.contourf(X1s, X2s, np.reshape(heatmap_preds_softmax[:,i], np.shape(X1s) ),levels=levels, cmap= mpl.cm.RdYlBu_r)
            plt.colorbar(ticks=ticks)
            
            # if show_data is a dataloader
            if show_data!=False:
                x,y = list(zip(*show_data.dataset.tensors[0].detach().cpu().numpy()))
                plt.scatter(x, y, alpha=0.3, c='gray')
                
            # title only for first
            if print_title:
                print_title=False
                plt.title(title)
            plt.grid(True)
            plt.gca().set_aspect("equal")
            plt.show()
        return 
        
    plt.colorbar(ticks=ticks)
    
    # if show_data is a dataloader
    if show_data!=False and plot_mode!='full':
        x,y = list(zip(*show_data.dataset.tensors[0].detach().cpu().numpy()))
        plt.scatter(x, y, alpha=0.3, c='gray')

    plt.title(title)
    plt.grid(True)
    plt.gca().set_aspect("equal")
    plt.show()
    return heatmap_preds