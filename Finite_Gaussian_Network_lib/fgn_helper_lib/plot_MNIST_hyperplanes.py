import matplotlib.pyplot as plt
import numpy as np
import torch
from .categorical_cmap import categorical_cmap
from .get_rand_orthogonal_set import get_rand_orthogonal_set

def plot_MNIST_hyperplanes(model, start, end, rectangle_factor=1.0, n_plots=5, grid_size=100, mode='canonical', title=None):
    
    # given an MNIST model, a start and end point in the MNIST hyperspace (ie images)
    # plots a grid of heatmaps of the model along hyperplanes that contain (end-start) vector 
    
    # pre-requisites
    # colors = c10
    c10 =  categorical_cmap(10, 9, cmap='tab10', reverse=True, continuous=False)
    # rand_orthogonal_set
    
    # device
    # get model device 
    device = next(model.parameters()).device
    
    # set model in eval mode
    model.eval()    
    
    # plots a n_plot*nplot grid
    fig, axes = plt.subplots(nrows=n_plots, ncols=n_plots)   
    
    if title is not None: fig.suptitle(title)
    
    # vector-tensor from start to end
    v0 = end-start
    
    # set of orthogonal vectors to v0, 
    Vs = get_rand_orthogonal_set(v0, mode=mode)
    
    # check that not too many plots is asked
    assert len(Vs)-1>n_plots**2
    
    # if gridsize is even, add 1 to ensure we pass by (0,0)
    gs = grid_size
    if grid_size%2==0: gs+=1

    # define the XY grid rectangle (default=square)
    X = np.linspace(0, 1, gs)
    Y = np.linspace(-rectangle_factor*0.5, rectangle_factor*0.5, gs)
    Xs, Ys = np.meshgrid(X,Y)
    # cartestian product
    XYs = np.array(list(zip(Xs.ravel(), Ys.ravel())))
    
    
    for r in range(n_plots):
        for c in range(n_plots):
            
            # new inputs
            inputs_list = [start + x_step*v0 + y_step*Vs[1+r*n_plots+c] for x_step, y_step in XYs]
            inputs = torch.stack(inputs_list)
            # compute Zs
            with torch.no_grad(): heatmap_preds = model(inputs.to(device))
            
            # apply softmax for probs, nans are replaced by 1.0 
            heatmap_preds_softmax = np.nan_to_num(np.array([np.exp(x)/sum(np.exp(x)) 
                                                            for x in heatmap_preds.cpu().numpy()]), nan=1.0)
            
            pred_classes = np.argmax(heatmap_preds_softmax, axis=1)
            pred_confidences = np.max(heatmap_preds_softmax, axis=1)
            
            to_plot = [pred_class+0.9*pred_confidence 
                       for pred_class, pred_confidence in zip(pred_classes, pred_confidences)]

            to_plot = np.reshape(to_plot, np.shape(Xs))

            # plot using pcolormesh
            if n_plots>1:
                pcm = axes[r,c].pcolormesh(Xs, Ys,
                                     to_plot, 
                                     cmap=c10, vmin=0, vmax=10, shading='auto')
                axes[r,c].xaxis.set_visible(False)
                axes[r,c].yaxis.set_visible(False)
            else:
                # only one subplot
                pcm = axes.pcolormesh(Xs, Ys,
                                     to_plot, 
                                     cmap=c10, vmin=0, vmax=10, shading='auto')
                axes.xaxis.set_visible(False)
                axes.yaxis.set_visible(False)
        
    if n_plots>1:
        cbar = fig.colorbar(pcm, ax=axes.ravel().tolist(), ticks=[x+0.5 for x in range(10)])
    else:
        cbar = fig.colorbar(pcm, ax=axes, ticks=[x+0.5 for x in range(10)])

    cbar.ax.set_yticklabels(range(10))
    
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('MNIST Class * Confidence', rotation=270)
    
    # end 
    plt.show()
    