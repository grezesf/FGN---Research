
import matplotlib.pyplot as plt
import numpy as np

import torch

def plot_pred_samples_grid(model, dataloader, inds=range(100), title=None, min_confidence=0.2, max_confidence=0.9):
    
    # given model and data, plots a 10*10 grid showing samples and where the max of predictions was <=0.2 and >=0.5

    # get model device 
    device = next(model.parameters()).device
    
    # dataloader values (might be huge and crash)
    if inds=='random':
        inds = np.random.choice(range(dataloader.dataset.tensors[0].size()[0]), 100)
    else:
        x_data = dataloader.dataset.tensors[0][inds]
    
    # compute predictions
    model.eval()
    with torch.no_grad(): preds = model(x_data.to(device))
    preds = preds.cpu().detach().numpy()
    
    # apply softmax for probabilities
    preds_softmax = np.array([np.exp(x)/sum(np.exp(x)) for x in preds.astype('float128')])
    
    # plot
    fig, axes = plt.subplots(nrows=10, ncols=10)
    
    if title is not None: 
        fig.suptitle(title+ '\nBlue<={}, Red>={} max prediction'.format(min_confidence, max_confidence))
    else:
        fig.suptitle('Blue<={}, Red>={} max prediction'.format(min_confidence, max_confidence))

    for r in range(10):
        for c in range(10):

            x = x_data[10*r+c].numpy().reshape((28,28))

            # is the max bigger than rest combined?
            if np.max(preds_softmax[10*r+c])>=max_confidence:
                axes[r,c].spines['bottom'].set_color('tab:red')
                axes[r,c].spines['top'].set_color('tab:red')
                axes[r,c].spines['left'].set_color('tab:red')
                axes[r,c].spines['right'].set_color('tab:red')

            # is the max close to the min?
            if np.max(preds_softmax[10*r+c])<=min_confidence:            
                axes[r,c].spines['bottom'].set_color('tab:blue')
                axes[r,c].spines['top'].set_color('tab:blue')
                axes[r,c].spines['left'].set_color('tab:blue')
                axes[r,c].spines['right'].set_color('tab:blue')

            axes[r,c].spines['bottom'].set_linewidth(3)
            axes[r,c].spines['top'].set_linewidth(3)
            axes[r,c].spines['left'].set_linewidth(3)
            axes[r,c].spines['right'].set_linewidth(3)
            axes[r,c].tick_params(axis='both', which='both',
                                  labelbottom='off', labelleft='off', bottom='off', left='off')
            axes[r,c].xaxis.set_visible(False)
            axes[r,c].yaxis.set_visible(False)


            axes[r,c].imshow(x, cmap=plt.cm.get_cmap('Greys'))
    #         axes[r,c].axis('off')
    plt.subplots_adjust(hspace=0., left=0.15, bottom=0.01, right=0.8, top=0.9, wspace=0.25)
    plt.show()