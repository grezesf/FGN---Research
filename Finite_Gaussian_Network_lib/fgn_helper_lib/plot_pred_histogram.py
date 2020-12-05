from scipy import stats

import matplotlib.pyplot as plt
import numpy as np

import torch

def plot_pred_histogram(model, dataloader, title=None, verbose=False):
    
    # given a model and a dataload, plots the histogram of the strongest predictions over the data
    
    # get model device 
    device = next(model.parameters()).device
    
    # dataloader values (might be huge and crash)
    x_data = dataloader.dataset.tensors[0]
    
    # compute predictions
    model.eval()
    with torch.no_grad(): preds = model(x_data.to(device))
    preds = preds.cpu().detach().numpy()
    
    # apply softmax for probabilities
    preds_softmax = np.array([np.exp(x)/sum(np.exp(x)) for x in preds.astype('float128')])
    # get the maximums
    preds_maxes = np.max(preds_softmax, axis=1)

    # histogram
    weights = np.ones_like(preds_maxes)/len(preds_maxes)
    plt.hist(preds_maxes, bins=(np.arange(51)+1)/51.0, rwidth=0.9, align='mid', weights=weights)
    plt.xticks((np.arange(10)+1)/10.0)
    plt.grid(True)
    if title is not None:
        plt.title(title)
#     plt.gca().set_aspect("equal")
    plt.show()
    
    # extra: quick description of the maxes
    if verbose: 
        print('Statistical description of the predictions maximums')
        print(stats.describe(preds_maxes))
        print()
        print('percentage of confident predictions (>=0.5):',float(len([x for x in preds_maxes if x>=0.5])/float(len(preds_maxes))))
        print()
        print('Sample predictions')
        inds = np.random.choice(list(range(preds_softmax.shape[0])), 5)
        for i, p in zip(inds, preds_softmax[inds]):
            print('index {} prediction {}'.format(i, p.round(2)))
        
        