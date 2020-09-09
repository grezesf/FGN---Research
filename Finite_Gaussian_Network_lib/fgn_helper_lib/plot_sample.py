import matplotlib.pyplot as plt
import numpy as np

def plot_sample(dataloader, title=None):
    
    # attempts to use plt.imshow to display a sample from the dataloader
    
    
    # dataloader values (might be huge and crash)
    ind = np.random.choice(range(dataloader.dataset.tensors[0].size()[0]))
    x_data = dataloader.dataset.tensors[0][ind]
    
    plt.imshow(x_data, cmap=plt.cm.get_cmap('Greys'))
    
    if title is not None:
        plt.title(title)
    plt.colorbar()
    plt.show()