import matplotlib.pyplot as plt
import numpy as np

def plot_sample(dataloader, title=None, print_class=False):
    
    # attempts to use plt.imshow to display a sample from the dataloader
    
    
    # dataloader values (might be huge and crash)
    ind = np.random.choice(range(dataloader.dataset.tensors[0].size()[0]))
    x_data = dataloader.dataset.tensors[0][ind]
    
    if print_class: print('sample class: {}'.format(dataloader.dataset.tensors[1][ind]))

    
    plt.imshow(x_data, cmap=plt.cm.get_cmap('Greys'))
    
    if title is not None:
        plt.title(title)
    plt.colorbar()
    plt.show()
    