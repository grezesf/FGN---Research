import matplotlib.pyplot as plt
import numpy as np

def plot_centers_histories(histories, num_rand_per_layer=2):
    
    # given a histories{} dict as given by fgnh.train()
    # plot centers history for only some hidden layer neuron
    # num_rand_per_layer is the number of neurons per layer to plot (chosen randomly) (must be larger than num neurons in layer)

    for k in histories.keys():
        if 'centers' in k:
            # choose 2 random neurons in the layer to print
            neurons_to_plot =  np.random.choice(range(np.shape(histories[k])[1]), num_rand_per_layer, replace=False)
            
            first_plot=True
            for idx,n in enumerate(neurons_to_plot):
                x = histories[k][:,n,:]
                plt.subplot(2,1,idx+1)
                plt.plot(x, marker='.', linestyle='-')
                
                if first_plot:
                    plt.title(k)
                    first_plot=False
                    
                plt.grid()                
            plt.show()