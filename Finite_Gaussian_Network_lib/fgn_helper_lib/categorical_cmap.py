import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

def categorical_cmap(nc, nsc, cmap="tab10", continuous=False, reverse=False):
    
    # The following would be a function categorical_cmap, 
    # which takes as input the number of categories (nc) and the number of subcategories (nsc) 
    # and returns a colormap with nc*nsc different colors, 
    # where for each category there are nsc colors of same hue.
    # 'reverse' controls if the most intense color is at the start (default) or the end
    # source: https://stackoverflow.com/questions/47222585/matplotlib-generic-colormap-from-tab10
    # use: c10 = categorical_cmap(10, 9, cmap='tab10', reverse=True, continuous=False)
    
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        arhsv[:,1] = np.linspace(chsv[1],0, nsc)
        arhsv[:,2] = np.linspace(chsv[2],1, nsc)
        if reverse:
            rgb = matplotlib.colors.hsv_to_rgb(arhsv[::-1])
        else:
            rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb       
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap
