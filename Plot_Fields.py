import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
from pylab import figure, cm
from matplotlib.colors import LogNorm
import tikzplotlib


F_non = np.load("F_non.npy")
F_lin = np.load("F_lin.npy")
H_non = np.load("H_non.npy")
H_lin = np.load("H_lin.npy")
E_non = np.load("E_non.npy")
E_lin = np.load("E_lin.npy")

F_gross = np.load("F_14_14_5_10_000.npy")

#steps = len(F_non[0,0,0,:])




def log_transform(im):
    '''returns log(image) scaled to the interval [0,1]'''
    try:
        (min, max) = (im[im > 0].min(), im.max())
        if (max > min) and (max > 0):
            return (np.log(im.clip(min, max)) - np.log(min)) / (np.log(max) - np.log(min))
    except:
        pass
    return im


def get_ticks(x,min_x,max_x):
    a = (np.e * min_x - max_x) / (np.e - 1)
    b = 1 / (min_x - a)

    return round(((((np.e)**x)/b+a)*100),1)

def plotten(X):
    steps = len(X[0,0,0,:])
    cmap = plt.cm.get_cmap("Blues").copy()
    cmap.set_bad(color='red')



    fig = plt.figure(figsize=(20, 20))

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(2, 4),
                    axes_pad=0.25,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.5
                    )
    t = 0
    for ax in grid:
        ax.set_axis_off()
        im = ax.imshow(log_transform(X[:,2,:,t].T), cmap=cmap)#, norm=LogNorm(vmin=0, vmax=1.1))

        t += int(steps/8)

    # when cbar_mode is 'single', for ax in grid, ax.cax = grid.cbar_axes[0]
    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)

    ticklabels_list = []

    for i in range(4):
        ticklabels_list.append(str(get_ticks(i/3,np.min(X),np.max(X)))+"%")

    cbar.ax.set_yticks(np.arange(0, 4/3, 1/3))
    cbar.ax.set_yticklabels(ticklabels_list)
    #tikzplotlib.save("testdatei.tex")
    plt.show()


plotten(F_gross)
