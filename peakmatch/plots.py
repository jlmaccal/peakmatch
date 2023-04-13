import matplotlib.pyplot as plt
import torchmetrics
from matplotlib.patches import ConnectionPatch
import torch

def transform_data(data, drange, a = -1, b = 1):
    return (b - a) * ( (data - drange[0]) / (drange[1] - drange[0])  ) + a

def plot_hsqc(ax, hsqc, hrange, nrange, edges=None):
    n = transform_data(hsqc[:, 0], [-1, 1], a = nrange[0], b = nrange[1] )
    h = transform_data(hsqc[:, 1], [-1, 1], a = hrange[0], b = hrange[1] )
    ax.scatter(h, n)

    # if edges != None:
    #     hh = np.vstack([ h[ [edges[0]] ], h[ [edges[1]] ], ])
    #     nn = np.vstack([ n[ [edges[0]] ], n[ [edges[1]] ], ])
    #     ax.plot(hh, nn, '-', c='C1')
    return h, n

def gen_assignment_fig(pred, fake, x, y,):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6,10))
    hrange = (7.3, 9.3)
    nrange = (103.3, 129.7)
    
    calc_h, calc_n = plot_hsqc(axes[0], pred, hrange, nrange)
    fake_h, fake_n = plot_hsqc(axes[1], fake, hrange, nrange)
    
    pred_y = torch.argmax(x, dim=1)

    for idx, (pred, target) in enumerate(zip(pred_y, y)):    
        
       
        hn_calc = (calc_h[pred], calc_n[pred])
        hn_fake = (fake_h[idx], fake_n[idx])

        if target == pred:  
            color = 'green'
            
        else:
            color='red'
            

        con = ConnectionPatch(xyA=hn_calc, xyB=hn_fake, coordsA="data", coordsB="data",
                           axesA=axes[0], axesB=axes[1], color=color)
        axes[1].add_artist(con)
    
    for i in torch.arange(0, calc_h.shape[0]):
        indices = torch.argwhere(pred_y == i)

        if len(indices) > 1:
            axes[0].scatter(calc_h[i], calc_n[i], color='orange')

        if len(indices) == 0:
            axes[0].scatter(calc_h[i], calc_n[i], color='black') 

    axes[1].set_xlim(axes[0].get_xlim())
    axes[1].set_ylim(axes[0].get_ylim())
    
    for ax in axes:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    return fig