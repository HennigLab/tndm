import matplotlib.pyplot as plt
import numpy as np

direction_colors = plt.cm.nipy_spectral(np.arange(8)/8)

def plot_behaviour(b_gt, b, di, n_show = None, ax=None, title=None):
    if ax is None:
        ax = plt.subplot(111)
    colors = plt.cm.nipy_spectral(np.arange(8)/8)
    if n_show is None:
        n_show = b.shape[0]
    for t in range(n_show):
        ax.plot(b_gt[t,:,0],b_gt[t,:,1],color=direction_colors[di[t]],alpha=.25)
        if b is not None:
            ax.plot(b[t,:,0],b[t,:,1],color=direction_colors[di[t]],alpha=1,lw=1.5)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xticks(())
    plt.yticks(())
    plt.axis('equal')
    
def plot_2factor(f, i1, i2, di, n_show = None, ax=None, labels=None, title=None):
    if labels is None:
        labels = ['Factor '+str(i+1) for i in (i1,i2)]
    if ax is None:
        ax = plt.subplot(111)
    colors = plt.cm.nipy_spectral(np.arange(8)/8)
    if n_show is None:
        n_show = f.shape[0]
    for t in range(n_show):
        plt.plot(f[t,:,i1],f[t,:,i2],color=direction_colors[np.array(di)[t]],alpha=.2, lw=1)
    for i in range(8):
        if np.sum(di==i)>0:
            plt.plot(np.mean(f.numpy()[di==i,:,i1],axis=0),
                     np.mean(f.numpy()[di==i,:,i2],axis=0),color=direction_colors[i],alpha=1, lw=2)
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.xticks(())
    plt.yticks(())
    plt.axis('equal') 
    
def plot_all_2factors(f, di, b=None, b_gt=None, n_show=10):
    n_factors = f.shape[-1]
    for f1 in range(n_factors):
        for f2 in range(f1+1,n_factors):
            ax = plt.subplot2grid((n_factors-1, n_factors-1),(f1,f2-1))
            plot_2factor(f, f1, f2, di, ax=ax, n_show=n_show)
    if (b is not None) and (b_gt is not None):
        ax = plt.subplot2grid((n_factors-1, n_factors-1),(n_factors-2, 0))
        plot_behaviour(b_gt, b, di, ax=ax)

def plot_1factor(f, i, di, n_show = 40, ax=None, labels=None, title=None):
    if labels is None:
        labels = 'Factor '+str(i+1)
    if ax is None:
        ax = plt.subplot(111)
    colors = plt.cm.nipy_spectral(np.arange(8)/8)
    n_show = f.shape[0]
    x = np.arange(f.shape[1])*10.
    for t in range(n_show):
        plt.plot(x, (f[t,:,i]),color=colors[di[t]],alpha=.2, lw=1)
    for d in range(8):
        if np.sum(di==d)>1:
            plt.plot(x, np.mean(f.numpy()[di==d,:,i],axis=0), color=colors[d],alpha=1, lw=2)
    plt.title(labels)
    plt.xlabel('Time (ms)')

def plot_all_1factors(f, di, b=None, b_gt=None):
    n_factors = f.shape[-1]
    for f1 in range(n_factors):
        ax = plt.subplot2grid((1, n_factors),(0,f1))
        plot_1factor(f, f1, di, ax=ax, n_show=np.min((80, f.shape[0])))
        
def plot_behaviour(b, b_gt, di, num=20):
    for i in range(np.min((num,b.shape[0]))):
        plt.plot(b_gt[i,:,0], b_gt[i,:,1], '--', alpha=.25, color=direction_colors[di[i]], lw=1.5)
        plt.plot(b[i,:,0],b[i,:,1], color=direction_colors[di[i]])
    plt.axis('equal')

def plot_single_behaviour(b, b_gt, di, trial=0):
    i=trial
    plt.plot(b_gt[i,:,0], b_gt[i,:,1], '--', alpha=.25, color=direction_colors[di[i]], lw=1.5)
    plt.plot(b[i,:,0],b[i,:,1], color=direction_colors[di[i]], lw=2)
    plt.axis('equal')

def plot_behaviour_weights(model):
    # IF TNDM: plot dense layer
    if 'TNDM' in type(model).__name__:
        print(model.behavioural_dense.name)
        if (model.behavioural_dense.name=='BehaviouralDense') or (model.behavioural_dense.name=='CausalBehaviouralDense'):
            in_size  = model.get_layer(name=model.behavioural_dense.name).mask_in_size
            out_size = model.get_layer(name=model.behavioural_dense.name).mask_out_size
            print(in_size,out_size)
            fig, axes = plt.subplots(out_size,in_size,figsize=(3*in_size,3*out_size))
            for i in range(in_size):
                for j in range(out_size):
                    axes[j,i].imshow(model.get_layer(name=model.behavioural_dense.name).kernel[i::in_size,j::out_size],vmin=-0.5, vmax=0.5, cmap=plt.cm.RdBu)
                    axes[j,i].set_ylabel(f'rel factor {i+1}')
                    l='X' if j==0 else 'Y'
                    axes[j,i].set_xlabel(f'beh {l}')
            fig.subplots_adjust(wspace=0.5)
            plt.show()
        else:
            plt.imshow(model.get_layer(name=model.behavioural_dense.name).kernel)
            print(model.get_layer(name=model.behavioural_dense.name).kernel.shape)
            in_size, out_size = model.get_layer(name=model.behavioural_dense.name).kernel.shape
            plt.show()
    else:
        in_size = z.shape[-1]
        out_size = data[f'test_{y_getter(name)}'].shape[-1]