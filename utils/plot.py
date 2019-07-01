
#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import pandas
import utils.helper as hp


def plot_history(history, path_outfile):
    fig, ax = plt.subplots(1)
    ax.plot(history['epoch'], history['loss'], label='training')
    ax.plot(history['epoch'], history['val_loss'], label='validation')
    ax.legend()
    ax.set(xlabel='epoch', ylabel='loss')
    plt.savefig('%s/loss.png' % path_outfile)
    plt.close()


def plot_traces(x_noisy,x_label ,x_pred, path_outfile, title = None):

    n = 4  # how many digits we will display
    #plt.figure(figsize=(30, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, 4, i + 1)

        plt.plot(x_noisy[i])
        plt.gray()
        #plt.title("noisy_traces")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, 4, i + 1 + n)
        plt.plot(x_label[i])
        plt.gray()
        #plt.title("true_signal")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


        ax = plt.subplot(3, 4, i + 1 +2* n)
        plt.plot(x_pred[i])
        plt.gray()
        #plt.title("predicted_traces")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(f"/.png")
    plt.close()


def plot_energy(pred_energy, true_energy, path_outfile):
    diff = (pred_energy - true_energy)/(true_energy+pred_energy)
    std = np.std(diff)
    plt.subplot(111)
    plt.hist(diff, bins=50)
    plt.figtext(0.8,0.8, f"std:{int(std)}")
   # plt.legend(f"mean:{mean},std:{std}")
    #plt.xlim(-2 , 2)
    plt.title('(E_pred - E_true)/(E_true+ E_pred)')
    plt.savefig(f"{path_outfile}/energy.png")
    plt.close()

def plot_hist_stats(ax, data, bins,weights=None, posx=0.05, posy=0.95, overflow=None,
                    underflow=None, rel=False, integral=None,
                    additional_text="", additional_text_pre="",
                    fontsize=22, color="k", va="top", ha="left",
                    median=True, quantiles=True, mean=True, std=True, N=True):
    list = []
    for x in data:
        if x < max(bins) and x> min(bins):
            list.append(x)

    data = np.array(list)
    tmean = data.mean()
    tstd = data.std()
    if weights is not None:
        def weighted_avg_and_std(values, weights):
            """
            Return the weighted average and standard deviation.

            values, weights -- Numpy ndarrays with the same shape.
            """
            average = np.average(values, weights=weights)
            variance = np.average((values - average) ** 2, weights=weights)  # Fast and numerically precise
            return (average, variance ** 0.5)
        tmean, tstd = weighted_avg_and_std(data, weights)

    textstr = additional_text_pre
    if (textstr != ""):
        textstr += "\n"
    if N:
        textstr += "$N=%i$\n" % data.size
#     import SignificantFigures as serror
    if mean:
        if weights is None:
#             textstr += "$\mu = %s \pm %s$\n" % serror.formatError(tmean,
#                                                 tstd / math.sqrt(data.size))
            textstr += "$\mu = %.4g$\n" % tmean
        else:
            textstr += "$\mu = %.2g$\n" % tmean
    if median:
        tweights = np.ones_like(data)
        if weights is not None:
            tweights = weights
        if quantiles:
            q1 = hp.quantile_1d(data, tweights, 0.16)
            q2 = hp.quantile_1d(data, tweights, 0.84)
            median = hp.median(data, tweights)
#             median_str = serror.formatError(median, 0.05 * (np.abs(median - q2) + np.abs(median - q1)))[0]
            textstr += "$\mathrm{median} = %.2g^{+%.2g}_{-%.2g}$\n" % (median, np.abs(median - q2),
                                                                       np.abs(median - q1))
        else:
            textstr += "$\mathrm{median} = %.2g $\n" % hp.median(data, tweights)
    if std:
        if rel:
            textstr += "$\sigma = %.2g$ (%.1f\%%)\n" % (tstd, tstd / tmean * 100.)
        else:
            textstr += "$\sigma = %.2g$\n" % (tstd)

    add = " (ignored)" if integral is not None else ""
    if(overflow):
        textstr += "$\mathrm{overflows} = %i$%s\n" % (overflow, add)
    if(underflow):
        textstr += "$\mathrm{underflows} = %i$%s\n" % (underflow, add)

    textstr += additional_text

    props = dict(boxstyle='square', facecolor='w', alpha=0.5)
    ax.text(posx, posy, textstr, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment=va, ha=ha, multialignment='left',
            bbox=props, color=color)




def get_histogram(data, bins=20, xlabel="delta_E", ylabel="entries", weights=None, integral=None, ylim=None,
                  title="E_pred-E_true/E_true", stats=True, show=False, stat_kwargs={}, funcs=None, overflow=True, label=None,
                  ax=None, kwargs={'facecolor': '0.7', 'alpha': 1, 'edgecolor': "k"},
                  figsize=None):
    """ creates a histogram using matplotlib from array """
    if(ax is None):
        if figsize is None:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    else:
        ax1 = ax

    std = np.std(data)
    mean= np.mean(data)
    print(f"mean:{mean},std:{std}")
    min_bin = -2
    max_bin = 2
    binwidth= np.abs(max_bin-min_bin)/bins

    bining = np.arange(min_bin, max_bin+ binwidth, binwidth)
    print(bining)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    n, bins, patches = ax1.hist(data, bins= bining, weights=weights, label=label, **kwargs)

    if(funcs):
        for func in funcs:
            xlim = np.array(ax1.get_xlim())
            print(xlim)
            xx = np.linspace(xlim[0], xlim[1], 100)
            if('args' in func):
                ax1.plot(xx, func['func'](xx), *func['args'])
                if('kwargs' in func):
                    ax1.plot(xx, func['func'](xx), *func['args'], **func['kwargs'])
            else:
                ax1.plot(xx, func['func'](xx))

    ylim = ylim or n.max() * 1.2
    ax1.set_ylim(0, ylim)
    ax1.set_xlim(bins[0], bins[-1])

    if label is not None:
        ax1.legend()

    if stats:
        if overflow:
            underflow = np.sum(data < bins[0])
            overflow = np.sum(data > bins[-1])
        else:
            underflow = None
            overflow = None

        if integral is not None:
            if isinstance(integral, list):
                underflow = np.sum(data < integral[0])
                overflow = np.sum(data > integral[-1])
                data = data[data > integral[0]]
                data = data[data < integral[-1]]
            else:
                data = data[data > bins[0]]
                data = data[data < bins[-1]]
        print(f"underflow:{underflow}overflow:{overflow}")
        plot_hist_stats(ax1, data,bins=bining ,overflow=overflow, underflow=underflow, weights=weights, integral=integral, **stat_kwargs)

    if(show):
        plt.show()
    if(ax is None):
        return fig, ax1


def plot_energy_distribution(x_rec_energy, path_outfile, label, x_energy=None, y_energy=None, binning=20):
    fig = plt.figure(figsize=(16, 10))
    fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)





    fig, ax = get_histogram(
        x_rec_energy, bins=binning, label=label, xlabel=r"$E$", ylabel="Entries",
        stat_kwargs={"ha": "right", "fontsize": 10, "posx": 0.95},
        kwargs={'alpha': 1, 'facecolor': 'r'})

    plt.savefig(f"{path_outfile}/energy_distr{label}.png")
    plt.close(fig)
