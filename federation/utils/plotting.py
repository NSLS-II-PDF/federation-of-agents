import numpy as np


def waterfall(ax, x, ys, alphas=None, color="k", sampling=1, offset=0.2, **kwargs):
    """
    Waterfall plot on axis.

    Parameters
    ----------
    ax: axis
    x: array
        1-d array for shared x value
    ys: array
        2-d array of y values to sample
    alphas: array, None
        1-d array of alpha values for each sample
    color
        mpl color
    sampling: int
        Sample rate for full ys set
    offset: float
        Offset to place in waterfall
    kwargs

    Returns
    -------

    """
    if alphas is None:
        alphas = np.ones_like(ys[:, 0])
    indicies = range(0, ys.shape[0])[::sampling]
    for plt_i, idx in enumerate(indicies):
        y = ys[idx, :] + plt_i * offset
        ax.plot(x, y, color=color, alpha=alphas[idx], **kwargs)
