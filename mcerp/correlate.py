import itertools

import numpy as np
from scipy.stats import rankdata
from scipy.stats.distributions import norm

from . import UncertainFunction


def correlate(params, corrmat) -> None:
    """
    Force a correlation matrix on a set of statistically distributed objects.
    This function works on objects in-place.

    Parameters
    ----------
    params : array
        An array of of uv objects.
    corrmat : 2d-array
        The correlation matrix to be imposed

    """
    # Make sure all inputs are compatible
    assert all([isinstance(param, UncertainFunction) for param in params]), (
        'All inputs to "correlate" must be of type "UncertainFunction"'
    )

    # Put each ufunc's samples in a column-wise matrix
    data = np.vstack([param._mcpts for param in params]).T

    # Apply the correlation matrix to the sampled data
    new_data = induce_correlations(data, corrmat)

    # Re-set the samples to the respective variables
    for i in range(len(params)):
        params[i]._mcpts = new_data[:, i]


def induce_correlations(data, corrmat):
    """
    Induce a set of correlations on a column-wise dataset

    Parameters
    ----------
    data : 2d-array
        An m-by-n array where m is the number of samples and n is the
        number of independent variables, each column of the array corresponding
        to each variable
    corrmat : 2d-array
        An n-by-n array that defines the desired correlation coefficients
        (between -1 and 1). Note: the matrix must be symmetric and
        positive-definite in order to induce.

    Returns
    -------
    new_data : 2d-array
        An m-by-n array that has the desired correlations.

    """
    # Create an rank-matrix
    data_rank = np.vstack([rankdata(datai) for datai in data.T]).T

    # Generate van der Waerden scores
    data_rank_score = data_rank / (data_rank.shape[0] + 1.0)
    data_rank_score = norm(0, 1).ppf(data_rank_score)

    # Calculate the lower triangular matrix of the Cholesky decomposition
    # of the desired correlation matrix
    p = chol(corrmat)

    # Calculate the current correlations
    t = np.corrcoef(data_rank_score, rowvar=0)

    # Calculate the lower triangular matrix of the Cholesky decomposition
    # of the current correlation matrix
    q = chol(t)

    # Calculate the re-correlation matrix
    s = np.dot(p, np.linalg.inv(q))

    # Calculate the re-sampled matrix
    new_data = np.dot(data_rank_score, s.T)

    # Create the new rank matrix
    new_data_rank = np.vstack([rankdata(datai) for datai in new_data.T]).T

    # Sort the original data according to new_data_rank
    for i in range(data.shape[1]):
        _vals, order = np.unique(
            np.hstack((data_rank[:, i], new_data_rank[:, i])),
            return_inverse=True,
        )
        old_order = order[: new_data_rank.shape[0]]
        new_order = order[-new_data_rank.shape[0] :]
        tmp = data[np.argsort(old_order), i][new_order]
        data[:, i] = tmp[:]

    return data


def plotcorr(X, plotargs=None, full=True, labels=None):
    """
    Plots a scatterplot matrix of subplots.

    Usage:

        plotcorr(X)

        plotcorr(..., plotargs=...)  # e.g., 'r*', 'bo', etc.

        plotcorr(..., full=...)  # e.g., True or False

        plotcorr(..., labels=...)  # e.g., ['label1', 'label2', ...]

    Each column of "X" is plotted against other columns, resulting in
    a ncols by ncols grid of subplots with the diagonal subplots labeled
    with "labels".  "X" is an array of arrays (i.e., a 2d matrix), a 1d array
    of MCERP.UncertainFunction/Variable objects, or a mixture of the two.
    Additional keyword arguments are passed on to matplotlib's "plot" command.
    Returns the matplotlib figure object containing the subplot grid.
    """
    import matplotlib.pyplot as plt

    X = [Xi._mcpts if isinstance(Xi, UncertainFunction) else Xi for Xi in X]
    X = np.atleast_2d(X)
    numvars, _numdata = X.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.0, wspace=0.0)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if full:
            if ax.get_subplotspec().is_first_col():
                ax.yaxis.set_ticks_position("left")
            if ax.get_subplotspec().is_last_col():
                ax.yaxis.set_ticks_position("right")
            if ax.get_subplotspec().is_first_row():
                ax.xaxis.set_ticks_position("top")
            if ax.get_subplotspec().is_last_row():
                ax.xaxis.set_ticks_position("bottom")
        else:
            if ax.get_subplotspec().is_first_row():
                ax.xaxis.set_ticks_position("top")
            if ax.get_subplotspec().is_last_col():
                ax.yaxis.set_ticks_position("right")

    # Label the diagonal subplots...
    if not labels:
        labels = ["x" + str(i) for i in range(numvars)]

    for i, label in enumerate(labels):
        axes[i, i].annotate(
            label,
            (0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
        )

    # Plot the data
    for i, j in zip(*np.triu_indices_from(axes, k=1), strict=False):
        if full:
            idx = [(i, j), (j, i)]
        else:
            idx = [(i, j)]
        for x, y in idx:
            # FIX #1: this needed to be changed from ...(data[x], data[y],...)
            if plotargs is None:
                if len(X[x]) > 100:
                    plotargs = ",b"  # pixel marker
                else:
                    plotargs = ".b"  # point marker
            axes[x, y].plot(X[y], X[x], plotargs)
            ylim = min(X[y]), max(X[y])
            xlim = min(X[x]), max(X[x])
            axes[x, y].set_ylim(
                xlim[0] - (xlim[1] - xlim[0]) * 0.1,
                xlim[1] + (xlim[1] - xlim[0]) * 0.1,
            )
            axes[x, y].set_xlim(
                ylim[0] - (ylim[1] - ylim[0]) * 0.1,
                ylim[1] + (ylim[1] - ylim[0]) * 0.1,
            )

    # Turn on the proper x or y axes ticks.
    if full:
        for i, j in zip(list(range(numvars)), itertools.cycle((-1, 0))):
            axes[j, i].xaxis.set_visible(True)
            axes[i, j].yaxis.set_visible(True)
    else:
        for i in range(numvars - 1):
            axes[0, i + 1].xaxis.set_visible(True)
            axes[i, -1].yaxis.set_visible(True)
        for i in range(1, numvars):
            for j in range(i):
                fig.delaxes(axes[i, j])

    # FIX #2: if numvars is odd, the bottom right corner plot doesn't have the
    # correct axes limits, so we pull them from other axes
    if numvars % 2:
        xlimits = axes[0, -1].get_xlim()
        ylimits = axes[-1, 0].get_ylim()
        axes[-1, -1].set_xlim(xlimits)
        axes[-1, -1].set_ylim(ylimits)

    return fig


def chol(A):
    """
    Calculate the lower triangular matrix of the Cholesky decomposition of
    a symmetric, positive-definite matrix.
    """
    A = np.array(A)
    assert A.shape[0] == A.shape[1], "Input matrix must be square"

    L = [[0.0] * len(A) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = (
                (A[i][i] - s) ** 0.5
                if (i == j)
                else (1.0 / L[j][j] * (A[i][j] - s))
            )

    return np.array(L)
