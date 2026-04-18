from __future__ import annotations

import numpy as np


def lhd(
    dist: object = None,
    size: int | None = None,
    dims: int = 1,
    form: str = "randomized",
    **options: object,
) -> np.ndarray | None:
    """
    Create a Latin-Hypercube sample design based on distributions defined in the
    `scipy.stats` module

    Parameters
    ----------
    dist: array_like
        frozen scipy.stats.rv_continuous or rv_discrete distribution objects
        that are defined previous to calling LHD

    size: int
        integer value for the number of samples to generate for each
        distribution object

    dims: int, optional
        if dist is a single distribution object, and dims > 1, the one
        distribution will be used to generate a size-by-dims sampled design

    form: str, optional (non-functional at the moment)
        determines how the sampling is to occur, with the following optional
        values:
            - 'randomized' - completely randomized sampling
            - 'spacefilling' - space-filling sampling (generally gives a more
              accurate sampling of the design when the number of sample points
              is small)
            - 'orthogonal' - balanced space-filling sampling (experimental)

        The 'spacefilling' and 'orthogonal' forms require some iterations to
        determine the optimal sampling pattern.

    iterations: int, optional (non-functional at the moment)
        used to control the number of allowable search iterations for generating
        'spacefilling' and 'orthogonal' designs

    Returns
    -------
    out: 2d-array,
        A 2d-array where each column corresponds to each input distribution and
        each row is a sample in the design

    Examples
    --------

    Single distribution:
        - uniform distribution, low = -1, width = 2

    >>> import scipy.stats as ss
    >>> d0 = ss.uniform(loc=-1, scale=2)
    >>> print lhd(dist=d0,size=5)
    [[ 0.51031081]
     [-0.28961427]
     [-0.68342107]
     [ 0.69784371]
     [ 0.12248842]]

    Single distribution for multiple variables:
        - normal distribution, mean = 0, stdev = 1

    >>> d1 = ss.norm(loc=0, scale=1)
    >>> print lhd(dist=d1,size=7,dims=5)
    [[-0.8612785   0.23034412  0.21808001]
     [ 0.0455778   0.07001606  0.31586419]
     [-0.978553    0.30394663  0.78483995]
     [-0.26415983  0.15235896  0.51462024]
     [ 0.80805686  0.38891031  0.02076505]
     [ 1.63028931  0.52104917  1.48016008]]

    Multiple distributions:
        - beta distribution, alpha = 2, beta = 5
        - exponential distribution, lambda = 1.5

    >>> d2 = ss.beta(2, 5)
    >>> d3 = ss.expon(scale=1 / 1.5)
    >>> print lhd(dist=(d1,d2,d3),size=6)
    [[-0.8612785   0.23034412  0.21808001]
     [ 0.0455778   0.07001606  0.31586419]
     [-0.978553    0.30394663  0.78483995]
     [-0.26415983  0.15235896  0.51462024]
     [ 0.80805686  0.38891031  0.02076505]
     [ 1.63028931  0.52104917  1.48016008]]


    Raises
    ------
    ValueError
        If ``dims`` or ``form`` is invalid.
    NotImplementedError
        If the ``orthogonal`` form is requested.
    TypeError
        If an unknown option is supplied.
    """
    iterations = int(options.pop("iterations", 100))
    showcorrelations = bool(options.pop("showcorrelations", False))
    if options:
        unknown = ", ".join(sorted(options))
        raise TypeError(f"Unexpected lhd option(s): {unknown}")
    if dims <= 0:
        raise ValueError('kwarg "dims" must be at least 1')
    if size is None or not dist:
        return None
    if form == "orthogonal":
        raise NotImplementedError(
            "Sorry. The orthogonal space-filling algorithm hasn't been "
            "implemented yet."
        )
    if form not in {"randomized", "spacefilling"}:
        raise ValueError(f'Invalid "form" value: {form}')

    distributions = _as_distribution_list(dist, dims)
    x = np.vstack((np.zeros(len(distributions)), np.ones(len(distributions))))
    unif_data = _lhs(x, samples=size)
    if form == "spacefilling":
        unif_data = _fill_space(unif_data, iterations)

    dist_data = np.empty_like(unif_data)
    for i, d in enumerate(distributions):
        dist_data[:, i] = d.ppf(unif_data[:, i])

    if showcorrelations and dist_data.shape[1] > 1:
        _show_correlation_diagnostics(dist_data)

    return dist_data


def _as_distribution_list(dist: object, dims: int) -> list[object]:
    if hasattr(dist, "__getitem__") and not hasattr(dist, "ppf"):
        return list(dist)
    return [dist] * dims


def _lhs(x: np.ndarray, samples: int = 20) -> np.ndarray:
    """
    _lhs(x) returns a latin-hypercube matrix (each row is a different
    set of sample inputs) using a default sample size of 20 for each column
    of X. X must be a 2xN matrix that contains the lower and upper bounds of
    each column. The lower bound(s) should be in the first row and the upper
    bound(s) should be in the second row.

    _lhs(x,samples=N) uses the sample size of N instead of the default (20).

    Returns
    -------
    numpy.ndarray
        A latin-hypercube matrix.
    """

    # determine the segment size
    segmentSize = 1.0 / samples

    # get the number of dimensions to sample (number of columns)
    numVars = x.shape[1]

    # populate each dimension
    out = np.zeros((samples, numVars))
    pointValue = np.zeros(samples)

    for n in range(numVars):
        for i in range(samples):
            segmentMin = i * segmentSize
            point = segmentMin + (np.random.random() * segmentSize)
            pointValue[i] = (point * (x[1, n] - x[0, n])) + x[0, n]
        out[:, n] = pointValue

    # now randomly arrange the different segments
    return _mix(out)


def _mix(data: np.ndarray, dim: str = "cols") -> np.ndarray:
    """
    Takes a data matrix and mixes up the values along dim (either "rows" or
    "cols"). In other words, if dim='rows', then each row's data is mixed
    ONLY WITHIN ITSELF. Likewise, if dim='cols', then each column's data is
    mixed ONLY WITHIN ITSELF.

    Returns
    -------
    numpy.ndarray
        The mixed data matrix.
    """
    data = np.atleast_2d(data).copy()
    n = data.shape[0]

    if dim == "rows":
        data = data.T

    data_rank = list(range(n))
    for i in range(data.shape[1]):
        new_data_rank = np.random.permutation(data_rank)
        _vals, order = np.unique(
            np.hstack((data_rank, new_data_rank)), return_inverse=True
        )
        old_order = order[:n]
        new_order = order[-n:]
        tmp = data[np.argsort(old_order), i][new_order]
        data[:, i] = tmp[:]

    if dim == "rows":
        data = data.T

    return data


def _euclid_distance(arr: np.ndarray) -> float:
    n = arr.shape[0]
    ans = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((arr[i] - arr[j]) ** 2))
            ans += 1.0 / d**2
    return ans


def _fill_space(data: np.ndarray, iterations: int) -> np.ndarray:
    best = 1e8
    data_opt = data.copy()
    d_opt = best
    for _it in range(iterations):
        d = _euclid_distance(data)
        if d < best:
            d_opt = d
            data_opt = data.copy()
            best = d

        data = _mix(data)

    print("Optimized Distance:", d_opt)
    return data_opt


def _show_correlation_diagnostics(dist_data: np.ndarray) -> None:
    cor_matrix = np.corrcoef(dist_data, rowvar=False)
    inv_cor_matrix = np.linalg.pinv(cor_matrix)
    VIF = np.max(np.diag(inv_cor_matrix))

    print("Correlation Matrix:\n", cor_matrix)
    print("Inverted Correlation Matrix:\n", inv_cor_matrix)
    print("Variance Inflation Factor (VIF):", VIF)
