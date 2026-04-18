import numpy as np


def lhd(
    dist=None,
    size=None,
    dims=1,
    form="randomized",
    iterations=100,
    showcorrelations=False,
):
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
    """
    assert dims > 0, 'kwarg "dims" must be at least 1'
    if not size or not dist:
        return None

    def _lhs(x, samples=20):
        """
        _lhs(x) returns a latin-hypercube matrix (each row is a different
        set of sample inputs) using a default sample size of 20 for each column
        of X. X must be a 2xN matrix that contains the lower and upper bounds of
        each column. The lower bound(s) should be in the first row and the upper
        bound(s) should be in the second row.

        _lhs(x,samples=N) uses the sample size of N instead of the default (20).

        Example:
            >>> x = np.array([[0, -1, 3], [1, 2, 6]])
            >>> print 'x:'; print x
            x:
            [[ 0 -1  3]
             [ 1  2  6]]

            >>> print 'lhs(x):'; print _lhs(x)
            lhs(x):
            [[ 0.02989122 -0.93918734  3.14432618]
             [ 0.08869833 -0.82140706  3.19875152]
             [ 0.10627442 -0.66999234  3.33814979]
             [ 0.15202861 -0.44157763  3.57036894]
             [ 0.2067089  -0.34845384  3.66930908]
             [ 0.26542056 -0.23706445  3.76361414]
             [ 0.34201421 -0.00779306  3.90818257]
             [ 0.37891646  0.15458423  4.15031708]
             [ 0.43501575  0.23561118  4.20320064]
             [ 0.4865449   0.36350601  4.45792314]
             [ 0.54804367  0.56069855  4.60911539]
             [ 0.59400712  0.7468415   4.69923486]
             [ 0.63708876  0.9159176   4.83611204]
             [ 0.68819855  0.98596354  4.97659182]
             [ 0.7368695   1.18923511  5.11135111]
             [ 0.78885724  1.28369441  5.2900157 ]
             [ 0.80966513  1.47415703  5.4081971 ]
             [ 0.86196731  1.57844205  5.61067689]
             [ 0.94784517  1.71823504  5.78021164]
             [ 0.96739728  1.94169017  5.88604772]]

            >>> print 'lhs(x,samples=5):'; print _lhs(x,samples=5)
            lhs(x,samples=5):
            [[ 0.1949127  -0.54124725  3.49238369]
             [ 0.21128576 -0.13439798  3.65652016]
             [ 0.47516308  0.39957406  4.5797308 ]
             [ 0.64400392  0.90890999  4.92379431]
             [ 0.96279472  1.79415307  5.52028238]]
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

    def _mix(data, dim="cols"):
        """
        Takes a data matrix and mixes up the values along dim (either "rows" or
        "cols"). In other words, if dim='rows', then each row's data is mixed
        ONLY WITHIN ITSELF. Likewise, if dim='cols', then each column's data is
        mixed ONLY WITHIN ITSELF.
        """
        data = np.atleast_2d(data)
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

    if form == "randomized":
        if hasattr(dist, "__getitem__"):  # if multiple distributions were input
            nvars = len(dist)
            x = np.vstack((np.zeros(nvars), np.ones(nvars)))
            unif_data = _lhs(x, samples=size)
            dist_data = np.empty_like(unif_data)
            for i, d in enumerate(dist):
                dist_data[:, i] = d.ppf(unif_data[:, i])

        else:  # if a single distribution was input
            nvars = dims
            x = np.vstack((np.zeros(nvars), np.ones(nvars)))
            unif_data = _lhs(x, samples=size)
            dist_data = np.empty_like(unif_data)
            for i in range(nvars):
                dist_data[:, i] = dist.ppf(unif_data[:, i])

    elif form == "spacefilling":

        def euclid_distance(arr):
            n = arr.shape[0]
            ans = 0.0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    d = np.sqrt(
                        np.sum([
                            (arr[i, k] - arr[j, k]) ** 2
                            for k in range(arr.shape[1])
                        ])
                    )
                    ans += 1.0 / d**2
            return ans

        def fill_space(data):
            best = 1e8
            for _it in range(iterations):
                d = euclid_distance(data)
                if d < best:
                    d_opt = d
                    data_opt = data.copy()

                data = _mix(data)

            print("Optimized Distance:", d_opt)
            return data_opt

        if hasattr(dist, "__getitem__"):  # if multiple distributions were input
            nvars = len(dist)
            x = np.vstack((np.zeros(nvars), np.ones(nvars)))
            unif_data = fill_space(_lhs(x, samples=size))
            dist_data = np.empty_like(unif_data)
            for i, d in enumerate(dist):
                dist_data[:, i] = d.ppf(unif_data[:, i])

        else:  # if a single distribution was input
            nvars = dims
            x = np.vstack((np.zeros(nvars), np.ones(nvars)))
            unif_data = fill_space(_lhs(x, samples=size))
            dist_data = np.empty_like(unif_data)
            for i in range(nvars):
                dist_data[:, i] = dist.ppf(unif_data[:, i])

    elif form == "orthogonal":
        raise NotImplementedError(
            "Sorry. The orthogonal space-filling algorithm hasn't been implemented yet."
        )
    else:
        raise ValueError(f'Invalid "form" value: {form}')

    if dist_data.shape[1] > 1:
        cor_matrix = np.zeros((nvars, nvars))
        for i in range(nvars):
            for j in range(nvars):
                x_data = dist_data[:, i].copy()
                y_data = dist_data[:, j].copy()
                x_mean = x_data.mean()
                y_mean = y_data.mean()
                num = np.sum((x_data - x_mean) * (y_data - y_mean))
                den = np.sqrt(
                    np.sum((x_data - x_mean) ** 2)
                    * np.sum((y_data - y_mean) ** 2)
                )
                cor_matrix[i, j] = num / den
                cor_matrix[j, i] = num / den
        inv_cor_matrix = np.linalg.pinv(cor_matrix)
        VIF = np.max(np.diag(inv_cor_matrix))

        if showcorrelations:
            print("Correlation Matrix:\n", cor_matrix)
            print("Inverted Correlation Matrix:\n", inv_cor_matrix)
            print("Variance Inflation Factor (VIF):", VIF)

    return dist_data
