"""
================================================================================
mcerp: Real-time latin-hypercube-sampling-based Monte Carlo Error Propagation
================================================================================

Authors: Abraham Lee
         Saud Zahir

Copyright (c) 2013, Abraham D. Lee

"""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"


import numpy as np
import scipy.stats as ss

from .lhd import lhd


__author__ = "Abraham Lee"

npts = 10000

CONSTANT_TYPES = (float, int, int)


class NotUpcast(Exception):
    """Raised when an object cannot be converted to a number with uncertainty"""


def to_uncertain_func(x):
    """
    Transforms x into an UncertainFunction-compatible object,
    unless it is already an UncertainFunction (in which case x is returned
    unchanged).

    Raises an exception unless 'x' belongs to some specific classes of
    objects that are known not to depend on UncertainFunction objects
    (which then cannot be considered as constants).
    """
    if isinstance(x, UncertainFunction):
        return x

    # ! In Python 2.6+, numbers.Number could be used instead, here:
    elif isinstance(x, CONSTANT_TYPES):
        # No variable => no derivative to define:
        return UncertainFunction([x] * npts)

    raise NotUpcast(
        f"{type(x)} cannot be converted to a number with uncertainty"
    )


class UncertainFunction:
    """
    UncertainFunction objects represent the uncertainty of a result of
    calculations with uncertain variables. Nearly all basic mathematical
    operations are supported.

    This class is mostly intended for internal use.
    """

    def __init__(self, mcpts) -> None:
        self._mcpts = np.atleast_1d(mcpts).flatten()
        self.tag = None

    @property
    def mean(self):
        """
        Mean value as a result of an uncertainty calculation
        """
        mn = np.mean(self._mcpts)
        return mn

    @property
    def var(self):
        """
        Variance value as a result of an uncertainty calculation
        """
        mn = self.mean
        vr = np.mean((self._mcpts - mn) ** 2)
        return vr

    @property
    def std(self):
        r"""
        Standard deviation value as a result of an uncertainty calculation,
        defined as::

                    ________
            std = \/variance

        """
        return self.var**0.5

    @property
    def skew(self):
        r"""
        Skewness coefficient value as a result of an uncertainty calculation,
        defined as::

              _____     m3
            \/beta1 = ------
                      std**3

        where m3 is the third central moment and std is the standard deviation
        """
        mn = self.mean
        sd = self.std
        sk = (
            0.0 if abs(sd) <= 1e-8 else np.mean((self._mcpts - mn) ** 3) / sd**3
        )
        return sk

    @property
    def kurt(self):
        """
        Kurtosis coefficient value as a result of an uncertainty calculation,
        defined as::

                      m4
            beta2 = ------
                    std**4

        where m4 is the fourth central moment and std is the standard deviation
        """
        mn = self.mean
        sd = self.std
        kt = (
            0.0 if abs(sd) <= 1e-8 else np.mean((self._mcpts - mn) ** 4) / sd**4
        )
        return kt

    @property
    def stats(self):
        """
        The first four standard moments of a distribution: mean, variance, and
        standardized skewness and kurtosis coefficients.
        """
        mn = self.mean
        vr = self.var
        sk = self.skew
        kt = self.kurt
        return [mn, vr, sk, kt]

    def percentile(self, val):
        """
        Get the distribution value at a given percentile or set of percentiles.
        This follows the NIST method for calculating percentiles.

        Parameters
        ----------
        val : scalar or array
            Either a single value or an array of values between 0 and 1.

        Returns
        -------
        out : scalar or array
            The actual distribution value that appears at the requested
            percentile value or values

        """
        try:
            # test to see if an input is given as an array
            out = [self.percentile(vi) for vi in val]
        except (ValueError, TypeError):
            if val <= 0:
                out = float(min(self._mcpts))
            elif val >= 1:
                out = float(max(self._mcpts))
            else:
                tmp = np.sort(self._mcpts)
                n = val * (len(tmp) + 1)
                k, d = int(n), n - int(n)
                out = float(tmp[k] + d * (tmp[k + 1] - tmp[k]))
        if isinstance(val, np.ndarray):
            out = np.array(out)
        return out

    def _to_general_representation(self, str_func):
        mn, vr, sk, kt = self.stats
        return (
            f"uv({str_func(mn)}, {str_func(vr)}, {str_func(sk)}, {str_func(kt)})"
            if any([vr, sk, kt])
            else str_func(mn)
        )

    def __str__(self) -> str:
        return self._to_general_representation(str)

    def __repr__(self) -> str:
        #        return self._to_general_representation(repr)
        return str(self)

    def describe(self, name=None) -> None:
        """
        Cleanly show what the four displayed distribution moments are:
            - Mean
            - Variance
            - Standardized Skewness Coefficient
            - Standardized Kurtosis Coefficient

        For a standard Normal distribution, these are [0, 1, 0, 3].

        If the object has an associated tag, this is presented. If the optional
        ``name`` kwarg is utilized, this is presented as with the moments.
        Otherwise, no unique name is presented.

        Example
        =======
        ::

            >>> x = N(0, 1, 'x')
            >>> x.describe()  # print tag since assigned
            MCERP Uncertain Value (x):
            ...

            >>> x.describe('foobar')  # 'name' kwarg takes precedence
            MCERP Uncertain Value (foobar):
            ...

            >>> y = x**2
            >>> y.describe('y')  # print name since assigned
            MCERP Uncertain Value (y):
            ...

            >>> y.describe()  # print nothing since no tag
            MCERP Uncertain Value:
            ...

        """
        mn, vr, sk, kt = self.stats
        if name is not None:
            s = "MCERP Uncertain Value (" + name + "):\n"
        elif self.tag is not None:
            s = "MCERP Uncertain Value (" + self.tag + "):\n"
        else:
            s = "MCERP Uncertain Value:\n"
        s += f" > Mean................... {mn: }\n"
        s += f" > Variance............... {vr: }\n"
        s += f" > Skewness Coefficient... {sk: }\n"
        s += f" > Kurtosis Coefficient... {kt: }\n"
        print(s)

    def plot(self, hist=False, show=False, **kwargs) -> None:
        """
        Plot the distribution of the UncertainFunction. By default, the
        distribution is shown with a kernel density estimate (kde).

        Optional
        --------
        hist : bool
            If true, a density histogram is displayed (histtype='stepfilled')
        show : bool
            If ``True``, the figure will be displayed after plotting the
            distribution. If ``False``, an explicit call to ``plt.show()`` is
            required to display the figure.
        kwargs : any valid matplotlib.pyplot.plot or .hist kwarg

        """
        import matplotlib.pyplot as plt

        vals = self._mcpts
        low = min(vals)
        high = max(vals)

        p = ss.kde.gaussian_kde(vals)
        xp = np.linspace(low, high, 100)

        if hist:
            h = plt.hist(
                vals,
                bins=int(np.sqrt(len(vals)) + 0.5),
                histtype="stepfilled",
                density=True,
                **kwargs,
            )
            plt.ylim(0, 1.1 * h[0].max())
        else:
            plt.plot(xp, p.evaluate(xp), **kwargs)

        plt.xlim(low - (high - low) * 0.1, high + (high - low) * 0.1)

        if show:
            self.show()

    def show(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()

    def __add__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[0]._mcpts + uf[1]._mcpts
        return UncertainFunction(mcpts)

    def __radd__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[0]._mcpts + uf[1]._mcpts
        return UncertainFunction(mcpts)

    def __mul__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[0]._mcpts * uf[1]._mcpts
        return UncertainFunction(mcpts)

    def __rmul__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[0]._mcpts * uf[1]._mcpts
        return UncertainFunction(mcpts)

    def __sub__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[0]._mcpts - uf[1]._mcpts
        return UncertainFunction(mcpts)

    def __rsub__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[1]._mcpts - uf[0]._mcpts
        return UncertainFunction(mcpts)

    def __div__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[0]._mcpts / uf[1]._mcpts
        return UncertainFunction(mcpts)

    def __rdiv__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[1]._mcpts / uf[0]._mcpts
        return UncertainFunction(mcpts)

    def __truediv__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[0]._mcpts / uf[1]._mcpts
        return UncertainFunction(mcpts)

    def __rtruediv__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[1]._mcpts / uf[0]._mcpts
        return UncertainFunction(mcpts)

    def __pow__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[0]._mcpts ** uf[1]._mcpts
        return UncertainFunction(mcpts)

    def __rpow__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[1]._mcpts ** uf[0]._mcpts
        return UncertainFunction(mcpts)

    def __neg__(self):
        mcpts = -self._mcpts
        return UncertainFunction(mcpts)

    def __pos__(self):
        mcpts = self._mcpts
        return UncertainFunction(mcpts)

    def __abs__(self):
        mcpts = np.abs(self._mcpts)
        return UncertainFunction(mcpts)

    def __eq__(self, val):
        """
        If we are comparing two distributions, check the resulting moments. If
        they are the same distribution, then the moments will all be zero and
        we can know that it is actually the same distribution we are comparing
        ``self`` to, otherwise, at least one statistical moment will be non-
        zero.

        If we are comparing ``self`` to a scalar, just do a normal comparison
        so that if the underlying distribution looks like a PMF, a meaningful
        probability of self==val is returned. This can still work quite safely
        for PDF distributions since the likelihood of comparing self to an
        actual sampled value is negligible when mcerp.npts is large.

        Examples:

            >>> h = H(50, 5, 10)  # Hypergeometric distribution (PMF)
            >>> h == 4  # what's the probability of getting 4 of the 5?
            0.004
            >>> sum([
            ...     h == i for i in (0, 1, 2, 3, 4, 5)
            ... ])  # sum of all discrete probabilities
            1.0

            >>> n = N(0, 1)  # Normal distribution (PDF)
            >>> n == 0  # what's the probability of being exactly 0.0?
            0.0
            >>> n > 0  # greater than 0.0?
            0.5
            >>> n < 0  # less than 0.0?
            0.5
            >>> n == 1  # exactly 1.0?
            0.0
        """
        if isinstance(val, UncertainFunction):
            diff = self - val
            return not (diff.mean or diff.var or diff.skew or diff.kurt)
        else:
            return len(self._mcpts[self._mcpts == val]) / float(npts)

    def __ne__(self, val):
        if isinstance(val, UncertainFunction):
            return not self == val
        else:
            return 1 - (self == val)

    def __lt__(self, val):
        """
        If we are comparing two distributions, perform statistical tests,
        otherwise, calculate the probability that the distribution is
        less than val
        """
        if isinstance(val, UncertainFunction):
            tstat, pval = ss.ttest_rel(self._mcpts, val._mcpts)
            sgn = np.sign(tstat)
            if pval > 0.05:  # Since, statistically, we can't really tell
                return False
            else:
                return True if sgn == -1 else False
        else:
            return len(self._mcpts[self._mcpts < val]) / float(npts)

    def __le__(self, val):
        if isinstance(val, UncertainFunction):
            return self < val  # since it doesn't matter to the test
        else:
            return len(self._mcpts[self._mcpts <= val]) / float(npts)

    def __gt__(self, val):
        """
        If we are comparing two distributions, perform statistical tests,
        otherwise, calculate the probability that the distribution is
        greater than val
        """
        if isinstance(val, UncertainFunction):
            tstat, pval = ss.ttest_rel(self._mcpts, val._mcpts)
            sgn = np.sign(tstat)
            if pval > 0.05:  # Since, statistically, we can't really tell
                return False
            else:
                return True if sgn == 1 else False
        else:
            return 1 - (self <= val)

    def __ge__(self, val):
        if isinstance(val, UncertainFunction):
            return self > val
        else:
            return 1 - (self < val)

    def __bool__(self) -> bool:
        return not (1 - ((self > 0) + (self < 0)))


class UncertainVariable(UncertainFunction):
    """
    UncertainVariable objects track the effects of uncertainty, characterized
    in terms of the first four standard moments of statistical distributions
    (mean, variance, skewness and kurtosis coefficients). Monte Carlo simulation,
    in conjunction with Latin-hypercube based sampling performs the calculations.

    Parameters
    ----------
    rv : scipy.stats.distribution
        A distribution to characterize the uncertainty

    tag : str, optional
        A string identifier when information about this variable is printed to
        the screen

    Notes
    -----

    The ``scipy.stats`` module contains many distributions which we can use to
    perform any necessary uncertainty calculation. It is important to follow
    the initialization syntax for creating any kind of distribution object:

        - *Location* and *Scale* values must use the kwargs ``loc`` and
          ``scale``
        - *Shape* values are passed in as non-keyword arguments before the
          location and scale, (see below for syntax examples)..

    The mathematical operations that can be performed on uncertain objects will
    work for any distribution supplied, but may be misleading if the supplied
    moments or distribution is not accurately defined. Here are some guidelines
    for creating UncertainVariable objects using some of the most common
    statistical distributions:

    +---------------------------+-------------+-------------------+-----+---------+
    | Distribution              | scipy.stats |  args             | loc | scale   |
    |                           | class name  | (shape params)    |     |         |
    +===========================+=============+===================+=====+=========+
    | Normal(mu, sigma)         | norm        |                   | mu  | sigma   |
    +---------------------------+-------------+-------------------+-----+---------+
    | Uniform(a, b)             | uniform     |                   | a   | b-a     |
    +---------------------------+-------------+-------------------+-----+---------+
    | Exponential(lamda)        | expon       |                   |     | 1/lamda |
    +---------------------------+-------------+-------------------+-----+---------+
    | Gamma(k, theta)           | gamma       | k                 |     | theta   |
    +---------------------------+-------------+-------------------+-----+---------+
    | Beta(alpha, beta, [a, b]) | beta        | alpha, beta       | a   | b-a     |
    +---------------------------+-------------+-------------------+-----+---------+
    | Log-Normal(mu, sigma)     | lognorm     | sigma             | mu  |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | Chi-Square(k)             | chi2        | k                 |     |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | F(d1, d2)                 | f           | d1, d2            |     |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | Triangular(a, b, c)       | triang      | c                 | a   | b-a     |
    +---------------------------+-------------+-------------------+-----+---------+
    | Student-T(v)              | t           | v                 |     |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | Weibull(lamda, k)         | exponweib   | lamda, k          |     |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | Bernoulli(p)              | bernoulli   | p                 |     |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | Binomial(n, p)            | binomial    | n, p              |     |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | Geometric(p)              | geom        | p                 |     |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | Hypergeometric(N, n, K)   | hypergeom   | N, n, K           |     |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | Poisson(lamda)            | poisson     | lamda             |     |         |
    +---------------------------+-------------+-------------------+-----+---------+

    Thus, each distribution above would have the same call signature::

        >>> import scipy.stats as ss
        >>> ss.your_dist_here(args, loc=loc, scale=scale)

    ANY SCIPY.STATS.DISTRIBUTION SHOULD WORK! IF ONE DOESN'T, PLEASE LET ME
    KNOW!

    Convenient constructors have been created to make assigning these
    distributions easier. They follow the parameter notation found in the
    respective Wikipedia articles:

    +---------------------------+---------------------------------------------------------------+
    | MCERP Distibution         | Wikipedia page                                                |
    +===========================+===============================================================+
    | N(mu, sigma)              | http://en.wikipedia.org/wiki/Normal_distribution              |
    +---------------------------+---------------------------------------------------------------+
    | U(a, b)                   | http://en.wikipedia.org/wiki/Uniform_distribution_(continuous)|
    +---------------------------+---------------------------------------------------------------+
    | Exp(lamda, [mu])          | http://en.wikipedia.org/wiki/Exponential_distribution         |
    +---------------------------+---------------------------------------------------------------+
    | Gamma(k, theta)           | http://en.wikipedia.org/wiki/Gamma_distribution               |
    +---------------------------+---------------------------------------------------------------+
    | Beta(alpha, beta, [a, b]) | http://en.wikipedia.org/wiki/Beta_distribution                |
    +---------------------------+---------------------------------------------------------------+
    | LogN(mu, sigma)           | http://en.wikipedia.org/wiki/Log-normal_distribution          |
    +---------------------------+---------------------------------------------------------------+
    | X2(df)                    | http://en.wikipedia.org/wiki/Chi-squared_distribution         |
    +---------------------------+---------------------------------------------------------------+
    | F(dfn, dfd)               | http://en.wikipedia.org/wiki/F-distribution                   |
    +---------------------------+---------------------------------------------------------------+
    | Tri(a, b, c)              | http://en.wikipedia.org/wiki/Triangular_distribution          |
    +---------------------------+---------------------------------------------------------------+
    | T(df)                     | http://en.wikipedia.org/wiki/Student's_t-distribution         |
    +---------------------------+---------------------------------------------------------------+
    | Weib(lamda, k)            | http://en.wikipedia.org/wiki/Weibull_distribution             |
    +---------------------------+---------------------------------------------------------------+
    | Bern(p)                   | http://en.wikipedia.org/wiki/Bernoulli_distribution           |
    +---------------------------+---------------------------------------------------------------+
    | B(n, p)                   | http://en.wikipedia.org/wiki/Binomial_distribution            |
    +---------------------------+---------------------------------------------------------------+
    | G(p)                      | http://en.wikipedia.org/wiki/Geometric_distribution           |
    +---------------------------+---------------------------------------------------------------+
    | H(M, n, N)                | http://en.wikipedia.org/wiki/Hypergeometric_distribution      |
    +---------------------------+---------------------------------------------------------------+
    | Pois(lamda)               | http://en.wikipedia.org/wiki/Poisson_distribution             |
    +---------------------------+---------------------------------------------------------------+

    Thus, the following are equivalent::

        >>> x = N(10, 1)
        >>> x = uv(ss.norm(loc=10, scale=1))

    Examples
    --------
    A three-part assembly

        >>> x1 = N(24, 1)
        >>> x2 = N(37, 4)
        >>> x3 = Exp(2)  # Exp(mu=0.5) works too

        >>> Z = (x1 * x2**2) / (15 * (1.5 + x3))
        >>> Z
        uv(1161.46231679, 116646.762981, 0.345533974771, 3.00791101068)

    The result shows the mean, variance, and standardized skewness and kurtosis
    of the output variable Z, which will vary from use to use due to the random
    nature of Monte Carlo simulation and latin-hypercube sampling techniques.

    Basic math operations may be applied to distributions, where all
    statistical calculations are performed using latin-hypercube enhanced Monte
    Carlo simulation. Nearly all of the built-in trigonometric-, logarithm-,
    etc. functions of the ``math`` module have uncertainty-compatible
    counterparts that should be used when possible since they support both
    scalar values and uncertain objects. These can be used after importing the
    ``umath`` module::

        >>> from mcerp.umath import * # sin(), sqrt(), etc.
        >>> sqrt(x1)
        uv(4.89791765647, 0.0104291897681, -0.0614940614672, 3.00264937735)

    At any time, the standardized statistics can be retrieved using::

        >>> x1.mean
        >>> x1.var  # x1.std (standard deviation) is also available
        >>> x1.skew
        >>> x1.kurt

    or all four together with::

        >>> x1.stats

    By default, the Monte Carlo simulation uses 10000 samples, but this can be
    changed at any time with::

        >>> mcerp.npts = number_of_samples

    Any value from 1,000 to 1,000,000 is recommended (more samples means more
    accurate, but also means more time required to perform the calculations).
    Although it can be changed, since variables retain their samples from one
    calculation to the next, this parameter should be changed before any
    calculations are performed to ensure parameter compatibility (this may
    change to be more dynamic in the future, but for now this is how it is).

    Also, to see the underlying distribution of the variable, and if matplotlib
    is installed, simply call its plot method::

        >>> x1.plot()

    Optional kwargs can be any valid kwarg used by matplotlib.pyplot.plot

    See Also
    --------
    N, U, Exp, Gamma, Beta, LogN, X2, F, Tri, PERT, T, Weib, Bern, B, G, H,
    Pois

    """

    def __init__(self, rv, tag=None) -> None:

        assert hasattr(rv, "dist"), (
            "Input must be a  distribution from the scipy.stats module."
        )
        self.rv = rv

        # generate the latin-hypercube points
        self._mcpts = lhd(dist=self.rv, size=npts).flatten()
        self.tag = tag

    def plot(self, hist=False, show=False, **kwargs) -> None:
        """
        Plot the distribution of the UncertainVariable. Continuous
        distributions are plotted with a line plot and discrete distributions
        are plotted with discrete circles.

        Optional
        --------
        hist : bool
            If true, a histogram is displayed
        show : bool
            If ``True``, the figure will be displayed after plotting the
            distribution. If ``False``, an explicit call to ``plt.show()`` is
            required to display the figure.
        kwargs : any valid matplotlib.pyplot.plot kwarg

        """
        import matplotlib.pyplot as plt

        if hist:
            vals = self._mcpts
            low = vals.min()
            high = vals.max()
            h = plt.hist(
                vals,
                bins=int(np.sqrt(len(vals)) + 0.5),
                histtype="stepfilled",
                density=True,
                **kwargs,
            )
            plt.ylim(0, 1.1 * h[0].max())
        else:
            bound = 0.0001
            low = self.rv.ppf(bound)
            high = self.rv.ppf(1 - bound)
            if hasattr(self.rv.dist, "pmf"):
                low = int(low)
                high = int(high)
                vals = list(range(low, high + 1))
                plt.plot(vals, self.rv.pmf(vals), "o", **kwargs)
            else:
                vals = np.linspace(low, high, 500)
                plt.plot(vals, self.rv.pdf(vals), **kwargs)
        plt.xlim(low - (high - low) * 0.1, high + (high - low) * 0.1)

        if show:
            self.show()


uv = UncertainVariable  # a nicer form for the user

# DON'T MOVE THIS IMPORT!!! The prior definitions must be in place before
# importing the correlation-related functions
from . import stats as stats
from . import umath as umath
from .correlate import *


###############################################################################
# Define some convenience constructors for common statistical distributions.
# Hopefully these are a little easier/more intuitive to use than the
# scipy.stats.distributions.
###############################################################################

###############################################################################
# CONTINUOUS DISTRIBUTIONS
###############################################################################


def Beta(alpha, beta, low=0, high=1, tag=None):
    """
    A Beta random variate

    Parameters
    ----------
    alpha : scalar
        The first shape parameter
    beta : scalar
        The second shape parameter

    Optional
    --------
    low : scalar
        Lower bound of the distribution support (default=0)
    high : scalar
        Upper bound of the distribution support (default=1)
    """
    assert alpha > 0 and beta > 0, (
        'Beta "alpha" and "beta" parameters must be greater than zero'
    )
    assert low < high, 'Beta "low" must be less than "high"'
    return uv(ss.beta(alpha, beta, loc=low, scale=high - low), tag=tag)


def BetaPrime(alpha, beta, tag=None):
    """
    A BetaPrime random variate

    Parameters
    ----------
    alpha : scalar
        The first shape parameter
    beta : scalar
        The second shape parameter

    """
    assert alpha > 0 and beta > 0, (
        'BetaPrime "alpha" and "beta" parameters must be greater than zero'
    )
    x = Beta(alpha, beta, tag)
    return x / (1 - x)


def Bradford(q, low=0, high=1, tag=None):
    """
    A Bradford random variate

    Parameters
    ----------
    q : scalar
        The shape parameter
    low : scalar
        The lower bound of the distribution (default=0)
    high : scalar
        The upper bound of the distribution (default=1)
    """
    assert q > 0, 'Bradford "q" parameter must be greater than zero'
    assert low < high, 'Bradford "low" parameter must be less than "high"'
    return uv(ss.bradford(q, loc=low, scale=high - low), tag=tag)


def Burr(c, k, tag=None):
    """
    A Burr random variate

    Parameters
    ----------
    c : scalar
        The first shape parameter
    k : scalar
        The second shape parameter

    """
    assert c > 0 and k > 0, (
        'Burr "c" and "k" parameters must be greater than zero'
    )
    return uv(ss.burr(c, k), tag=tag)


def ChiSquared(k, tag=None):
    """
    A Chi-Squared random variate

    Parameters
    ----------
    k : int
        The degrees of freedom of the distribution (must be greater than one)
    """
    assert int(k) == k and k >= 1, (
        'Chi-Squared "k" must be an integer greater than 0'
    )
    return uv(ss.chi2(k), tag=tag)


Chi2 = ChiSquared  # for more concise use


def Erf(h, tag=None):
    """
    An Error Function random variate.

    This distribution is derived from a normal distribution by setting
    m = 0 and s = 1/(h*sqrt(2)), and thus is used in similar situations
    as the normal distribution.

    Parameters
    ----------
    h : scalar
        The scale parameter.
    """
    assert h > 0, 'Erf "h" must be greater than zero'
    return Normal(0, 1 / (h * 2**0.5), tag)


def Erlang(k, lamda, tag=None):
    """
    An Erlang random variate.

    This distribution is the same as a Gamma(k, theta) distribution, but
    with the restriction that k must be a positive integer. This
    is provided for greater compatibility with other simulation tools, but
    provides no advantage over the Gamma distribution in its applications.

    Parameters
    ----------
    k : int
        The shape parameter (must be a positive integer)
    lamda : scalar
        The scale parameter (must be greater than zero)
    """
    assert int(k) == k and k > 0, 'Erlang "k" must be a positive integer'
    assert lamda > 0, 'Erlang "lamda" must be greater than zero'
    return Gamma(k, lamda, tag)


def Exponential(lamda, tag=None):
    """
    An Exponential random variate

    Parameters
    ----------
    lamda : scalar
        The inverse scale (as shown on Wikipedia). (FYI: mu = 1/lamda.)
    """
    assert lamda > 0, 'Exponential "lamda" must be greater than zero'
    return uv(ss.expon(scale=1.0 / lamda), tag=tag)


Exp = Exponential  # for more concise use


def ExtValueMax(mu, sigma, tag=None):
    """
    An Extreme Value Maximum random variate.

    Parameters
    ----------
    mu : scalar
        The location parameter
    sigma : scalar
        The scale parameter (must be greater than zero)
    """
    assert sigma > 0, 'ExtremeValueMax "sigma" must be greater than zero'
    p = U(0, 1)._mcpts[:]
    return UncertainFunction(mu - sigma * np.log(-np.log(p)), tag=tag)


EVMax = ExtValueMax  # for more concise use


def ExtValueMin(mu, sigma, tag=None):
    """
    An Extreme Value Minimum random variate.

    Parameters
    ----------
    mu : scalar
        The location parameter
    sigma : scalar
        The scale parameter (must be greater than zero)
    """
    assert sigma > 0, 'ExtremeValueMin "sigma" must be greater than zero'
    p = U(0, 1)._mcpts[:]
    return UncertainFunction(mu + sigma * np.log(-np.log(1 - p)), tag=tag)


EVMin = ExtValueMin  # for more concise use


def Fisher(d1, d2, tag=None):
    """
    An F (fisher) random variate

    Parameters
    ----------
    d1 : int
        Numerator degrees of freedom
    d2 : int
        Denominator degrees of freedom
    """
    assert int(d1) == d1 and d1 >= 1, (
        'Fisher (F) "d1" must be an integer greater than 0'
    )
    assert int(d2) == d2 and d2 >= 1, (
        'Fisher (F) "d2" must be an integer greater than 0'
    )
    return uv(ss.f(d1, d2), tag=tag)


F = Fisher  # for more concise use


def Gamma(k, theta, tag=None):
    """
    A Gamma random variate

    Parameters
    ----------
    k : scalar
        The shape parameter (must be positive and non-zero)
    theta : scalar
        The scale parameter (must be positive and non-zero)
    """
    assert k > 0 and theta > 0, (
        'Gamma "k" and "theta" parameters must be greater than zero'
    )
    return uv(ss.gamma(k, scale=theta), tag=tag)


def LogNormal(mu, sigma, tag=None):
    """
    A Log-Normal random variate

    Parameters
    ----------
    mu : scalar
        The location parameter
    sigma : scalar
        The scale parameter (must be positive and non-zero)
    """
    assert sigma > 0, 'Log-Normal "sigma" must be positive'
    return uv(ss.lognorm(sigma, loc=mu), tag=tag)


LogN = LogNormal  # for more concise use


def Normal(mu, sigma, tag=None):
    """
    A Normal (or Gaussian) random variate

    Parameters
    ----------
    mu : scalar
        The mean value of the distribution
    sigma : scalar
        The standard deviation (must be positive and non-zero)
    """
    assert sigma > 0, 'Normal "sigma" must be greater than zero'
    return uv(ss.norm(loc=mu, scale=sigma), tag=tag)


N = Normal  # for more concise use


def Pareto(q, a, tag=None):
    """
    A Pareto random variate (first kind)

    Parameters
    ----------
    q : scalar
        The scale parameter
    a : scalar
        The shape parameter (the minimum possible value)
    """
    assert q > 0 and a > 0, 'Pareto "q" and "a" must be positive scalars'
    p = Uniform(0, 1, tag)
    return a * (1 - p) ** (-1.0 / q)


def Pareto2(q, b, tag=None):
    """
    A Pareto random variate (second kind). This form always starts at the
    origin.

    Parameters
    ----------
    q : scalar
        The scale parameter
    b : scalar
        The shape parameter
    """
    assert q > 0 and b > 0, 'Pareto2 "q" and "b" must be positive scalars'
    return Pareto(q, b, tag) - b


def PERT(low, peak, high, g=4.0, tag=None):
    """
    A PERT random variate

    Parameters
    ----------
    low : scalar
        Lower bound of the distribution support
    peak : scalar
        The location of the distribution's peak (low <= peak <= high)
    high : scalar
        Upper bound of the distribution support

    Optional
    --------
    g : scalar
        Controls the uncertainty of the distribution around the peak. Smaller
        values make the distribution flatter and more uncertain around the
        peak while larger values make it focused and less uncertain around
        the peak. (Default: 4)
    """
    a, b, c = [float(x) for x in [low, peak, high]]
    assert a <= b <= c, (
        'PERT "peak" must be greater than "low" and less than "high"'
    )
    assert g >= 0, 'PERT "g" must be non-negative'
    mu = (a + g * b + c) / (g + 2)
    if mu == b:
        a1 = a2 = 3.0
    else:
        a1 = ((mu - a) * (2 * b - a - c)) / ((b - mu) * (c - a))
        a2 = a1 * (c - mu) / (mu - a)

    return Beta(a1, a2, a, c, tag)


def StudentT(v, tag=None):
    """
    A Student-T random variate

    Parameters
    ----------
    v : int
        The degrees of freedom of the distribution (must be greater than one)
    """
    assert int(v) == v and v >= 1, (
        'Student-T "v" must be an integer greater than 0'
    )
    return uv(ss.t(v), tag=tag)


T = StudentT  # for more concise use


def Triangular(low, peak, high, tag=None):
    """
    A triangular random variate

    Parameters
    ----------
    low : scalar
        Lower bound of the distribution support
    peak : scalar
        The location of the triangle's peak (low <= peak <= high)
    high : scalar
        Upper bound of the distribution support
    """
    assert low <= peak <= high, (
        'Triangular "peak" must lie between "low" and "high"'
    )
    low, peak, high = [float(x) for x in [low, peak, high]]
    return uv(
        ss.triang(
            (1.0 * peak - low) / (high - low), loc=low, scale=(high - low)
        ),
        tag=tag,
    )


Tri = Triangular  # for more concise use


def Uniform(low, high, tag=None):
    """
    A Uniform random variate

    Parameters
    ----------
    low : scalar
        Lower bound of the distribution support.
    high : scalar
        Upper bound of the distribution support.
    """
    assert low < high, 'Uniform "low" must be less than "high"'
    return uv(ss.uniform(loc=low, scale=high - low), tag=tag)


U = Uniform  # for more concise use


def Weibull(lamda, k, tag=None):
    """
    A Weibull random variate

    Parameters
    ----------
    lamda : scalar
        The scale parameter
    k : scalar
        The shape parameter
    """
    assert lamda > 0 and k > 0, (
        'Weibull "lamda" and "k" parameters must be greater than zero'
    )
    return uv(ss.exponweib(lamda, k), tag=tag)


Weib = Weibull  # for more concise use


###############################################################################
# DISCRETE DISTRIBUTIONS
###############################################################################


def Bernoulli(p, tag=None):
    """
    A Bernoulli random variate

    Parameters
    ----------
    p : scalar
        The probability of success
    """
    assert 0 < p < 1, (
        'Bernoulli probability "p" must be between zero and one, non-inclusive'
    )
    return uv(ss.bernoulli(p), tag=tag)


Bern = Bernoulli  # for more concise use


def Binomial(n, p, tag=None):
    """
    A Binomial random variate

    Parameters
    ----------
    n : int
        The number of trials
    p : scalar
        The probability of success
    """
    assert int(n) == n and n > 0, (
        'Binomial number of trials "n" must be an integer greater than zero'
    )
    assert 0 < p < 1, (
        'Binomial probability "p" must be between zero and one, non-inclusive'
    )
    return uv(ss.binom(n, p), tag=tag)


B = Binomial  # for more concise use


def Geometric(p, tag=None):
    """
    A Geometric random variate

    Parameters
    ----------
    p : scalar
        The probability of success
    """
    assert 0 < p < 1, (
        'Geometric probability "p" must be between zero and one, non-inclusive'
    )
    return uv(ss.geom(p), tag=tag)


G = Geometric  # for more concise use


def Hypergeometric(N, n, K, tag=None):
    """
    A Hypergeometric random variate

    Parameters
    ----------
    N : int
        The total population size
    n : int
        The number of individuals of interest in the population
    K : int
        The number of individuals that will be chosen from the population

    Example
    -------
    (Taken from the wikipedia page) Assume we have an urn with two types of
    marbles, 45 black ones and 5 white ones. Standing next to the urn, you
    close your eyes and draw 10 marbles without replacement. What is the
    probability that exactly 4 of the 10 are white?
    ::

        >>> black = 45
        >>> white = 5
        >>> draw = 10

        # Now we create the distribution
        >>> h = H(black + white, white, draw)

        # To check the probability, in this case, we can use the underlying
        #  scipy.stats object
        >>> h.rv.pmf(4)  # What is the probability that white count = 4?
        0.0039645830580151975

    """
    assert int(N) == N and N > 0, (
        'Hypergeometric total population size "N" must be an integer greater than zero.'
    )
    assert int(n) == n and 0 < n <= N, (
        'Hypergeometric interest population size "n" must be an integer greater than zero and no more than the total population size.'
    )
    assert int(K) == K and 0 < K <= N, (
        'Hypergeometric chosen population size "K" must be an integer greater than zero and no more than the total population size.'
    )
    return uv(ss.hypergeom(N, n, K), tag=tag)


H = Hypergeometric  # for more concise use


def Poisson(lamda, tag=None):
    """
    A Poisson random variate

    Parameters
    ----------
    lamda : scalar
        The rate of an occurance within a specified interval of time or space.
    """
    assert lamda > 0, 'Poisson "lamda" must be greater than zero.'
    return uv(ss.poisson(lamda), tag=tag)


Pois = Poisson  # for more concise use


###############################################################################
# STATISTICAL FUNCTIONS
###############################################################################


def covariance_matrix(nums_with_uncert):
    """
    Calculate the covariance matrix of uncertain variables, oriented by the
    order of the inputs

    Parameters
    ----------
    nums_with_uncert : array-like
        A list of variables that have an associated uncertainty

    Returns
    -------
    cov_matrix : 2d-array-like
        A nested list containing covariance values

    Example
    -------

        >>> x = N(1, 0.1)
        >>> y = N(10, 0.1)
        >>> z = x + 2 * y
        >>> covariance_matrix([x, y, z])
        [[  9.99694861e-03   2.54000840e-05   1.00477488e-02]
         [  2.54000840e-05   9.99823207e-03   2.00218642e-02]
         [  1.00477488e-02   2.00218642e-02   5.00914772e-02]]

    """
    ufuncs = list(map(to_uncertain_func, nums_with_uncert))
    cov_matrix = []
    for i1, expr1 in enumerate(ufuncs):
        coefs_expr1 = []
        mean1 = expr1.mean
        for _i2, expr2 in enumerate(ufuncs[: i1 + 1]):
            mean2 = expr2.mean
            coef = np.mean((expr1._mcpts - mean1) * (expr2._mcpts - mean2))
            coefs_expr1.append(coef)
        cov_matrix.append(coefs_expr1)

    # We symmetrize the matrix:
    for i, covariance_coefs in enumerate(cov_matrix):
        covariance_coefs.extend(
            cov_matrix[j][i] for j in range(i + 1, len(cov_matrix))
        )

    return cov_matrix


def correlation_matrix(nums_with_uncert):
    """
    Calculate the correlation matrix of uncertain variables, oriented by the
    order of the inputs

    Parameters
    ----------
    nums_with_uncert : array-like
        A list of variables that have an associated uncertainty

    Returns
    -------
    corr_matrix : 2d-array-like
        A nested list containing covariance values

    Example
    -------

        >>> x = N(1, 0.1)
        >>> y = N(10, 0.1)
        >>> z = x + 2 * y
        >>> correlation_matrix([x, y, z])
        [[ 0.99969486  0.00254001  0.4489385 ]
         [ 0.00254001  0.99982321  0.89458702]
         [ 0.4489385   0.89458702  1.        ]]

    """
    ufuncs = list(map(to_uncertain_func, nums_with_uncert))
    data = np.vstack([ufunc._mcpts for ufunc in ufuncs])
    return np.corrcoef(data.T, rowvar=0)
