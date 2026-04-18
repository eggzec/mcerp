# ``mcerp`` Second Order Error Propagation for Python

[![Tests](https://github.com/eggzec/mcerp/actions/workflows/code_test.yml/badge.svg)](https://github.com/eggzec/mcerp/actions/workflows/code_test.yml)
[![Documentation](https://github.com/eggzec/mcerp/actions/workflows/docs_build.yml/badge.svg)](https://github.com/eggzec/mcerp/actions/workflows/docs_build.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![codecov](https://codecov.io/gh/eggzec/mcerp/branch/master/graph/badge.svg)](https://codecov.io/gh/eggzec/mcerp)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=eggzec_mcerp&metric=alert_status)](https://sonarcloud.io/project/overview?id=eggzec_mcerp)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](./LICENSE)

[![PyPI Downloads](https://img.shields.io/pypi/dm/mcerp.svg?label=PyPI%20downloads)](https://pypi.org/project/mcerp/)
[![Python versions](https://img.shields.io/pypi/pyversions/mcerp.svg)](https://pypi.org/project/mcerp/)

## Overview

``mcerp`` is the Python implementation of the original Fortran code `mcerp` by N. D. Cox to apply a second-order analysis to [error propagation](http://en.wikipedia.org/wiki/Propagation_of_uncertainty) (or uncertainty analysis). The ``mcerp`` package allows you to **easily** and **transparently** track the effects of uncertainty through mathematical calculations. Advanced mathematical functions, similar to those in the standard [math](http://docs.python.org/library/math.html) module can also be evaluated directly.

In order to correctly use ``mcerp``, the **first eight statistical moments** of the underlying distribution are required. These are the *mean*, *variance*, and then the *standardized third through eighth moments*. These can be input manually in the form of an array, but they can also be **conveniently generated** using either the **nice constructors** or directly by using the distributions from the ``scipy.stats`` sub-module. See the examples below for usage examples of both input methods. The result of all calculations generates a *mean*, *variance*, and *standardized skewness and kurtosis* coefficients.

## Basic examples

Let's begin by importing all the available constructors:

```python
>>> from mcerp import *   # uv, normal, uniform, exponential, etc.
```

Now, we can see that there are several equivalent ways to specify a statistical distribution, say a Normal distribution with a mean value of 10 and a standard deviation of 1:

- Manually input the first 8 moments (mean, variance, and 3rd-8th standardized central moments):

```python
>>> x = uv([10, 1, 0, 3, 0, 15, 0, 105])
```

- Use the ``rv`` kwarg to input a distribution from the ``scipy.stats`` module:

```python
>>> x = uv(rv=ss.norm(loc=10, scale=1))
```

- Use a built-in convenience constructor (typically the easiest if you can):

```python
>>> x = normal(10, 1)
```

### A Simple Example

Now let's walk through an example of a three-part assembly stack-up:

```python
>>> x1 = normal(24, 1)  # normally distributed
>>> x2 = normal(37, 4)  # normally distributed
>>> x3 = exponential(2)  # exponentially distributed
>>> Z = (x1*x2**2)/(15*(1.5 + x3))
```

We can now see the results of the calculations in two ways:

1. The usual ``print`` statement (or simply the object if in a terminal):

```python
>>> Z  # "print" is optional at the command-line
uv(1176.45, 99699.6822917, 0.708013052944, 6.16324345127)
```

2. The ``describe`` class method that explains briefly what the values are:

```python
>>> Z.describe()
mcerp Uncertain Value:
 > Mean...................  1176.45
 > Variance...............  99699.6822917
 > Skewness Coefficient...  0.708013052944
 > Kurtosis Coefficient...  6.16324345127
```

### Distribution Moments

The eight moments of any input variable (and four of any output variable) can be accessed using the ``moments`` class method, as in:

```python
>>> x1.moments()
[24.0, 1.0, 0.0, 3.0000000000000053, 0.0, 15.000000000000004, 0.0, 105.0]
>>> Z.moments()
[1176.45, 99699.6822917, 0.708013052944, 6.16324345127]
```

### Correlations

Statistical correlations are correctly handled, even after calculations have taken place:

```python
>>> x1 - x1
0.0
>>> square = x1**2
>>> square - x1*x1
0.0
```

### Derivatives

Derivatives with respect to original variables are calculated and are accessed using the **intuitive class methods**:

```python
>>> Z.d(x1)  # dZ/dx1
45.63333333333333

>>> Z.d2(x2)  # d^2Z/dx2^2
1.6

>>> Z.d2c(x1, x3)  # d^2Z/dx1dx3 (order doesn't matter)
-22.816666666666666
```

When we need multiple derivatives at a time, we can use the ``gradient`` and ``hessian`` class methods:

```python
>>> Z.gradient([x1, x2, x3])
[45.63333333333333, 59.199999999999996, -547.6]

>>> Z.hessian([x1, x2, x3])
[[0.0, 2.466666666666667, -22.816666666666666], [2.466666666666667, 1.6, -29.6], [-22.816666666666666, -29.6, 547.6]]
```

### Error Components/Variance Contributions

Another useful feature is available through the ``error_components`` class method that has various ways of representing the first- and second-order variance components:

```python
>>> Z.error_components(pprint=True)
COMPOSITE VARIABLE ERROR COMPONENTS
uv(37.0, 16.0, 0.0, 3.0) = 58202.9155556 or 58.378236%
uv(24.0, 1.0, 0.0, 3.0) = 2196.15170139 or 2.202767%
uv(0.5, 0.25, 2.0, 9.0) = -35665.8249653 or 35.773258%
```

### Advanced Example

Here's a *slightly* more advanced example, estimating the statistical properties of volumetric gas flow through an orifice meter:

```python
>>> from mcerp import normal, umath  # sin, exp, sqrt, etc.
>>> H = normal(64, 0.5)
>>> M = normal(16, 0.1)
>>> P = normal(361, 2)
>>> t = normal(165, 0.5)
>>> C = 38.4
>>> Q = C*umath.sqrt((520*H*P)/(M*(t + 460)))
>>> Q.describe()
mcerp Uncertain Value:
 > Mean...................  1330.99973939
 > Variance...............  58.210762839
 > Skewness Coefficient...  0.0109422068056
 > Kurtosis Coefficient...  3.00032693502
```

This seems to indicate that even though there are products, divisions, and the usage of ``sqrt``, the result resembles a normal distribution (i.e., Q ~ N(1331, 7.63), where the standard deviation = sqrt(58.2) = 7.63).

## Main Features

1. **Transparent calculations** with derivatives automatically calculated. **No or little modification** to existing code required.
2. Basic `NumPy` support without modification.
3. Nearly all standard [math](http://docs.python.org/library/math.html) module functions supported through the ``mcerp.umath`` sub-module. If you think a function is in there, it probably is.
4. Nearly all derivatives calculated analytically.
5. **Easy continuous distribution constructors**:
    - ``normal(mu, sigma)`` or ``N(mu, sigma)`` : [Normal distribution](http://en.wikipedia.org/wiki/Normal_distribution)
    - ``uniform(a, b)`` or ``U(a, b)`` : [Uniform distribution](http://en.wikipedia.org/wiki/Uniform_distribution_(continuous))
    - ``exponential(lamda, [mu])`` or ``Exp(lamda, [mu])`` : [Exponential distribution](http://en.wikipedia.org/wiki/Exponential_distribution)
    - ``gamma(k, theta)`` or ``Gamma(k, theta)`` : [Gamma distribution](http://en.wikipedia.org/wiki/Gamma_distribution)
    - ``beta(alpha, beta, [a, b])`` or ``Beta(alpha, beta, [a, b])`` : [Beta distribution](http://en.wikipedia.org/wiki/Beta_distribution)
    - ``log_normal(mu, sigma)`` or ``LogN(mu, sigma)`` : [Log-normal distribution](http://en.wikipedia.org/wiki/Log-normal_distribution)
    - ``chi_squared(k)`` or ``Chi2(k)`` : [Chi-squared distribution](http://en.wikipedia.org/wiki/Chi-squared_distribution)
    - ``f_distribution(d1, d2)`` or ``F(d1, d2)`` : [F-distribution](http://en.wikipedia.org/wiki/F-distribution)
    - ``triangular(a, b, c)`` or ``Tri(a, b, c)`` : [Triangular distribution](http://en.wikipedia.org/wiki/Triangular_distribution)
    - ``student_t(v)`` or ``T(v)`` : [T-distribution](http://en.wikipedia.org/wiki/Student's_t-distribution)
    - ``weibull(lamda, k)`` or ``Weib(lamda, k)`` : [Weibull distribution](http://en.wikipedia.org/wiki/Weibull_distribution)

    The location, scale, and shape parameters follow the notation in the respective Wikipedia articles. *Discrete distributions are not recommended for use at this time. If you need discrete distributions, try the* [mcerp](http://pypi.python.org/pypi/mcerp) *python package instead.*

## Installation

You have several easy, convenient options to install the ``mcerp`` package.

### pip

```bash
pip install mcerp
```

To install with plotting support:
```bash
pip install mcerp[plot]
```

To install all optional dependencies:
```bash
pip install mcerp[all]
```


### uv

```bash
uv add mcerp
uv sync
```

Or in an existing uv environment:
```bash
uv pip install mcerp
```


### git

To install the latest version from git:
```bash
pip install --upgrade "git+https://github.com/eggzec/mcerp.git#egg=mcerp"
```

#### Requirements

- Python >=3.10
- [NumPy](http://www.numpy.org/) : Numeric Python
- [SciPy](http://scipy.org) : Scientific Python (the nice distribution constructors require this)
- [Matplotlib](http://matplotlib.org/) : Python plotting library (optional)

## See Also

- [uncertainties](http://pypi.python.org/pypi/uncertainties) : First-order error propagation.
- [mcerp](http://pypi.python.org/pypi/mcerp) : Real-time latin-hypercube sampling-based Monte Carlo error propagation.

## Acknowledgements

The author wishes to thank [Eric O. LEBIGOT](http://www.linkedin.com/pub/eric-lebigot/22/293/277) who first developed the [uncertainties](http://pypi.python.org/pypi/uncertainties) python package (for first-order error propagation), from which many inspiring ideas (like maintaining object correlations, etc.) are re-used and/or have been slightly evolved. *If you don't need second order functionality, his package is an excellent alternative since it is optimized for first-order uncertainty analysis.*

## References

- N.D. Cox, 1979, *Tolerance Analysis by Computer*, Journal of Quality Technology, Vol. 11, No. 2, pp. 80-87
