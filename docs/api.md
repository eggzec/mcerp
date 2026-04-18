# API Reference

Import the public API from `mcerp`:

```python
from mcerp import *
```

For explicit imports:

```python
from mcerp import N, U, Exp, Gamma, Beta, correlate, correlation_matrix
```

## Core Classes

| Name | Purpose |
| --- | --- |
| `UncertainVariable` | An input uncertainty backed by a `scipy.stats` distribution. |
| `UncertainFunction` | A calculated uncertain value produced by arithmetic or functions. |
| `uv(rv, tag=None)` | Create an uncertain variable directly from a frozen `scipy.stats` distribution. |
| `npts` | Global sample count. Default is `10000`. Set before creating variables. |

## Tracking Variables

Constructors accept an optional `tag=...` keyword argument. This is useful when
several variables share the same distribution but represent different physical
quantities:

```python
from mcerp import N

length = N(24, 1, tag="length")
width = N(24, 1, tag="width")

length.tag
width.describe()
```

If `.describe()` is called without an explicit name, the tag is shown in the
printed output.

## Common Properties and Methods

| Attribute or method | Description |
| --- | --- |
| `.mean` | Sample mean. |
| `.var` | Sample variance. |
| `.std` | Sample standard deviation. |
| `.skew` | Standardized skewness coefficient. |
| `.kurt` | Standardized kurtosis coefficient. |
| `.stats` | `[mean, var, skew, kurt]`. |
| `.percentile(p)` | Value at percentile `p`, where `p` is between `0` and `1`. |
| `.describe(name=None)` | Print a formatted moment summary. |
| `.plot(hist=False, show=False, **kwargs)` | Plot a distribution or sampled output. Requires Matplotlib. |
| `.show()` | Display pending Matplotlib plots. |

## Continuous Distributions

| Constructor | Distribution |
| --- | --- |
| `Beta(alpha, beta, [low, high])` | Beta distribution |
| `BetaPrime(alpha, beta, [loc, scale])` | Beta prime distribution |
| `Bradford(q, [low, high])` | Bradford distribution |
| `Burr(c, k)` | Burr distribution |
| `ChiSquared(k)` or `Chi2(k)` | Chi-squared distribution |
| `Erf(h)` | Error-function distribution |
| `Erlang(k, lamda)` | Erlang distribution |
| `Exponential(lamda)` or `Exp(lamda)` | Exponential distribution |
| `ExtValueMax(mu, sigma)` or `EVMax(mu, sigma)` | Extreme value maximum distribution |
| `ExtValueMin(mu, sigma)` or `EVMin(mu, sigma)` | Extreme value minimum distribution |
| `Fisher(d1, d2)` or `F(d1, d2)` | F distribution |
| `Gamma(k, theta)` | Gamma distribution |
| `LogNormal(mu, sigma)` or `LogN(mu, sigma)` | Log-normal distribution |
| `Normal(mu, sigma)` or `N(mu, sigma)` | Normal distribution |
| `Pareto(q, a)` | Pareto distribution, first kind |
| `Pareto2(q, b)` | Pareto distribution, second kind |
| `PERT(low, peak, high)` | PERT distribution |
| `StudentT(v)` or `T(v)` | Student's t distribution |
| `Triangular(low, peak, high)` or `Tri(low, peak, high)` | Triangular distribution |
| `Uniform(low, high)` or `U(low, high)` | Uniform distribution |
| `Weibull(lamda, k)` or `Weib(lamda, k)` | Weibull distribution |

Example:

```python
from mcerp import N, Uniform, Gamma

x = N(10, 1)
y = Uniform(0, 5)
z = Gamma(9, 1 / 6)
```

## Discrete Distributions

| Constructor | Distribution |
| --- | --- |
| `Bernoulli(p)` or `Bern(p)` | Bernoulli distribution |
| `Binomial(n, p)` or `B(n, p)` | Binomial distribution |
| `Geometric(p)` or `G(p)` | Geometric distribution |
| `Hypergeometric(N, n, K)` or `H(N, n, K)` | Hypergeometric distribution |
| `Poisson(lamda)` or `Pois(lamda)` | Poisson distribution |

Example:

```python
from mcerp import H, Pois

h = H(50, 5, 10)
p = Pois(3)
```

## SciPy Distributions

Any compatible frozen `scipy.stats` distribution can be wrapped with `uv`:

```python
import scipy.stats as ss
from mcerp import uv

x = uv(ss.norm(loc=10, scale=1))
```

## Correlation Utilities

| Function | Description |
| --- | --- |
| `correlate(params, corrmat)` | Reorder samples in-place to approximately match a target correlation matrix. |
| `correlation_matrix(values)` | Return the sample correlation matrix. |
| `covariance_matrix(values)` | Return the sample covariance matrix. |
| `plotcorr(values, labels=None, full=False, show=False)` | Plot pairwise sample relationships. Requires Matplotlib. |
| `induce_correlations(data, corrmat)` | Return correlated sample data for a target matrix. |
| `chol(matrix)` | Cholesky-like helper used by correlation routines. |

## Mathematical Functions

Use `mcerp.umath` when applying mathematical functions to uncertain values:

```python
import mcerp.umath as umath

y = umath.sqrt(x) + umath.sin(x)
```

Available functions include:

`acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`, `ceil`, `cos`, `cosh`,
`degrees`, `exp`, `expm1`, `fabs`, `floor`, `hypot`, `ln`, `log`, `log10`,
`log1p`, `radians`, `sin`, `sinh`, `sqrt`, `tan`, `tanh`, and `trunc`.
