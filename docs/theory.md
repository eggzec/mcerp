# Theory

## Overview

`mcerp` implements Monte Carlo error propagation (MCERP), a method for
estimating the output distribution of a deterministic function whose inputs are
random variables. Given a model

$$
Y = f(X_1, X_2, \ldots, X_d)
$$

where each input \(X_i\) is described by a probability distribution, MCERP
generates samples from the inputs, evaluates the model at those sampled values,
and characterizes the output sample. The result is an uncertain value with
estimated mean, variance, skewness, kurtosis, percentiles, and sampled
probabilities.

The method is particularly useful for:

- Tolerance analysis: predicting how part-to-part variation propagates through
  an assembly.
- Uncertainty analysis: quantifying the uncertainty in calculated quantities.
- Risk analysis: estimating probabilities such as \(P(Y > y_0)\).
- Nonlinear models: evaluating functions where first-order formulas are too
  limited.
- Distribution-aware calculations: carrying skewed, bounded, discrete, or
  heavy-tailed input distributions through a model.

## Monte Carlo Error Propagation

### System Model

Let \(f\) be a deterministic function of \(d\) uncertain inputs:

$$
Y = f(X_1, X_2, \ldots, X_d)
$$

Each input \(X_i\) has a cumulative distribution function \(F_i\). In MCERP,
the unknown output distribution is approximated by evaluating \(f\) at a finite
set of sampled input vectors:

$$
\mathbf{x}^{(j)} =
\left(x_1^{(j)}, x_2^{(j)}, \ldots, x_d^{(j)}\right),
\qquad j = 1, 2, \ldots, N
$$

The output sample is

$$
y^{(j)} = f\left(x_1^{(j)}, x_2^{(j)}, \ldots, x_d^{(j)}\right)
$$

and the set

$$
\{y^{(1)}, y^{(2)}, \ldots, y^{(N)}\}
$$

is treated as an empirical approximation of the output distribution.

### Propagation Without Derivatives

Classical error propagation often relies on a Taylor expansion around the input
means. For small uncertainties and nearly linear models this is efficient, but
it can lose information when:

- the model is strongly nonlinear,
- the output is bounded or thresholded,
- the input distributions are asymmetric,
- products, ratios, powers, or logarithms dominate the calculation,
- the desired answer is a probability or percentile rather than only a
  variance.

MCERP avoids derivative formulas. The actual Python calculation is applied to
the sampled values. If the model can be written with supported arithmetic and
`mcerp.umath` functions, the uncertainty propagates through the same operations
used for deterministic inputs.

## Latin Hypercube Sampling

### Probability-Space Sampling

`mcerp` uses Latin hypercube sampling (LHS), not plain random sampling. For a
sample count \(N\), the unit interval is split into \(N\) equal-probability
strata:

$$
I_j = \left[\frac{j-1}{N}, \frac{j}{N}\right],
\qquad j = 1, 2, \ldots, N
$$

For each input dimension, one probability value is selected from every stratum:

$$
u_i^{(j)} \in I_j
$$

The sampled probabilities are then randomly permuted within each input
dimension. This keeps each input well spread over its probability range while
still producing randomized sample pairings.

### Transforming To Input Distributions

Each probability sample is transformed through the inverse cumulative
distribution function, also called the percent point function:

$$
x_i^{(j)} = F_i^{-1}\left(u_i^{(j)}\right)
$$

In the implementation, this is the `ppf` method of the corresponding
`scipy.stats` distribution. This is why `mcerp` can use both its convenient
constructors and arbitrary compatible frozen SciPy distributions.

### Why LHS Helps

With ordinary random sampling, points may cluster in one region and leave
another region underrepresented. LHS forces every input distribution to be
sampled across its full probability range. For the same \(N\), this usually
gives more stable moment estimates than unconstrained random sampling,
especially when the number of model evaluations is modest.

LHS does not make the answer exact. MCERP still produces estimates, and results
can vary between sessions because the stratified points and permutations are
random.

## Output Sample Construction

### Elementwise Arithmetic

Each uncertain value stores an array of sampled points:

$$
\mathbf{x}_i =
\left[x_i^{(1)}, x_i^{(2)}, \ldots, x_i^{(N)}\right]
$$

Arithmetic is performed element-by-element. For example, if

$$
Z = X_1 + X_2
$$

then

$$
z^{(j)} = x_1^{(j)} + x_2^{(j)}
$$

for every sample index \(j\). Multiplication, division, powers, negation, and
absolute value follow the same samplewise rule.

### Mathematical Functions

Functions in `mcerp.umath` also operate on the sample arrays. For example,

$$
Z = \sqrt{X}
$$

is evaluated as

$$
z^{(j)} = \sqrt{x^{(j)}}
$$

for each sampled point. This lets functions such as `sqrt`, `sin`, `log`, and
`exp` preserve the simulated uncertainty structure.

## Sample Moment Estimates

### Mean

The estimated mean of the output is the sample average:

$$
\hat{\mu}_Y = \frac{1}{N}\sum_{j=1}^{N} y^{(j)}
$$

In `mcerp`, this is available as:

```python
Y.mean
```

### Variance and Standard Deviation

The estimated variance is

$$
\hat{\sigma}_Y^2 =
\frac{1}{N}\sum_{j=1}^{N}\left(y^{(j)}-\hat{\mu}_Y\right)^2
$$

and the estimated standard deviation is

$$
\hat{\sigma}_Y = \sqrt{\hat{\sigma}_Y^2}
$$

In `mcerp`:

```python
Y.var
Y.std
```

### Higher Central Moments

The third and fourth central moments are estimated by

$$
\hat{\mu}_{3,Y} =
\frac{1}{N}\sum_{j=1}^{N}\left(y^{(j)}-\hat{\mu}_Y\right)^3
$$

and

$$
\hat{\mu}_{4,Y} =
\frac{1}{N}\sum_{j=1}^{N}\left(y^{(j)}-\hat{\mu}_Y\right)^4
$$

These moments measure distribution shape: asymmetry and tail weight.

## Output Distribution Characterization

### Skewness and Kurtosis

The skewness coefficient is the standardized third central moment:

$$
\hat{\gamma}_{1,Y} =
\frac{\hat{\mu}_{3,Y}}{\hat{\sigma}_Y^3}
$$

A symmetric distribution has skewness near \(0\). Positive skewness indicates a
longer right tail; negative skewness indicates a longer left tail.

The kurtosis coefficient is the standardized fourth central moment:

$$
\hat{\beta}_{2,Y} =
\frac{\hat{\mu}_{4,Y}}{\hat{\sigma}_Y^4}
$$

`mcerp` reports ordinary kurtosis, not excess kurtosis. A normal distribution
therefore has kurtosis near \(3\), not \(0\).

In `mcerp`:

```python
Y.skew
Y.kurt
Y.stats
```

where `.stats` returns:

```python
[Y.mean, Y.var, Y.skew, Y.kurt]
```

### Percentiles

Percentiles are estimated from the sorted output sample. For a percentile
\(p\), where \(0 \le p \le 1\), the value \(q_p\) satisfies approximately

$$
P(Y \le q_p) = p
$$

In `mcerp`:

```python
Y.percentile(0.95)
```

Multiple percentiles can be requested at once:

```python
Y.percentile([0.05, 0.5, 0.95])
```

### Probability Estimates

When an uncertain value is compared with a scalar, `mcerp` estimates a sampled
probability. For example,

$$
P(Y > a) \approx
\frac{1}{N}\sum_{j=1}^{N}\mathbf{1}\left(y^{(j)} > a\right)
$$

In Python:

```python
Y > a
Y <= a
Y == a
```

For continuous distributions, equality with an exact scalar usually has
probability \(0\). For discrete distributions, equality can estimate a
meaningful probability mass.

## Comparing Two Uncertain Values

When two uncertain values are compared with `<` or `>`, `mcerp` does not return
the fraction of elementwise comparisons. Instead, it performs a paired t-test
on the two sample arrays.

For two uncertain values \(A\) and \(B\), the paired differences are

$$
d^{(j)} = a^{(j)} - b^{(j)}
$$

The t-test checks whether the mean difference is statistically distinguishable
from zero. The sign of the t-statistic determines the direction:

- `A < B` is true when the difference is significantly negative.
- `A > B` is true when the difference is significantly positive.
- If the p-value is not small enough, the comparison returns `False`.

Equality between two uncertain values checks whether the propagated difference
has zero reported moments. This is mainly useful for identifying expressions
that are algebraically identical over the same root samples, such as:

```python
Z * Z == Z**2
```

## Correlation Enforcement

### Target Correlation Matrix

By default, inputs are sampled to be as independent as practical. When real
inputs are correlated, `mcerp` can reorder existing samples to approximately
match a target correlation matrix.

For \(d\) inputs, the target matrix is

$$
R =
\begin{bmatrix}
1 & \rho_{12} & \cdots & \rho_{1d} \\
\rho_{21} & 1 & \cdots & \rho_{2d} \\
\vdots & \vdots & \ddots & \vdots \\
\rho_{d1} & \rho_{d2} & \cdots & 1
\end{bmatrix}
$$

The matrix must be symmetric and positive definite.

### Rank-Based Reordering

`mcerp.correlate(...)` preserves the marginal sample values of each input. It
does not draw new values from the distributions. Instead, it reorders existing
samples so their ranks induce the desired correlation structure.

The procedure is:

1. Rank each input sample column.
2. Convert ranks to normal scores using the standard normal inverse CDF.
3. Compute the Cholesky factors of the current and target correlation
   matrices.
4. Transform the score matrix toward the target correlation.
5. Convert the transformed scores back to ranks.
6. Reorder the original sampled values according to those ranks.

Because the original column values are reused, each input keeps its marginal
distribution while its relationship with the other inputs changes.

### Correlated Calculations

Correlations should be applied before derived calculations:

```python
import numpy as np
from mcerp import Exp, N, correlate

x1 = N(24, 1)
x2 = N(37, 4)
x3 = Exp(2)

R = np.array([
    [1.0, -0.75, 0.0],
    [-0.75, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])

correlate([x1, x2, x3], R)

Z = (x1 * x2**2) / (15 * (1.5 + x3))
```

The output distribution of \(Z\) may change substantially when correlations
are imposed, especially for products, ratios, and models where one input
amplifies another.

## Accuracy and Sample Count

### Number of Samples

The default sample count is:

```python
import mcerp

mcerp.npts
```

```text
10000
```

The count can be changed before variables are created:

```python
mcerp.npts = 50000
```

Larger \(N\) generally improves moment and probability estimates, but also
increases memory use and computation time. Values from \(1{,}000\) to
\(1{,}000{,}000\) are typical, depending on model cost and the accuracy
required.

### Consistent Sample Lengths

All uncertain values involved in one calculation should have the same sample
length. Existing variables keep their sample arrays, so change `mcerp.npts`
before creating input distributions.

### Sources of Error

MCERP estimates can differ from the exact output distribution because of:

- finite sample size,
- random LHS point placement and permutation,
- numerical behavior of the deterministic model,
- poor representation of rare tail events,
- invalid or misspecified input distributions,
- invalid target correlation matrices.

The method is often much more informative than a local linear approximation,
but it is still a simulation estimate. Treat very small probabilities and
extreme percentiles with appropriate sample-size caution.

## Relationship To Other Methods

`mcerp` is a sampling-based uncertainty propagation tool. It differs from:

- First-order propagation, which uses a local linear approximation.
- Second-order propagation, which uses a Taylor expansion and higher moments.
- Symbolic propagation, which tries to derive an exact analytical form.

The main advantage of MCERP is directness: define distributions, write the
calculation, and inspect the simulated output distribution. The main cost is
that results are approximate and require enough samples to stabilize the
statistics of interest.
