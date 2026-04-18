"""
================================================================================
mcerp: Real-time latin-hypercube-sampling-based Monte Carlo Error Propagation
================================================================================

Authors: Abraham Lee
         Saud Zahir

Copyright (c) 2013, Abraham D. Lee

"""

from importlib.metadata import PackageNotFoundError, version

from . import stats, umath
from .core import (
    CONSTANT_TYPES,
    PERT,
    B,
    Bern,
    Bernoulli,
    Beta,
    BetaPrime,
    Binomial,
    Bradford,
    Burr,
    Chi2,
    ChiSquared,
    Erf,
    Erlang,
    EVMax,
    EVMin,
    Exp,
    Exponential,
    ExtValueMax,
    ExtValueMin,
    F,
    Fisher,
    G,
    Gamma,
    H,
    Hypergeometric,
    LogN,
    LogNormal,
    N,
    Normal,
    NotUpcast,
    NotUpcastError,
    Pareto,
    Pareto2,
    Pois,
    Poisson,
    StudentT,
    T,
    Tri,
    Triangular,
    U,
    UncertainFunction,
    UncertainVariable,
    Uniform,
    Weib,
    Weibull,
    correlation_matrix,
    covariance_matrix,
    npts,
    to_uncertain_func,
    uv,
)
from .correlate import chol, correlate, induce_correlations, plotcorr


__author__ = "Abraham Lee"

try:  # noqa RUF067
    __version__ = version("mcerp")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "CONSTANT_TYPES",
    "PERT",
    "B",
    "Bern",
    "Bernoulli",
    "Beta",
    "BetaPrime",
    "Binomial",
    "Bradford",
    "Burr",
    "Chi2",
    "ChiSquared",
    "EVMax",
    "EVMin",
    "Erf",
    "Erlang",
    "Exp",
    "Exponential",
    "ExtValueMax",
    "ExtValueMin",
    "F",
    "Fisher",
    "G",
    "Gamma",
    "H",
    "Hypergeometric",
    "LogN",
    "LogNormal",
    "N",
    "Normal",
    "NotUpcast",
    "NotUpcastError",
    "Pareto",
    "Pareto2",
    "Pois",
    "Poisson",
    "StudentT",
    "T",
    "Tri",
    "Triangular",
    "U",
    "UncertainFunction",
    "UncertainVariable",
    "Uniform",
    "Weib",
    "Weibull",
    "chol",
    "correlate",
    "correlation_matrix",
    "covariance_matrix",
    "induce_correlations",
    "npts",
    "plotcorr",
    "stats",
    "to_uncertain_func",
    "umath",
    "uv",
]
