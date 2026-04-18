"""
================================================================================
mcerp: Real-time latin-hypercube-sampling-based Monte Carlo Error Propagation
================================================================================

Generalizes mathematical operators that work on numeric objects (from the math
module or numpy) compatible with objects with uncertainty distributions

Author: Abraham Lee
Copyright: 2013
"""

import numpy as np

from mcerp import UncertainFunction, to_uncertain_func


__author__ = "Abraham Lee"


def abs(x):
    """
    Absolute value
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.abs(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.abs(x)


def acos(x):
    """
    Inverse cosine
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arccos(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arccos(x)


def acosh(x):
    """
    Inverse hyperbolic cosine
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arccosh(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arccosh(x)


def asin(x):
    """
    Inverse sine
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arcsin(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arcsin(x)


def asinh(x):
    """
    Inverse hyperbolic sine
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arcsinh(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arcsinh(x)


def atan(x):
    """
    Inverse tangent
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arctan(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arctan(x)


def atanh(x):
    """
    Inverse hyperbolic tangent
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arctanh(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arctanh(x)


def ceil(x):
    """
    Ceiling function (round towards positive infinity)
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.ceil(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.ceil(x)


def cos(x):
    """
    Cosine
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.cos(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.cos(x)


def cosh(x):
    """
    Hyperbolic cosine
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.cosh(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.cosh(x)


def degrees(x):
    """
    Convert radians to degrees
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.degrees(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.degrees(x)


def exp(x):
    """
    Exponential function
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.exp(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.exp(x)


def expm1(x):
    """
    Calculate exp(x) - 1
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.expm1(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.expm1(x)


def fabs(x):
    """
    Absolute value function
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.fabs(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.fabs(x)


def floor(x):
    """
    Floor function (round towards negative infinity)
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.floor(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.floor(x)


def hypot(x, y):
    """
    Calculate the hypotenuse given two "legs" of a right triangle
    """
    if isinstance(x, UncertainFunction) or isinstance(x, UncertainFunction):
        ufx = to_uncertain_func(x)
        ufy = to_uncertain_func(y)
        mcpts = np.hypot(ufx._mcpts, ufy._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.hypot(x, y)


def ln(x):
    """
    Natural logarithm (same as "log(x)")
    """
    return log(x)


def log(x):
    """
    Natural logarithm
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.log(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.log(x)


def log10(x):
    """
    Base-10 logarithm
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.log10(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.log10(x)


def log1p(x):
    """
    Natural logarithm of (1 + x)
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.log1p(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.log1p(x)


def radians(x):
    """
    Convert degrees to radians
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.radians(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.radians(x)


def sin(x):
    """
    Sine
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.sin(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.sin(x)


def sinh(x):
    """
    Hyperbolic sine
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.sinh(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.sinh(x)


def sqrt(x):
    """
    Square-root function
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.sqrt(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.sqrt(x)


def tan(x):
    """
    Tangent
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.tan(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.tan(x)


def tanh(x):
    """
    Hyperbolic tangent
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.tanh(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.tanh(x)


def trunc(x):
    """
    Truncate the values to the integer value without rounding
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.trunc(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.trunc(x)
