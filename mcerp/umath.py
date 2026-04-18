"""
================================================================================
mcerp: Real-time latin-hypercube-sampling-based Monte Carlo Error Propagation
================================================================================

Generalizes mathematical operators that work on numeric objects (from the math
module or numpy) compatible with objects with uncertainty distributions

Author: Abraham Lee
Copyright: 2013
"""

from __future__ import annotations

import numpy as np

from .core import UncertainFunction, to_uncertain_func


__author__ = "Abraham Lee"


def _abs(x: object) -> object:
    """
    Absolute value


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.abs(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.abs(x)


def acos(x: object) -> object:
    """
    Inverse cosine


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arccos(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arccos(x)


def acosh(x: object) -> object:
    """
    Inverse hyperbolic cosine


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arccosh(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arccosh(x)


def asin(x: object) -> object:
    """
    Inverse sine


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arcsin(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arcsin(x)


def asinh(x: object) -> object:
    """
    Inverse hyperbolic sine


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arcsinh(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arcsinh(x)


def atan(x: object) -> object:
    """
    Inverse tangent


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arctan(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arctan(x)


def atanh(x: object) -> object:
    """
    Inverse hyperbolic tangent


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arctanh(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arctanh(x)


def ceil(x: object) -> object:
    """
    Ceiling function (round towards positive infinity)


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.ceil(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.ceil(x)


def cos(x: object) -> object:
    """
    Cosine


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.cos(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.cos(x)


def cosh(x: object) -> object:
    """
    Hyperbolic cosine


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.cosh(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.cosh(x)


def degrees(x: object) -> object:
    """
    Convert radians to degrees


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.degrees(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.degrees(x)


def exp(x: object) -> object:
    """
    Exponential function


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.exp(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.exp(x)


def expm1(x: object) -> object:
    """
    Calculate exp(x) - 1


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.expm1(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.expm1(x)


def fabs(x: object) -> object:
    """
    Absolute value function


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.fabs(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.fabs(x)


def floor(x: object) -> object:
    """
    Floor function (round towards negative infinity)


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.floor(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.floor(x)


def hypot(x: object, y: object) -> object:
    """
    Calculate the hypotenuse given two "legs" of a right triangle


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction) or isinstance(y, UncertainFunction):
        ufx = to_uncertain_func(x)
        ufy = to_uncertain_func(y)
        mcpts = np.hypot(ufx._mcpts, ufy._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.hypot(x, y)


def ln(x: object) -> object:
    """
    Natural logarithm (same as "log(x)")


    Returns
    -------
    object
        The transformed value.
    """
    return log(x)


def log(x: object) -> object:
    """
    Natural logarithm


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.log(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.log(x)


def log10(x: object) -> object:
    """
    Base-10 logarithm


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.log10(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.log10(x)


def log1p(x: object) -> object:
    """
    Natural logarithm of (1 + x)


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.log1p(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.log1p(x)


def radians(x: object) -> object:
    """
    Convert degrees to radians


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.radians(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.radians(x)


def sin(x: object) -> object:
    """
    Sine


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.sin(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.sin(x)


def sinh(x: object) -> object:
    """
    Hyperbolic sine


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.sinh(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.sinh(x)


def sqrt(x: object) -> object:
    """
    Square-root function


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.sqrt(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.sqrt(x)


def tan(x: object) -> object:
    """
    Tangent


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.tan(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.tan(x)


def tanh(x: object) -> object:
    """
    Hyperbolic tangent


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.tanh(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.tanh(x)


def trunc(x: object) -> object:
    """
    Truncate the values to the integer value without rounding


    Returns
    -------
    object
        The transformed value.
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.trunc(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.trunc(x)


globals()["abs"] = _abs
globals()["abs"].__name__ = "abs"
globals()["abs"].__qualname__ = "abs"
__all__ = [
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "ceil",
    "cos",
    "cosh",
    "degrees",
    "exp",
    "expm1",
    "fabs",
    "floor",
    "hypot",
    "ln",
    "log",
    "log1p",
    "log10",
    "radians",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
]
