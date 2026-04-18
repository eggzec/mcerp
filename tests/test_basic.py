import numpy as np

from mcerp import Exp, N, correlate, correlation_matrix


def test_basic_calculations() -> None:
    """Test basic calculations"""
    x1 = N(24, 1)
    x2 = N(37, 4)
    x3 = Exp(2)

    # Check that stats are reasonable
    assert abs(x1.mean - 24) < 0.1
    assert abs(x1.std - 1) < 0.1
    assert x3.mean > 0  # Exp(2) has mean 0.5

    Z = (x1 * x2**2) / (15 * (1.5 + x3))

    # Check that Z has reasonable mean
    assert Z.mean > 1000  # From the output, around 1161
    assert Z.mean < 1300


def test_correlation_matrix() -> None:
    """Test correlation matrix calculation"""
    x1 = N(24, 1)
    x2 = N(37, 4)
    x3 = Exp(2)

    corr = correlation_matrix([x1, x2, x3])

    # Should be 3x3 matrix
    assert corr.shape == (3, 3)
    # Diagonal should be 1
    assert np.allclose(np.diag(corr), 1.0)


def test_correlate_function() -> None:
    """Test applying correlation"""
    x1 = N(24, 1)
    x2 = N(37, 4)
    x3 = Exp(2)

    c = np.array([[1.0, -0.75, 0.0], [-0.75, 1.0, 0.0], [0.0, 0.0, 1.0]])

    correlate([x1, x2, x3], c)

    corr_after = correlation_matrix([x1, x2, x3])

    # Check that correlation between x1 and x2 is approximately -0.75
    assert abs(corr_after[0, 1] + 0.75) < 0.1
    assert abs(corr_after[1, 0] + 0.75) < 0.1


def test_comparison_operations() -> None:
    """Test comparison operations on uncertain values"""
    x1 = N(24, 1)
    Z = (x1 * N(37, 4) ** 2) / (15 * (1.5 + Exp(2)))

    # These return uncertain values representing probabilities
    prob1 = x1 < 15
    prob2 = Z >= 1000

    # Should be between 0 and 1
    assert 0 <= prob1 <= 1
    assert 0 <= prob2 <= 1
