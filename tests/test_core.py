import sys
import types

import numpy as np
import pytest
import scipy.stats as ss

import mcerp
from mcerp import Chi2, Exp, Gamma, N, stats, umath
from mcerp.correlate import chol, induce_correlations, plotcorr
from mcerp.lhd import lhd


@pytest.fixture(autouse=True)
def small_sample_count(monkeypatch):
    monkeypatch.setattr(mcerp, "npts", 4)
    np.random.seed(12345)


def uncertain(values):
    return mcerp.UncertainFunction(np.array(values, dtype=float))


def assert_uncertain_points_close(value, expected):
    assert isinstance(value, mcerp.UncertainFunction)
    assert np.allclose(value._mcpts, expected)


def test_three_part_assembly() -> None:
    """Example of a three part assembly"""
    x1 = N(24, 1)
    x2 = N(37, 4)
    x3 = Exp(2)  # Exp(mu=0.5) is the same
    Z = (x1 * x2**2) / (15 * (1.5 + x3))
    Z.describe()


def test_volumetric_gas_flow() -> None:
    """Example of volumetric gas flow through orifice meter"""
    H = N(64, 0.5)
    M = N(16, 0.1)
    P = N(361, 2)
    t = N(165, 0.5)
    C = 38.4
    Q = C * umath.sqrt((520 * H * P) / (M * (t + 460)))
    Q.describe()


def test_manufacturing_tolerance_stackup() -> None:
    """Example of manufacturing tolerance stackup"""
    # for a gamma distribution we need the following conversions:
    # shape = mean**2/var
    # scale = var/mean
    mn = 1.5
    vr = 0.25
    k = mn**2 / vr
    theta = vr / mn
    x = Gamma(k, theta)
    y = Gamma(k, theta)
    z = Gamma(k, theta)
    w = x + y + z
    w.describe()


def test_scheduling() -> None:
    """Example of scheduling facilities (six stations)"""
    s1 = N(10, 1)
    s2 = N(20, 2**0.5)
    mn1 = 1.5
    vr1 = 0.25
    k1 = mn1**2 / vr1
    theta1 = vr1 / mn1
    s3 = Gamma(k1, theta1)
    mn2 = 10
    vr2 = 10
    k2 = mn2**2 / vr2
    theta2 = vr2 / mn2
    s4 = Gamma(k2, theta2)
    s5 = Exp(5)  # Exp(mu=0.2) is the same
    s6 = Chi2(10)
    T = s1 + s2 + s3 + s4 + s5 + s6
    T.describe()


def test_two_bar_truss_analysis() -> None:
    """Example of two-bar truss stress/deflection analysis"""
    H = N(30, 5 / 3.0, tag="H")
    B = N(60, 0.5 / 3.0, tag="B")
    d = N(3, 0.1 / 3, tag="d")
    t = N(0.15, 0.01 / 3, tag="t")
    E = N(30000, 1500 / 3.0, tag="E")
    rho = N(0.3, 0.01 / 3.0, tag="rho")
    P = N(66, 3 / 3.0, tag="P")
    pi = np.pi
    wght = 2 * pi * rho * d * t * umath.sqrt((B / 2) ** 2 + H**2)
    strs = (P * umath.sqrt((B / 2) ** 2 + H**2)) / (2 * pi * d * t * H)
    buck = (pi**2 * E * (d**2 + t**2)) / (8 * ((B / 2) ** 2 + H**2))
    defl = (P * ((B / 2) ** 2 + H**2) ** (1.5)) / (2 * pi * d * t * H**2 * E)
    print(wght.describe("wght"))
    print(strs.describe("strs"))
    print(buck.describe("buck"))
    print(defl.describe("defl"))


def test_to_uncertain_func_accepts_constants_and_rejects_unknowns() -> None:
    value = uncertain([1, 2, 3])

    assert mcerp.to_uncertain_func(value) is value
    assert np.allclose(mcerp.to_uncertain_func(3.5)._mcpts, [3.5] * mcerp.npts)

    with pytest.raises(mcerp.NotUpcast):
        mcerp.to_uncertain_func(object())


def test_uncertain_function_statistics_and_percentiles(capsys) -> None:
    value = uncertain([1, 2, 3, 4])
    value.tag = "sample"

    assert value.mean == pytest.approx(2.5)
    assert value.var == pytest.approx(1.25)
    assert value.std == pytest.approx(1.25**0.5)
    assert len(value.stats) == 4
    assert value.percentile(0) == 1
    assert value.percentile(1) == 4
    assert value.percentile(0.5) == pytest.approx(3.5)
    assert np.allclose(value.percentile(np.array([0, 1])), [1, 4])
    assert "uv(" in repr(value)

    value.describe()
    value.describe("named")
    printed = capsys.readouterr().out
    assert "MCERP Uncertain Value (sample)" in printed
    assert "MCERP Uncertain Value (named)" in printed


def test_uncertain_arithmetic_and_comparisons() -> None:
    value = uncertain([1, 2, 3, 4])
    same = uncertain([1, 2, 3, 4])

    assert_uncertain_points_close(value + 2, [3, 4, 5, 6])
    assert_uncertain_points_close(2 + value, [3, 4, 5, 6])
    assert_uncertain_points_close(value * 2, [2, 4, 6, 8])
    assert_uncertain_points_close(2 * value, [2, 4, 6, 8])
    assert_uncertain_points_close(value - 1, [0, 1, 2, 3])
    assert_uncertain_points_close(10 - value, [9, 8, 7, 6])
    assert_uncertain_points_close(value / 2, [0.5, 1, 1.5, 2])
    assert_uncertain_points_close(8 / value, [8, 4, 8 / 3, 2])
    assert_uncertain_points_close(value**2, [1, 4, 9, 16])
    assert_uncertain_points_close(2**value, [2, 4, 8, 16])
    assert_uncertain_points_close(-value, [-1, -2, -3, -4])
    assert_uncertain_points_close(+value, [1, 2, 3, 4])
    assert_uncertain_points_close(abs(uncertain([-1, 2])), [1, 2])

    assert value == same
    assert not (value != same)
    assert value == 2
    assert value != 2
    assert (value < 3) == pytest.approx(0.5)
    assert (value <= 3) == pytest.approx(0.75)
    assert (value > 3) == pytest.approx(0.25)
    assert (value >= 3) == pytest.approx(0.5)
    assert bool(uncertain([-1, 1]))


UNARY_CASES = [
    ("abs", np.abs, np.array([-1.5, 0.0, 2.5])),
    ("acos", np.arccos, np.array([-0.5, 0.0, 0.5])),
    ("acosh", np.arccosh, np.array([1.0, 2.0, 3.0])),
    ("asin", np.arcsin, np.array([-0.5, 0.0, 0.5])),
    ("asinh", np.arcsinh, np.array([-1.0, 0.0, 1.0])),
    ("atan", np.arctan, np.array([-1.0, 0.0, 1.0])),
    ("atanh", np.arctanh, np.array([-0.5, 0.0, 0.5])),
    ("ceil", np.ceil, np.array([-1.2, 0.1, 1.8])),
    ("cos", np.cos, np.array([-1.0, 0.0, 1.0])),
    ("cosh", np.cosh, np.array([-1.0, 0.0, 1.0])),
    ("degrees", np.degrees, np.array([0.0, np.pi / 2, np.pi])),
    ("exp", np.exp, np.array([-1.0, 0.0, 1.0])),
    ("expm1", np.expm1, np.array([-1.0, 0.0, 1.0])),
    ("fabs", np.fabs, np.array([-1.5, 0.0, 2.5])),
    ("floor", np.floor, np.array([-1.2, 0.1, 1.8])),
    ("ln", np.log, np.array([1.0, 2.0, 4.0])),
    ("log", np.log, np.array([1.0, 2.0, 4.0])),
    ("log10", np.log10, np.array([1.0, 10.0, 100.0])),
    ("log1p", np.log1p, np.array([0.0, 1.0, 3.0])),
    ("radians", np.radians, np.array([0.0, 90.0, 180.0])),
    ("sin", np.sin, np.array([-1.0, 0.0, 1.0])),
    ("sinh", np.sinh, np.array([-1.0, 0.0, 1.0])),
    ("sqrt", np.sqrt, np.array([1.0, 4.0, 9.0])),
    ("tan", np.tan, np.array([-1.0, 0.0, 1.0])),
    ("tanh", np.tanh, np.array([-1.0, 0.0, 1.0])),
    ("trunc", np.trunc, np.array([-1.8, 0.1, 1.8])),
]


@pytest.mark.parametrize(("name", "numpy_func", "values"), UNARY_CASES)
def test_umath_scalar_and_uncertain_inputs(name, numpy_func, values) -> None:
    func = getattr(umath, name)

    assert np.allclose(func(values), numpy_func(values))
    assert_uncertain_points_close(func(uncertain(values)), numpy_func(values))


def test_umath_hypot_scalar_and_uncertain_inputs() -> None:
    assert umath.hypot(3, 4) == 5
    assert_uncertain_points_close(
        umath.hypot(uncertain([3, 5, 8, 15]), 4), np.hypot([3, 5, 8, 15], 4)
    )


VALID_CONSTRUCTORS = [
    (mcerp.Beta, (2, 5)),
    (mcerp.Bradford, (2,)),
    (mcerp.Burr, (2, 3)),
    (mcerp.ChiSquared, (3,)),
    (mcerp.Erf, (2,)),
    (mcerp.Erlang, (2, 3)),
    (mcerp.Exponential, (2,)),
    (mcerp.Fisher, (3, 4)),
    (mcerp.Gamma, (2, 3)),
    (mcerp.LogNormal, (0, 0.5)),
    (mcerp.Normal, (0, 1)),
    (mcerp.Pareto, (2, 1)),
    (mcerp.Pareto2, (2, 1)),
    (mcerp.PERT, (0, 0.5, 1)),
    (mcerp.StudentT, (3,)),
    (mcerp.Triangular, (0, 0.5, 1)),
    (mcerp.Uniform, (0, 1)),
    (mcerp.Weibull, (1, 2)),
    (mcerp.Bernoulli, (0.5,)),
    (mcerp.Binomial, (5, 0.5)),
    (mcerp.G, (0.5,)),
    (mcerp.Hypergeometric, (20, 5, 4)),
    (mcerp.Poisson, (2,)),
]


@pytest.mark.parametrize(("constructor", "args"), VALID_CONSTRUCTORS)
def test_distribution_constructors_return_uncertain_values(constructor, args):
    value = constructor(*args, tag="x")

    assert isinstance(value, mcerp.UncertainFunction)
    assert len(value._mcpts) == mcerp.npts


@pytest.mark.parametrize(
    ("constructor", "args"),
    [
        (mcerp.Beta, (0, 1)),
        (mcerp.Beta, (1, 1, 2, 1)),
        (mcerp.Bradford, (0,)),
        (mcerp.Burr, (0, 1)),
        (mcerp.ChiSquared, (0,)),
        (mcerp.Erlang, (1.5, 1)),
        (mcerp.Exponential, (0,)),
        (mcerp.Fisher, (0, 1)),
        (mcerp.Gamma, (0, 1)),
        (mcerp.LogNormal, (0, 0)),
        (mcerp.Normal, (0, 0)),
        (mcerp.Pareto, (0, 1)),
        (mcerp.Pareto2, (1, 0)),
        (mcerp.PERT, (0, 2, 1)),
        (mcerp.StudentT, (0,)),
        (mcerp.Triangular, (0, 2, 1)),
        (mcerp.Uniform, (1, 1)),
        (mcerp.Weibull, (0, 1)),
        (mcerp.Bernoulli, (0,)),
        (mcerp.Binomial, (0, 0.5)),
        (mcerp.G, (1,)),
        (mcerp.Hypergeometric, (0, 1, 1)),
        (mcerp.Poisson, (0,)),
    ],
)
def test_distribution_constructor_validation_paths(constructor, args):
    with pytest.raises((AssertionError, ValueError)):
        constructor(*args)


@pytest.mark.parametrize(
    "call", [lambda: mcerp.ExtValueMax(0, 1), lambda: mcerp.ExtValueMin(0, 1)]
)
def test_special_constructors_return_uncertain_values(call) -> None:
    value = call()

    assert isinstance(value, mcerp.UncertainFunction)


def test_beta_prime_current_error_path_is_covered() -> None:
    with pytest.raises(TypeError):
        mcerp.BetaPrime(2, 3)


def test_covariance_matrix_and_stats_wrappers() -> None:
    first = uncertain([1, 2, 3, 4])
    second = uncertain([2, 4, 6, 8])

    cov = mcerp.covariance_matrix([first, second])
    assert np.asarray(cov).shape == (2, 2)
    assert cov[1][0] == pytest.approx(cov[0][1])
    assert stats.tmean(first) == pytest.approx(2.5)
    assert stats.tvar(first) == pytest.approx(np.var([1, 2, 3, 4], ddof=1))
    assert stats.scoreatpercentile(first, 50) == pytest.approx(2.5)
    assert stats.pearsonr(first, second).statistic == pytest.approx(1.0)
    assert stats.rankdata(first).tolist() == [1, 2, 3, 4]
    assert stats.wrap(np.mean)(first) == pytest.approx(2.5)


def test_lhd_error_and_spacefilling_paths(capsys) -> None:
    dist = ss.uniform(0, 1)

    assert lhd() is None
    with pytest.raises((AssertionError, ValueError)):
        lhd(dist=dist, size=2, dims=0)
    with pytest.raises(NotImplementedError):
        lhd(dist=dist, size=2, form="orthogonal")
    with pytest.raises(ValueError):
        lhd(dist=dist, size=2, form="bad")

    sample = lhd(
        dist=dist,
        size=6,
        dims=2,
        form="spacefilling",
        iterations=2,
        showcorrelations=True,
    )
    assert sample.shape == (6, 2)
    assert "Optimized Distance:" in capsys.readouterr().out


def test_correlation_helpers_and_plotcorr(monkeypatch) -> None:
    assert np.allclose(
        chol([[4, 2], [2, 3]]), np.linalg.cholesky([[4, 2], [2, 3]])
    )
    with pytest.raises((AssertionError, ValueError)):
        chol(np.ones((2, 3)))

    data = np.column_stack((np.arange(1, 9), np.arange(8, 0, -1)))
    corrmat = np.array([[1.0, 0.5], [0.5, 1.0]])
    adjusted = induce_correlations(data.copy(), corrmat)
    assert adjusted.shape == data.shape

    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.subplots = fake_subplots
    monkeypatch.setattr(sys.modules["mcerp.correlate"], "plt", pyplot)
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.__path__ = []
    matplotlib.pyplot = pyplot
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot

    fig = plotcorr(
        [uncertain(np.arange(5)), uncertain(np.arange(5, 10))],
        labels=["a", "b"],
    )
    assert fig.axes

    fig = plotcorr(
        np.vstack([np.arange(5), np.arange(5, 10), np.arange(10, 15)]),
        plotargs="r.",
        full=False,
        labels=["a", "b", "c"],
    )
    assert fig.axes


class FakeAxisSide:
    def set_visible(self, visible: object) -> None:
        self.visible = visible

    def set_ticks_position(self, position: str) -> None:
        self.position = position


class FakeSubplotSpec:
    def __init__(self, row: int, col: int, size: int) -> None:
        self.row = row
        self.col = col
        self.size = size

    def is_first_col(self) -> bool:
        return self.col == 0

    def is_last_col(self) -> bool:
        return self.col == self.size - 1

    def is_first_row(self) -> bool:
        return self.row == 0

    def is_last_row(self) -> bool:
        return self.row == self.size - 1


class FakeAxis:
    def __init__(self, row: int, col: int, size: int) -> None:
        self.xaxis = FakeAxisSide()
        self.yaxis = FakeAxisSide()
        self.spec = FakeSubplotSpec(row, col, size)
        self.xlim = (0, 1)
        self.ylim = (0, 1)

    def get_subplotspec(self) -> FakeSubplotSpec:
        return self.spec

    def annotate(self, *_args: object, **_kwargs: object) -> None:
        return None

    def plot(self, *_args: object, **_kwargs: object) -> None:
        return None

    def set_ylim(self, low: object, high: object = None) -> None:
        self.ylim = low if high is None else (low, high)

    def set_xlim(self, low: object, high: object = None) -> None:
        self.xlim = low if high is None else (low, high)

    def get_xlim(self) -> object:
        return self.xlim

    def get_ylim(self) -> object:
        return self.ylim


class FakeFigure:
    def __init__(self, axes: np.ndarray) -> None:
        self.axes = list(axes.flat)

    def subplots_adjust(self, **_kwargs: object) -> None:
        return None

    def delaxes(self, axis: FakeAxis) -> None:
        self.axes.remove(axis)


def fake_subplots(
    nrows: int, ncols: int, figsize: tuple[int, int]
) -> tuple[FakeFigure, np.ndarray]:
    axes = np.empty((nrows, ncols), dtype=object)
    for row in range(nrows):
        for col in range(ncols):
            axes[row, col] = FakeAxis(row, col, nrows)
    return FakeFigure(axes), axes
