import numpy as np

from mcerp import Chi2, Exp, Gamma, N, umath


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
