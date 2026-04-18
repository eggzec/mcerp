import scipy.stats as ss

from mcerp.lhd import lhd


def test_latin_hypercube() -> None:
    # test single distribution
    d0 = ss.uniform(loc=-1, scale=2)  # uniform distribution,low=-1, width=2
    print(lhd(dist=d0, size=5))

    # test single distribution for multiple variables
    d1 = ss.norm(loc=0, scale=1)  # normal distribution, mean=0, stdev=1
    print(lhd(dist=d1, size=7, dims=5))

    # test multiple distributions
    d2 = ss.beta(2, 5)  # beta distribution, alpha=2, beta=5
    d3 = ss.expon(scale=1 / 1.5)  # exponential distribution, lambda=1.5
    print(lhd(dist=(d1, d2, d3), size=6))

    rand_lhs = lhd(dist=(d0, d1, d2, d3), size=100)
    spac_lhs = lhd(
        dist=(d0, d1, d2, d3),
        size=100,
        form="spacefilling",
        iterations=100,
        showcorrelations=True,
    )

    # Basic assertions to verify the output shapes
    assert rand_lhs.shape == (100, 4)
    assert spac_lhs.shape == (100, 4)
