# ``mcerp`` Real-time latin-hypercube sampling-based Monte Carlo ERror Propagation for Python

[![Tests](https://github.com/eggzec/mcerp/actions/workflows/code_test.yml/badge.svg)](https://github.com/eggzec/mcerp/actions/workflows/code_test.yml)
[![Documentation](https://github.com/eggzec/mcerp/actions/workflows/docs_build.yml/badge.svg)](https://github.com/eggzec/mcerp/actions/workflows/docs_build.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![codecov](https://codecov.io/gh/eggzec/mcerp/branch/master/graph/badge.svg)](https://codecov.io/gh/eggzec/mcerp)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=eggzec_mcerp&metric=alert_status)](https://sonarcloud.io/project/overview?id=eggzec_mcerp)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](./LICENSE)

[![PyPI Downloads](https://img.shields.io/pypi/dm/mcerp.svg?label=PyPI%20downloads)](https://pypi.org/project/mcerp/)
[![Python versions](https://img.shields.io/pypi/pyversions/mcerp.svg)](https://pypi.org/project/mcerp/)

## Overview

``mcerp`` is a stochastic calculator for `Monte Carlo methods` that uses 
`latin-hypercube sampling` to perform non-order specific 
`error propagation` (or uncertainty analysis). 

With this package you can **easily** and **transparently** track the effects
of uncertainty through mathematical calculations. Advanced mathematical 
functions, similar to those in the standard `math` module, and statistical
functions like those in the `scipy.stats` module, can also be evaluated 
directly.

If you are familiar with Excel-based risk analysis programs like *@Risk*, 
*Crystal Ball*, *ModelRisk*, etc., this package **will work wonders** for you
(and probably even be faster!) and give you more modelling flexibility with 
the powerful Python language. This package also *doesn't cost a penny*, 
compared to those commercial packages which cost *thousands of dollars* for a 
single-seat license. Feel free to copy and redistribute this package as much 
as you desire!

1. **Transparent calculations**. **No or little modification** to existing 
   code required.
    
2. Basic `NumPy` support without modification. (I haven't done extensive 
   testing, so please let me know if you encounter bugs.)

3. Advanced mathematical functions supported through the ``mcerp.umath`` 
   sub-module. If you think a function is in there, it probably is. If it 
   isn't, please request it!

4. **Easy statistical distribution constructors**. The location, scale, 
   and shape parameters follow the notation in the respective Wikipedia 
   articles and other relevant web pages.

5. **Correlation enforcement** and variable sample visualization capabilities.

6. **Probability calculations** using conventional comparison operators.

7. Advanced Scipy **statistical function compatibility** with package 
   functions. Depending on your version of Scipy, some functions might not
   work.

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
- [soerp](http://pypi.python.org/pypi/soerp) : Second Order ERror Propagation.
