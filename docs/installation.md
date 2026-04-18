# Installation

`mcerp` supports Python 3.10 or later.

## pip

Install the core package:

```bash
pip install mcerp
```

Install with plotting support:

```bash
pip install "mcerp[plot]"
```

Install all optional dependencies:

```bash
pip install "mcerp[all]"
```

## uv

Add `mcerp` to a project:

```bash
uv add mcerp
uv sync
```

Install into an existing environment:

```bash
uv pip install mcerp
```

## From Git

Install the latest source from GitHub:

```bash
pip install --upgrade "git+https://github.com/eggzec/mcerp.git#egg=mcerp"
```

## Requirements

- Python 3.10 or later
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/) for plotting

NumPy and SciPy are installed automatically by package installers. Matplotlib
is optional and only needed for plotting distributions or correlation plots.
