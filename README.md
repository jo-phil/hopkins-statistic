# hopkins-statistic

[![CI](https://github.com/jo-phil/hopkins-statistic/actions/workflows/ci.yml/badge.svg)](https://github.com/jo-phil/hopkins-statistic/actions/workflows/ci.yml "View workflow runs")
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/jo-phil/dd436ee381602e21922433dfcdd0b9ec/raw/f3d7e554e7fc565a44b7e7781e6804253f965cd2/coverage.json)](https://github.com/jo-phil/hopkins-statistic/actions/workflows/ci.yml?query=branch%3Amain+event%3Apush "View coverage report")
[![PyPI - Version](https://img.shields.io/pypi/v/hopkins-statistic)](https://pypi.org/project/hopkins-statistic/ "View latest release on PyPI")
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hopkins-statistic)](https://pypi.org/project/hopkins-statistic/ "View supported Python versions on PyPI")
[![Docs](https://github.com/jo-phil/hopkins-statistic/actions/workflows/docs.yml/badge.svg)](https://jo-phil.github.io/hopkins-statistic/ "Read the documentation")
[![DOI](https://zenodo.org/badge/1108713722.svg)](https://doi.org/10.5281/zenodo.18172791 "View on Zenodo")

hopkins-statistic is a library for computing the Hopkins statistic to assess
clustering tendency (also known as cluster tendency or clusterability) by
testing for departure from complete spatial randomness (CSR) in point patterns.

This implementation defaults to the formulation of Cross and Jain (1982),
raising distances to the power of the data dimension. In two dimensions this is
equivalent to the original definition by Hopkins and Skellam (1954) and, under
the CSR null hypothesis, the statistic has a Beta distribution, so *p*-values
can be computed analytically.

## Installation

```bash
pip install hopkins-statistic
```

## Usage

```python
import numpy as np
from hopkins_statistic import hopkins

rng = np.random.default_rng(42)

# Simple clustered example: two Gaussian blobs
centers = np.array([[0, 0], [0, 1]])
labels = rng.integers(len(centers), size=100)
X = centers[labels] + rng.normal(scale=0.1, size=(100, 2))

statistic = hopkins(X, rng=rng)
print(f"{statistic:.3f}")
#> 0.771
```

## License

MIT.
See [LICENSE](https://github.com/jo-phil/hopkins-statistic/blob/main/LICENSE).
