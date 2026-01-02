# hopkins-statistic

[![CI](https://github.com/jo-phil/hopkins-statistic/actions/workflows/ci.yml/badge.svg)](https://github.com/jo-phil/hopkins-statistic/actions/workflows/ci.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/hopkins-statistic)](https://pypi.org/project/hopkins-statistic/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hopkins-statistic)](https://pypi.org/project/hopkins-statistic/)
[![Docs](https://github.com/jo-phil/hopkins-statistic/actions/workflows/docs.yml/badge.svg)](https://jo-phil.github.io/hopkins-statistic/)

hopkins-statistic is a library for computing the Hopkins statistic to
test for departure from complete spatial randomness (CSR), i.e., the
presence of clustering or regularity in point patterns.

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
