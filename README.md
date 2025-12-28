# hopkins-statistic

[![CI](https://github.com/jo-phil/hopkins-statistic/actions/workflows/ci.yml/badge.svg)](https://github.com/jo-phil/hopkins-statistic/actions/workflows/ci.yml)
[![Docs](https://github.com/jo-phil/hopkins-statistic/actions/workflows/docs.yml/badge.svg)](https://jo-phil.github.io/hopkins-statistic/)

A Python package for computing the Hopkins statistic to test for departure from
complete spatial randomness (CSR), often used to assess clustering tendency.

## Installation

```bash
pip install hopkins-statistic
```

## Usage

```python
import numpy as np
from hopkins_statistic import hopkins

rng = np.random.default_rng(42)

# Clustered test data
centers = np.array([[0, 0], [0, 1]])
labels = rng.integers(len(centers), size=100)
X = centers[labels] + rng.normal(scale=0.1, size=(100, 2))

# For strongly clustered data, the statistic is often > 0.7
H = hopkins(X, rng=rng)
print(f"{H:.2f}")
#> 0.77
```

## License

MIT.
See [LICENSE](https://github.com/jo-phil/hopkins-statistic/blob/main/LICENSE).
