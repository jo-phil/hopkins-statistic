import math
import numbers
import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree


class HopkinsUndefinedWarning(RuntimeWarning):
    """Warning issued when the Hopkins statistic is undefined."""


def hopkins(
    X: ArrayLike,
    *,
    m: int | float = 0.1,
    power: int | float | None = None,
    rng: np.random.Generator | int | None = None,
) -> float:
    """Compute the Hopkins statistic.

    The Hopkins statistic is a test statistic for the null hypothesis of
    complete spatial randomness (CSR), i.e., that points are independently
    and uniformly distributed within the sampling frame. Under this null,
    the expected value of the statistic is 0.5. Larger values indicate
    more clustering than expected under CSR, while smaller values indicate
    more regular spacing.

    Args:
        X: Array-like of shape `(n, d)`, with `n >= 3` observations
            in `d >= 1` dimensions. All values must be finite.
        m: Sample size, or its fraction of `n`.
            If int, this must satisfy `1 <= m <= n`.
            If float, this must satisfy `0 < m <= 1`,
            and the sample size is `ceil(m * n)`.
        power: Exponent applied to Euclidean distances. Defaults to `d`.
            Must be positive and finite.
        rng: Random number generator or seed passed to `np.random.default_rng`.
            Specify for repeatable behavior.

    Returns:
        The Hopkins statistic, a number between 0 and 1 (or NaN if undefined).

    Warns:
        HopkinsUndefinedWarning: If all observations in X are identical.

    Notes:
        * The sampling frame is taken as the axis-aligned bounding box
          of `X`; results should be interpreted accordingly.
        * Euclidean distances on non-spatial data often benefit from
          feature scaling.
        * The sample size `m` is typically chosen between 10 and one
          tenth of `n` to balance variance and the approximations used
          for the null distribution of the statistic.
        * The exponent `power` defaults to `d`, the number of columns
          in `X`; other values alter the null distribution.

    Examples:
        >>> import numpy as np
        >>> from hopkins_statistic import hopkins
        >>> rng = np.random.default_rng(42)
        >>> X = rng.uniform(size=(100, 2))
        >>> hopkins(X, rng=rng)  # near half under CSR
        0.51...

        >>> centers = np.array([[0, 0], [5, 5]])
        >>> labels = rng.integers(2, size=100)
        >>> X = centers[labels] + rng.normal(size=(100, 2))
        >>> hopkins(X, rng=rng)  # larger for clustered data
        0.86...

    """
    X = np.asarray(X, dtype=float)

    n, d = _validate_shape(X)
    m = _parse_m(m, n)
    power = _parse_power(power, d)
    rng = np.random.default_rng(rng)

    if not np.isfinite(X).all():
        msg = "X must contain only finite values; found NaN or inf."
        raise ValueError(msg)

    lower, upper = X.min(axis=0), X.max(axis=0)
    if np.all(lower == upper):
        msg = "All observations in X are identical."
        warnings.warn(msg, HopkinsUndefinedWarning, stacklevel=2)
        return math.nan

    null_sample = rng.uniform(lower, upper, size=(m, d))
    data_sample = X[rng.choice(n, size=m, replace=False)]

    tree = KDTree(X)
    u = tree.query(null_sample, k=1)[0]
    w = np.asarray(tree.query(data_sample, k=2)[0])[:, 1]  # 1st NN is itself

    u_sum = np.sum(u**power)
    w_sum = np.sum(w**power)

    return float(u_sum / (u_sum + w_sum))


def _validate_shape(X: np.ndarray) -> tuple[int, int]:
    if X.ndim != 2:
        msg = f"X must be a 2D array of shape (n, d); got shape {X.shape}."
        raise ValueError(msg)

    n, d = X.shape
    if n < 3:
        msg = f"X must contain at least 3 observations; got n={n}."
        raise ValueError(msg)
    if d < 1:
        msg = "X must have at least 1 feature (d >= 1); got d=0."
        raise ValueError(msg)

    return n, d


def _parse_m(m: int | float, n: int) -> int:
    if isinstance(m, numbers.Integral) and not isinstance(m, bool):
        if not 1 <= m <= n:
            msg = f"m must satisfy 1 <= m <= n; got m={m}, n={n}."
            raise ValueError(msg)
        return int(m)

    if isinstance(m, numbers.Real) and not isinstance(m, bool):
        if not 0 < m <= 1:
            msg = f"If m is a float, it must satisfy 0 < m <= 1; got m={m}."
            raise ValueError(msg)
        return math.ceil(m * n)

    msg = f"m must be int or float; got {type(m).__name__}."
    raise TypeError(msg)


def _parse_power(power: int | float | None, d: int) -> int | float:
    if power is None:
        return d

    if not isinstance(power, numbers.Real) or isinstance(power, bool):
        msg = f"power must be a real number; got {type(power).__name__}."
        raise TypeError(msg)

    if not math.isfinite(power):
        msg = f"power must be finite; got power={power}."
        raise ValueError(msg)
    if power <= 0:
        msg = f"power must be positive; got power={power}."
        raise ValueError(msg)

    return power
