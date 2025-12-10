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

    Args:
        X: Array of shape `(n, d)`, with `n` observations
            in `d` dimensions and `n >= 3`.
        m: Sample size, or its fraction of `n`.
            If int, this must satisfy `1 <= m <= n`.
            If float, this must satisfy `0 < m <= 1`,
            and the sample size is `ceil(m * n)`.
        power: Exponent applied to distances. Defaults to `d`.
            Must be positive and finite.
        rng: Random number generator or seed that will be passed to
            `np.random.default_rng`. Specify for repeatable behavior.

    Returns:
        The Hopkins statistic, a number between 0 and 1 (or NaN if undefined).

    Warns:
        HopkinsUndefinedWarning: If all observations in X are identical.

    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        msg = f"X must be a 2D array of shape (n, d); got shape {X.shape}."
        raise ValueError(msg)

    n, d = X.shape
    if n < 3:
        msg = f"X must contain at least 3 observations; got n={n}."
        raise ValueError(msg)

    if isinstance(m, numbers.Integral):
        if not 1 <= m <= n:
            msg = f"m must satisfy 1 <= m <= n; got m={m}, n={n}."
            raise ValueError(msg)
        m = int(m)
    elif isinstance(m, numbers.Real):
        if not 0 < m <= 1:
            msg = f"If m is a float, it must satisfy 0 < m <= 1; got m={m}."
            raise ValueError(msg)
        m = math.ceil(m * n)
    else:
        msg = f"m must be int or float; got {type(m).__name__}."
        raise TypeError(msg)

    power = d if power is None else power
    if not math.isfinite(power) or power <= 0:
        msg = f"power must be positive and finite; got power={power}."
        raise ValueError(msg)

    rng = np.random.default_rng(rng)

    if not np.isfinite(X).all():
        msg = "X must contain only finite values; found NaN or inf."
        raise ValueError(msg)

    lower = X.min(axis=0)
    upper = X.max(axis=0)
    if np.all(lower == upper):
        msg = "All observations in X are identical."
        warnings.warn(msg, HopkinsUndefinedWarning, stacklevel=2)
        return math.nan

    null_sample = rng.uniform(lower, upper, size=(m, d))
    data_sample = X[rng.choice(n, size=m, replace=False)]

    tree = KDTree(X)

    u_dists, _ = tree.query(null_sample, k=1)
    w_dists, _ = tree.query(data_sample, k=[2])  # the sample itself is in X
    w_dists = np.ravel(w_dists)  # w_dists of shape (m, 1) due to k=[2]

    u_sum = np.sum(u_dists**power)
    w_sum = np.sum(w_dists**power)
    total = u_sum + w_sum

    return float(u_sum / total)
