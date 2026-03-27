from typing import Literal, NamedTuple, TypeAlias

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import beta

from ._statistic import Frame, _hopkins
from ._typing import RNGLike, SeedLike

Alternative: TypeAlias = Literal["clustered", "regular", "two-sided"]


class HopkinsTestResult(NamedTuple):
    """Result of a Hopkins test."""

    statistic: float
    """The Hopkins statistic."""
    pvalue: float
    """The p-value associated with the given alternative."""


# Exclude constructor from public API docs
HopkinsTestResult.__new__.__doc__ = None


def _hopkins_pvalue(
    H: float,
    *,
    m: int,
    alternative: Alternative,
) -> float:
    if alternative == "clustered":
        pvalue = beta.sf(H, m, m)
    elif alternative == "regular":
        pvalue = beta.cdf(H, m, m)
    elif alternative == "two-sided":
        pvalue = 2 * min(beta.cdf(H, m, m), beta.sf(H, m, m))
    else:
        msg = f"Invalid alternative: {alternative}"
        raise ValueError(msg)

    return float(pvalue)


def hopkins_test(
    X: ArrayLike,
    *,
    m: int | float = 0.1,
    frame: Frame = "bbox",
    toroidal: bool = False,
    alternative: Alternative = "clustered",
    rng: RNGLike | SeedLike | None = None,
) -> HopkinsTestResult:
    """Perform a Hopkins test.

    The Hopkins test tests the null hypothesis of complete spatial
    randomness (CSR) by comparing the observed Hopkins statistic to its
    Beta(m, m) null distribution.

    Args:
        X: Array-like of shape `(n, d)`, with `n >= 3` observations
            in `d >= 1` dimensions. Must contain only finite real values.
        m: Sample size, or its fraction of the `n_in` points in the `frame`.
            - If int, this must satisfy `1 <= m <= n_in`.
            - If float, this must satisfy `0 < m <= 1`,
              and the sample size is `ceil(m * n_in)`.
        frame: Area sampling frame. Must be one of:
            - Literal `bbox` to use the axis-aligned bounding box of `X`, or
            - Pair `(lower, upper)` defining the bounds of a rectangular
              sampling frame. Both must be broadcastable to shape `(d,)`.
              While data points outside a given frame are ignored during
              sampling, they can still be nearest neighbors.
        toroidal: If True, compute distances with periodic boundary conditions.
        alternative: Alternative hypothesis of departure from CSR toward more
            `clustered` or `regular` data, or in either direction: `two-sided`.
        rng: Random number generator or seed passed to
            `numpy.random.default_rng`. Specify for repeatable behavior.

    Returns:
        The result of the Hopkins test (statistic and p-value).

    """
    X = np.asarray(X, dtype=float)

    statistic, m = _hopkins(
        X, m=m, frame=frame, toroidal=toroidal, power=None, rng=rng
    )
    pvalue = _hopkins_pvalue(statistic, m=m, alternative=alternative)

    return HopkinsTestResult(statistic=statistic, pvalue=pvalue)
