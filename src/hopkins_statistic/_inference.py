from typing import Literal, NamedTuple, TypeAlias

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import beta

from hopkins_statistic._statistic import _parse_m, _validate_shape, hopkins
from hopkins_statistic._typing import RNGLike, SeedLike

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
    alternative: Alternative = "clustered",
    rng: RNGLike | SeedLike | None = None,
) -> HopkinsTestResult:
    """Perform a Hopkins test.

    Args:
        X: Array-like of shape `(n, d)`, with `n >= 3` observations
            in `d >= 1` dimensions. Must contain only finite values.
        m: Sample size, or its fraction of `n`.
            - If int, this must satisfy `1 <= m <= n`.
            - If float, this must satisfy `0 < m <= 1`,
              and the sample size is `ceil(m * n)`.
        alternative: Alternative hypothesis of departure from CSR toward more
            `clustered` or `regular` data, or in either direction: `two-sided`.
        rng: Random number generator or seed passed to
            `numpy.random.default_rng`. Specify for repeatable behavior.

    Returns:
        The result of the Hopkins test (statistic and p-value).

    Warns:
        `HopkinsUndefinedWarning`: If all observations in `X` are identical.

    """
    X = np.asarray(X, dtype=float)

    n, _ = _validate_shape(X)
    m = _parse_m(m, n)

    statistic = hopkins(X, m=m, rng=rng)
    pvalue = _hopkins_pvalue(statistic, m=m, alternative=alternative)

    return HopkinsTestResult(statistic=statistic, pvalue=pvalue)
