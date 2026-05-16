from typing import Literal, NamedTuple, TypeAlias

from numpy.typing import ArrayLike
from scipy.stats import beta

from ._sampling import Frame
from ._statistic import _hopkins
from ._typing import ToRNG

Alternative: TypeAlias = Literal["clustered", "regular", "two-sided"]
"""Alternative hypothesis for [`hopkins_test`][]."""


class HopkinsTestResult(NamedTuple):
    """Result of a Hopkins test.

    Attributes:
        statistic: The Hopkins statistic.
        pvalue: The p-value associated with the given alternative.
    """

    statistic: float
    pvalue: float


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
    rng: ToRNG = None,
) -> HopkinsTestResult:
    """Perform a Hopkins test.

    The Hopkins test tests the null hypothesis of complete spatial
    randomness (CSR) by comparing the observed Hopkins statistic to its
    Beta(m, m) null distribution.

    Parameters:
        X: Array-like of shape `(n, d)`, with `n >= 3` observations
            in `d >= 1` dimensions. Must contain only finite real values.
        m: Sample size, or its fraction of the `n_in` points in the `frame`.

            - If `int`, this must satisfy `1 <= m <= n_in`.
            - If `float`, this must satisfy `0 < m <= 1`,
              and the sample size is `ceil(m * n_in)`.

        frame: Area sampling frame. Must be one of:

            - Literal `'bbox'` to use the axis-aligned bounding box of `X`, or
            - Literal `'hull'` to use the convex hull of `X`, or
            - Pair `(lower, upper)` defining the bounds of a rectangular
              sampling frame. Both must be broadcastable to shape `(d,)`.
              While data points outside a given frame are ignored during
              sampling, they can still be nearest neighbors.

        toroidal: If `True`, compute distances with periodic boundary conditions.
        alternative: Alternative hypothesis of departure from CSR toward more
            `'clustered'` or `'regular'` data, or in either direction: `'two-sided'`.
        rng: Random number generator or seed to be passed to
            [`numpy.random.default_rng`][]. Specify for reproducibility.

    Returns:
        The result of the Hopkins test (statistic and p-value).
    """
    statistic, m = _hopkins(
        X, m=m, frame=frame, toroidal=toroidal, power=None, rng=rng
    )
    pvalue = _hopkins_pvalue(statistic, m=m, alternative=alternative)

    return HopkinsTestResult(statistic=statistic, pvalue=pvalue)
