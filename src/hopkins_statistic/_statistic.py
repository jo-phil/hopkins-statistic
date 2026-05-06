import math
import numbers

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree

from ._sampling import Box, Frame, SamplingFrame, resolve_frame
from ._typing import FloatArray, FloatArray1D, FloatArray2D, ToRNG


def hopkins(
    X: ArrayLike,
    *,
    m: int | float = 0.1,
    frame: Frame = "bbox",
    toroidal: bool = False,
    power: int | float | None = None,
    rng: ToRNG = None,
) -> float:
    """Compute the Hopkins statistic.

    The Hopkins statistic measures clustering tendency by comparing
    nearest-neighbor distances of sampled data points with those of
    points placed uniformly at random in the sampling frame.

    Parameters:
        X: Array-like of shape `(n, d)`, with `n >= 3` observations
            in `d >= 1` dimensions. Must contain only finite real values.
        m: Sample size, or its fraction of the `n_in` points in the `frame`.

            - If `int`, this must satisfy `1 <= m <= n_in`.
            - If `float`, this must satisfy `0 < m <= 1`,
              and the sample size is `ceil(m * n_in)`.

        frame: Area sampling frame. Must be one of:

            - Literal `'bbox'` to use the axis-aligned bounding box of `X`, or
            - Pair `(lower, upper)` defining the bounds of a rectangular
              sampling frame. Both must be broadcastable to shape `(d,)`.
              While data points outside a given frame are ignored during
              sampling, they can still be nearest neighbors.

        toroidal: If True, compute distances with periodic boundary conditions.
        power: Exponent applied to Euclidean distances. Defaults to `d`.
            Must be positive and finite.
        rng: Random number generator or seed to be passed to
            [`numpy.random.default_rng`][]. Specify for reproducibility.

    Returns:
        The Hopkins statistic, a value between 0 and 1 (NaN if undefined).
    """
    statistic, _ = _hopkins(
        X, m=m, frame=frame, toroidal=toroidal, power=power, rng=rng
    )
    return statistic


def _hopkins(
    X: ArrayLike,
    *,
    m: int | float,
    frame: Frame,
    toroidal: bool,
    power: int | float | None,
    rng: ToRNG,
) -> tuple[float, int]:
    X = np.asarray(X, dtype=float)

    d = _validate_shape(X)
    toroidal = _parse_toroidal(toroidal)
    power = _parse_power(power, d)
    rng = np.random.default_rng(rng)

    if not np.isfinite(X).all():
        msg = "X must contain only finite values; found NaN or inf."
        raise ValueError(msg)

    implicit_frame = isinstance(frame, str)
    frame: SamplingFrame = resolve_frame(X, frame)

    boxsize = None
    if toroidal:
        X, frame, boxsize = _apply_toroidal_topology(X, frame)

    X_in = X if implicit_frame else X[frame.contains(X)]
    if toroidal and len(X_in) < len(X):
        msg = "Points must not be outside the frame in toroidal topology."
        raise ValueError(msg)

    m = _parse_m(m, len(X_in))

    null_sample = frame.sample(m, rng)
    data_sample = rng.choice(X_in, size=m, replace=False, axis=0)

    tree = KDTree(X, boxsize=boxsize)
    u = tree.query(null_sample, k=1)[0]
    w = np.asarray(tree.query(data_sample, k=2)[0])[:, 1]  # 1st NN is itself

    u_sum = np.sum(u**power)
    w_sum = np.sum(w**power)

    return float(u_sum / (u_sum + w_sum)), m


def _apply_toroidal_topology(
    X: FloatArray2D, frame: SamplingFrame
) -> tuple[FloatArray2D, SamplingFrame, FloatArray1D]:
    if not isinstance(frame, Box):
        msg = "toroidal topology requires a rectangular frame"
        raise TypeError(msg)

    X = X - frame.lower
    frame = Box(
        np.zeros_like(frame.lower),
        frame.upper - frame.lower,
        dim=X.shape[1],
    )

    return X, frame, np.nextafter(frame.upper, np.inf)


def _validate_shape(X: FloatArray) -> int:
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

    return int(d)


def _parse_m(m: int | float, n: int) -> int:
    if isinstance(m, numbers.Integral) and not isinstance(m, bool):
        if not 1 <= m <= n:
            msg = f"m must satisfy 1 <= m <= n_in; got m={m}, n_in={n}."
            raise ValueError(msg)
        return int(m)

    if isinstance(m, numbers.Real) and not isinstance(m, bool):
        if not 0 < m <= 1:
            msg = f"If m is a float, it must satisfy 0 < m <= 1; got m={m}."
            raise ValueError(msg)
        return math.ceil(m * n)

    msg = f"m must be int or float; got {type(m).__name__}."
    raise TypeError(msg)


def _parse_toroidal(toroidal: bool) -> bool:  # noqa: FBT001
    if isinstance(toroidal, (bool, np.bool_)):
        return bool(toroidal)
    msg = f"toroidal must be bool; got {type(toroidal).__name__}"
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
