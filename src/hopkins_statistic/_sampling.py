from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from ._typing import BoolArray1D, FloatArray1D, FloatArray2D, ToRNG

Frame: TypeAlias = Literal["bbox"] | tuple[ArrayLike, ArrayLike] | ArrayLike


class SamplingFrame(ABC):
    @abstractmethod
    def contains(self, X: FloatArray2D) -> BoolArray1D: ...

    @abstractmethod
    def sample(self, m: int, rng: ToRNG) -> FloatArray2D: ...


class Box(SamplingFrame):
    lower: FloatArray1D
    upper: FloatArray1D

    def __init__(
        self, lower: ArrayLike, upper: ArrayLike, *, dim: int
    ) -> None:
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        try:
            lower = np.broadcast_to(lower, (dim,))
            upper = np.broadcast_to(upper, (dim,))
        except ValueError:
            msg = "bounds must each be scalar or match the data dimension"
            raise ValueError(msg) from None

        if np.any(lower > upper):
            msg = "lower bounds must be lower than upper bounds"
            raise ValueError(msg)

        self.lower = lower
        self.upper = upper
        self.dim = dim

    def contains(self, X: FloatArray2D) -> BoolArray1D:
        return np.all((self.lower <= X) & (self.upper >= X), axis=1)

    def sample(self, m: int, rng: ToRNG) -> FloatArray2D:
        rng = np.random.default_rng(rng)
        return rng.uniform(self.lower, self.upper, size=(m, self.dim))


def resolve_frame(X: FloatArray2D, frame: Frame) -> SamplingFrame:
    match frame:
        case "bbox":
            return Box(np.min(X, axis=0), np.max(X, axis=0), dim=X.shape[1])
        case [lower, upper]:
            return Box(lower, upper, dim=X.shape[1])
        case Sequence() | np.ndarray() if not isinstance(frame, str | bytes):
            msg = (
                "frame must be 'bbox' or a pair of bounds (lower, upper); "
                f"got {type(frame).__name__} of length {len(frame)}."
            )
            raise ValueError(msg)
        case _:
            msg = (
                "frame must be 'bbox' or a pair of bounds (lower, upper); "
                f"got {type(frame).__name__}."
            )
            raise TypeError(msg)
