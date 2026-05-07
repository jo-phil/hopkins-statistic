from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import ConvexHull as ScipyConvexHull
from scipy.spatial import Delaunay, QhullError
from scipy.special import logsumexp

from ._typing import (
    BoolArray1D,
    FloatArray1D,
    FloatArray2D,
    FloatArray3D,
    ToRNG,
)

Frame: TypeAlias = (
    Literal["bbox", "hull"] | tuple[ArrayLike, ArrayLike] | ArrayLike
)


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

    @classmethod
    def from_data(cls, X: FloatArray2D) -> "Box":
        return cls(np.min(X, axis=0), np.max(X, axis=0), dim=X.shape[1])


class ConvexHull(SamplingFrame):
    simplices: FloatArray3D
    simplex_probs: FloatArray1D
    dim: int

    def __init__(self, X: FloatArray2D) -> None:
        try:
            vertices = X[ScipyConvexHull(X).vertices]
            simplices = vertices[Delaunay(vertices).simplices]
        except QhullError as e:
            msg = "X must span a full-dimensional convex hull"
            raise ValueError(msg) from e

        # Simplex volumes are abs(det(simplex_edges)) / dim!.
        # The common divisor cancels out when normalizing probabilities.
        # Perform operations in log-space for numerical stability.
        simplex_edges = simplices[:, 1:, :] - simplices[:, :1, :]
        _, logabsdet = np.linalg.slogdet(simplex_edges)
        log_simplex_probs = logabsdet - logsumexp(logabsdet)

        self.simplices = simplices
        self.simplex_probs = np.exp(log_simplex_probs)
        self.dim = X.shape[1]

    def contains(self, X: FloatArray2D) -> BoolArray1D:
        raise NotImplementedError

    def sample(self, m: int, rng: ToRNG) -> FloatArray2D:
        rng = np.random.default_rng(rng)

        # Sample simplices weighted by volume with replacement.
        sampled_simplices = rng.choice(
            self.simplices, size=m, p=self.simplex_probs, axis=0
        )

        # Sample points uniformly from simplices via barycentric coordinates.
        barycentric_coords = rng.dirichlet(np.ones(self.dim + 1), size=m)
        return np.einsum("ij,ijd->id", barycentric_coords, sampled_simplices)


def resolve_frame(X: FloatArray2D, frame: Frame) -> SamplingFrame:
    rule = "frame must be 'bbox', 'hull', or a pair of bounds (lower, upper)"

    match frame:
        case "bbox":
            return Box.from_data(X)
        case "hull":
            # Qhull requires at least 2D data; the bbox is equivalent in 1D.
            return ConvexHull(X) if X.shape[1] > 1 else Box.from_data(X)

        case [lower, upper]:
            return Box(lower, upper, dim=X.shape[1])

        case Sequence() | np.ndarray() if not isinstance(frame, str | bytes):
            msg = f"{rule}; got {type(frame).__name__} of length {len(frame)}."
            raise ValueError(msg)
        case _:
            msg = f"{rule}; got {type(frame).__name__}."
            raise TypeError(msg)
