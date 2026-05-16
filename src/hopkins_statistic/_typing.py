from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence

BoolArray1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.bool_]]
FloatArray: TypeAlias = np.ndarray[Any, np.dtype[np.float64]]
FloatArray1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
FloatArray2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]
FloatArray3D: TypeAlias = np.ndarray[
    tuple[int, int, int], np.dtype[np.float64]
]

# See SPEC 7: https://scientific-python.org/specs/spec-0007/
ToRNG: TypeAlias = (
    Generator
    | BitGenerator
    | int
    | np.integer[Any]
    | Sequence[int]
    | SeedSequence
    | np.ndarray[Any, np.dtype[np.integer[Any]]]
    | None
)
"""Random number generator or seed to be passed to [`numpy.random.default_rng`][]."""
