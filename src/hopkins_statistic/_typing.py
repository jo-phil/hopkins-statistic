from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence

FloatArray: TypeAlias = np.ndarray[Any, np.dtype[np.float64]]
FloatArray1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
FloatArray2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]

# See SPEC 7: https://scientific-python.org/specs/spec-0007/
RNGLike: TypeAlias = Generator | BitGenerator
SeedLike: TypeAlias = (
    int
    | np.integer[Any]
    | Sequence[int]
    | SeedSequence
    | np.ndarray[Any, np.dtype[np.integer[Any]]]
)
ToRNG: TypeAlias = RNGLike | SeedLike | None
