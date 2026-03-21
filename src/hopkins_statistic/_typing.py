import sys

if sys.version_info >= (3, 12):
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType  # pragma: no cover

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar  # pragma: no cover

from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence

DType = TypeVar("DType", bound=np.generic, default=Any)
Array: TypeAlias = np.ndarray[Any, np.dtype[DType]]
Array1D: TypeAlias = np.ndarray[tuple[int], np.dtype[DType]]
Array2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[DType]]

# See SPEC 7: https://scientific-python.org/specs/spec-0007/
RNGLike = TypeAliasType("RNGLike", Generator | BitGenerator)
SeedLike = TypeAliasType(
    "SeedLike",
    int
    | np.integer[Any]
    | Sequence[int]
    | SeedSequence
    | np.ndarray[Any, np.dtype[np.integer[Any]]],
)
