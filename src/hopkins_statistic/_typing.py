import sys

if sys.version_info >= (3, 13):
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence

# See [SPEC 7](https://scientific-python.org/specs/spec-0007/)
RNGLike = TypeAliasType("RNGLike", Generator | BitGenerator)
SeedLike = TypeAliasType(
    "SeedLike",
    int
    | np.integer[Any]
    | Sequence[int]
    | SeedSequence
    | np.ndarray[Any, np.dtype[np.integer[Any]]],
)
