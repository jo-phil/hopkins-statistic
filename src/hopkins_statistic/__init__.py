"""Compute the Hopkins statistic to assess clustering tendency.

Functions:
    hopkins: Compute the Hopkins statistic.
    hopkins_test: Perform a Hopkins test.
"""

__all__ = [
    "Alternative",
    "Frame",
    "HopkinsTestResult",
    "ToRNG",
    "hopkins",
    "hopkins_test",
]

from ._inference import Alternative, HopkinsTestResult, hopkins_test
from ._sampling import Frame
from ._statistic import hopkins
from ._typing import ToRNG
