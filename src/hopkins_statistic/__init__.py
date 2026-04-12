"""Compute the Hopkins statistic to assess clustering tendency.

Functions:
    hopkins: Compute the Hopkins statistic.
    hopkins_test: Perform a Hopkins test.
"""

__all__ = [
    "HopkinsTestResult",
    "hopkins",
    "hopkins_test",
]

from ._inference import HopkinsTestResult, hopkins_test
from ._statistic import hopkins
