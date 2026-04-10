"""Compute the Hopkins statistic to assess clustering tendency.

Main entry points are the `hopkins` and `hopkins_test` functions.
"""

__all__ = [
    "hopkins",
    "hopkins_test",
    "HopkinsTestResult",
]

from ._inference import HopkinsTestResult, hopkins_test
from ._statistic import hopkins
