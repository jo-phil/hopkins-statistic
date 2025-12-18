"""Hopkins statistic for assessing clustering tendency.

Implements the Hopkins statistic to test for departure from complete
spatial randomness (CSR). The main entry point is the `hopkins` function.
"""

from ._statistic import HopkinsUndefinedWarning, hopkins

__all__ = ["HopkinsUndefinedWarning", "hopkins"]
