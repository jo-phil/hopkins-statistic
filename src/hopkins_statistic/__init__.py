"""Compute the Hopkins statistic to assess clustering tendency.

Main entry points are the `hopkins` and `hopkins_test` functions.

## Installation
.. include:: ../../README.md
    :start-after: ## Installation
    :end-before: ## License

.. include:: ../../docs/background.md

"""

__all__ = [
    "hopkins",
    "hopkins_test",
    "HopkinsTestResult",
    "HopkinsUndefinedWarning",
]

from ._inference import HopkinsTestResult, hopkins_test
from ._statistic import HopkinsUndefinedWarning, hopkins
