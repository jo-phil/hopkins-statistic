"""Compute the Hopkins statistic to assess clustering tendency.

This library implements the Hopkins statistic as defined by [Hopkins and
Skellam (1954)](#2) and generalized by [Cross and Jain (1982)](#1).
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
