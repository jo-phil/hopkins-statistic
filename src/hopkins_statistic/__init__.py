"""Compute the Hopkins statistic to assess clustering tendency.

This library implements the Hopkins statistic as defined by [Hopkins and
Skellam (1954)](#2) and generalized by [Cross and Jain (1982)](#1).
The main entry point is the `hopkins` function.

## Installation
.. include:: ../../README.md
    :start-after: ## Installation
    :end-before: ## License

.. include:: ../../docs/background.md

"""

from ._statistic import HopkinsUndefinedWarning, hopkins

__all__ = ["hopkins", "HopkinsUndefinedWarning"]
