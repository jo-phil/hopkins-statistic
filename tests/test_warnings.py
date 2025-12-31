import math

import numpy as np
import pytest

from hopkins_statistic import HopkinsUndefinedWarning, hopkins, hopkins_test

X_degenerate = np.ones((3, 1))


def test_hopkins_warns_for_degenerate_data():
    with pytest.warns(HopkinsUndefinedWarning):
        statistic = hopkins(X_degenerate)
    assert math.isnan(statistic)


def test_hopkins_test_warns_for_degenerate_data():
    with pytest.warns(HopkinsUndefinedWarning):
        result = hopkins_test(X_degenerate)
    assert math.isnan(result.statistic)
    assert math.isnan(result.pvalue)
