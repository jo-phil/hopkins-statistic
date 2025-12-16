import math

import numpy as np
import pytest

from hopkins_statistic import HopkinsUndefinedWarning, hopkins


def test_degenerate_data():
    with pytest.warns(HopkinsUndefinedWarning):
        assert math.isnan(hopkins(np.ones((3, 1))))
