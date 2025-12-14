import math

import numpy as np
import pytest

from hopkins_statistic import HopkinsUndefinedWarning, hopkins
from tests.constants import D, N


def test_degenerate_data():
    with pytest.warns(HopkinsUndefinedWarning):
        assert math.isnan(hopkins(np.ones((N, D))))
