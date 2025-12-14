import math
import sys

import numpy as np
import pytest

from hopkins_statistic import hopkins
from tests.constants import D, N


@pytest.mark.parametrize(
    "X", [1, np.ones(N), np.ones((N, D, D))], ids=["scalar", "1d", "3d"]
)
def test_X_must_be_2d(X):
    with pytest.raises(ValueError, match=r"X must be a 2D array"):
        hopkins(X)


def test_X_must_contain_at_least_three_observations():
    with pytest.raises(ValueError, match=r"at least 3 observations"):
        hopkins(np.ones((2, D)))


def test_X_with_three_observations_is_accepted(rng):
    assert 0 < hopkins(rng.uniform(size=(3, D)), rng=rng) < 1


@pytest.mark.parametrize("fill_value", [np.inf, np.nan])
def test_X_must_be_finite(fill_value):
    with pytest.raises(ValueError, match=r"X must contain only finite values"):
        hopkins(np.full((N, D), fill_value=fill_value))


def test_m_must_be_numeric(X_uniform):
    with pytest.raises(TypeError, match=r"m must be int or float"):
        hopkins(X_uniform, m="1")


@pytest.mark.parametrize("m", [-1, 0, N + 1])
def test_m_integer_must_satisfy_bounds(X_uniform, m):
    with pytest.raises(ValueError, match=r"1 <= m <= n"):
        hopkins(X_uniform, m=m)


@pytest.mark.parametrize("m", [-0.1, 0.0, 1.1])
def test_m_float_must_satisfy_bounds(X_uniform, m):
    with pytest.raises(ValueError, match=r"0 < m <= 1"):
        hopkins(X_uniform, m=m)


@pytest.mark.parametrize("m", [1, N, sys.float_info.min, 1.0])
def test_m_on_bounds_is_accepted(X_uniform, m, rng):
    assert 0 < hopkins(X_uniform, m=m, rng=rng) < 1


def test_power_must_be_numeric(X_uniform):
    with pytest.raises(TypeError, match=r"m must be int or float"):
        hopkins(X_uniform, m="1")


@pytest.mark.parametrize("power", [-1, 0, math.inf, math.nan])
def test_power_must_be_positive_and_finite(X_uniform, power):
    with pytest.raises(ValueError, match=r"power must be positive and finite"):
        hopkins(X_uniform, power=power)
