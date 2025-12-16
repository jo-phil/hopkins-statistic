import math

import numpy as np
import pytest

from hopkins_statistic import hopkins

# Smallest valid shape used in validation tests
N, D = 3, 1
X = np.arange(N * D).reshape((N, D))


@pytest.mark.parametrize(
    "X_invalid",
    [1, np.ones(N), np.ones((N, D, D))],
    ids=["scalar", "1d", "3d"],
)
def test_X_must_be_2d(X_invalid):
    with pytest.raises(ValueError, match=r"X must be a 2D array"):
        hopkins(X_invalid)


def test_X_must_contain_at_least_three_observations():
    with pytest.raises(ValueError, match=r"at least 3 observations"):
        hopkins(X[:-1])


@pytest.mark.parametrize("fill_value", [np.inf, np.nan])
def test_X_must_be_finite(fill_value):
    with pytest.raises(ValueError, match=r"X must contain only finite values"):
        hopkins(np.full((N, D), fill_value=fill_value))


def test_m_must_be_numeric():
    with pytest.raises(TypeError, match=r"m must be int or float"):
        hopkins(X, m="1")


@pytest.mark.parametrize("m", [-1, 0, N + 1])
def test_m_integer_must_satisfy_bounds(m):
    with pytest.raises(ValueError, match=r"1 <= m <= n"):
        hopkins(X, m=m)


@pytest.mark.parametrize("m", [-0.1, 0.0, 1.1, math.nan])
def test_m_float_must_satisfy_bounds(m):
    with pytest.raises(ValueError, match=r"0 < m <= 1"):
        hopkins(X, m=m)


@pytest.mark.parametrize("m", [1, N, 0.1, 1.0])
def test_m_on_bounds_is_accepted(m, rng):
    assert 0 < hopkins(X, m=m, rng=rng) < 1


def test_power_must_be_numeric():
    with pytest.raises(TypeError):
        hopkins(X, power="1")


@pytest.mark.parametrize("power", [-1, 0, math.inf, math.nan])
def test_power_must_be_positive_and_finite(power):
    with pytest.raises(ValueError, match=r"power must be positive and finite"):
        hopkins(X, power=power)
