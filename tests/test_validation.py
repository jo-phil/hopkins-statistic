import math

import numpy as np
import pytest

from hopkins_statistic import hopkins, hopkins_test

# Smallest valid shape used in validation tests
N, D = 3, 1
X = np.arange(N * D).reshape((N, D))


@pytest.fixture(params=[hopkins, hopkins_test], ids=lambda f: f.__name__)
def hopkins_func(request):
    return request.param


def test_minimal_valid_X_is_accepted(rng):
    assert 0 < hopkins(X, rng=rng) < 1


@pytest.mark.parametrize(
    "X_invalid",
    [1, np.ones(N), np.ones((N, D, D))],
    ids=["scalar", "1d", "3d"],
)
def test_X_must_be_2d(hopkins_func, X_invalid):
    with pytest.raises(ValueError, match=r"X must be a 2D array"):
        hopkins_func(X_invalid)


def test_X_must_contain_at_least_three_observations(hopkins_func):
    with pytest.raises(ValueError, match=r"at least 3 observations"):
        hopkins_func(X[:-1])


def test_X_must_have_at_least_one_feature(hopkins_func):
    with pytest.raises(ValueError, match=r"at least 1 feature"):
        hopkins_func(np.empty((N, 0)))


@pytest.mark.parametrize("fill_value", [np.inf, np.nan])
def test_X_must_be_finite(hopkins_func, fill_value):
    with pytest.raises(ValueError, match=r"X must contain only finite values"):
        hopkins_func(np.full((N, D), fill_value=fill_value))


@pytest.mark.parametrize("m", [1, N, 0.1, 1.0])
def test_m_on_bounds_is_accepted(m, rng):
    assert 0 < hopkins(X, m=m, rng=rng) < 1


@pytest.mark.parametrize("m", [True, "1"])
def test_m_must_be_numeric(hopkins_func, m):
    with pytest.raises(TypeError, match=r"m must be int or float"):
        hopkins_func(X, m=m)


@pytest.mark.parametrize("m", [-1, 0, N + 1])
def test_m_integer_must_satisfy_bounds(hopkins_func, m):
    with pytest.raises(ValueError, match=r"1 <= m <= n"):
        hopkins_func(X, m=m)


@pytest.mark.parametrize("m", [-0.1, 0.0, 1.1, math.nan])
def test_m_float_must_satisfy_bounds(hopkins_func, m):
    with pytest.raises(ValueError, match=r"0 < m <= 1"):
        hopkins_func(X, m=m)


@pytest.mark.parametrize("power", [True, "1"])
def test_power_must_be_numeric(power):
    with pytest.raises(TypeError, match=r"power must be a real number"):
        hopkins(X, power=power)


@pytest.mark.parametrize("power", [-1, 0])
def test_power_must_be_positive(power):
    with pytest.raises(ValueError, match=r"power must be positive"):
        hopkins(X, power=power)


@pytest.mark.parametrize("power", [math.inf, math.nan])
def test_power_must_be_finite(power):
    with pytest.raises(ValueError, match=r"power must be finite"):
        hopkins(X, power=power)


@pytest.mark.parametrize("alternative", ["clustered", "regular", "two-sided"])
def test_valid_alternative_is_accepted(alternative):
    result = hopkins_test(X, alternative=alternative)
    assert 0 < result.statistic < 1
    assert 0 < result.pvalue < 1


def test_invalid_alternative_is_rejected():
    with pytest.raises(ValueError, match=r"Invalid alternative"):
        hopkins_test(X, alternative="")
