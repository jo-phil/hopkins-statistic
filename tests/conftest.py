import numpy as np
import pytest

from hopkins_statistic import hopkins, hopkins_test


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture(params=[False, True], ids=["euclidean", "toroidal"])
def toroidal(request):
    return request.param


@pytest.fixture(params=[hopkins, hopkins_test], ids=lambda f: f.__name__)
def hopkins_func(request):
    return request.param
