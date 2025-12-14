import numpy as np
import pytest

from tests.constants import SEED, D, N


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def X_uniform(rng):
    return rng.uniform(size=(N, D))
