import itertools

import numpy as np
import pytest
from scipy.stats import beta

from hopkins_statistic import hopkins

# Smallest reasonable shape used in behavioral tests
N, D = 100, 2


def test_near_half_under_uniform_null(rng):
    X = rng.uniform(size=(N, D))
    assert 0.4 < hopkins(X, rng=rng) < 0.6


@pytest.mark.slow
@pytest.mark.parametrize("d", [2, 3, 5])
def test_beta_moments_under_uniform_null(d, rng):
    m = N // 10
    Hs = [hopkins(rng.uniform(size=(N, d)), m=m, rng=rng) for _ in range(1000)]
    assert np.mean(Hs) == pytest.approx(0.5, abs=0.05)
    assert np.std(Hs) == pytest.approx(beta.std(m, m), abs=0.05)


def test_high_under_clustering(rng):
    corners = np.array(list(itertools.product([0, 1], repeat=D)))
    indices = rng.integers(len(corners), size=N)
    X = (corners[indices] + rng.normal(scale=0.1, size=(N, D))) % 1
    assert hopkins(X, rng=rng) > 0.7


def test_one_under_extreme_clustering(rng):
    X = [[0, 0], [1, 1]] * 2
    assert hopkins(X, rng=rng) == 1.0


def test_low_under_regularity(rng):
    X = list(itertools.product(range(10), repeat=2))
    assert hopkins(X, rng=rng) < 0.3


def test_invariant_under_scale_and_shift(rng):
    X = rng.uniform(size=(N, D))
    H1 = hopkins(X, rng=42)
    H2 = hopkins(2 * X + 1, rng=42)
    assert H2 == pytest.approx(H1)  # noqa: SIM300
