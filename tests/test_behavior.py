import itertools

import numpy as np
import pytest
from scipy.stats import beta

from hopkins_statistic import hopkins, hopkins_test

N, D = 100, 2  # smallest reasonable shape for behavioral tests


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture(params=range(10), ids=lambda val: f"seed={val}")
def seed(request):
    return request.param


@pytest.mark.slow
@pytest.mark.parametrize("d", [2, 3, 5])
def test_beta_moments_under_uniform_null(d, rng):
    m = N // 10
    Xs = rng.uniform(size=(1000, N, d))
    Hs = [hopkins(X, m=m, rng=rng) for X in Xs]
    assert np.mean(Hs) == pytest.approx(0.5, abs=0.05)
    assert np.std(Hs) == pytest.approx(beta.std(m, m), abs=0.05)


def test_low_under_regularity(seed):
    X = np.array(list(itertools.product(range(int(N ** (1 / D))), repeat=D)))

    result = hopkins_test(X, alternative="regular", rng=seed)
    assert result.statistic < 0.3
    assert result.pvalue < 0.001

    heuristic = hopkins(X, power=1, rng=seed)
    assert heuristic != pytest.approx(result.statistic)
    assert heuristic < 0.35


def test_high_under_clustering(seed):
    rng = np.random.default_rng(seed)
    corners = np.array(list(itertools.product([0, 1], repeat=D)))
    indices = rng.integers(len(corners), size=N)
    X = (corners[indices] + rng.normal(scale=0.1, size=(N, D))) % 1

    result = hopkins_test(X, rng=seed)
    assert result.statistic > 0.7
    assert result.pvalue < 0.001

    heuristic = hopkins(X, power=1, rng=seed)
    assert heuristic != pytest.approx(result.statistic)
    assert heuristic > 0.7


def test_one_under_extreme_clustering(rng):
    X = [[0, 0], [1, 1]] * 2
    assert hopkins(X, rng=rng) == 1.0


def test_invariant_under_scale_and_shift(rng):
    X = rng.uniform(size=(N, D))
    H1 = hopkins(X, rng=42)
    H2 = hopkins(2 * X + 1, rng=42)
    assert H2 == pytest.approx(H1)  # noqa: SIM300
