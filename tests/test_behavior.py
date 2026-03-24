import itertools
from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import beta

from hopkins_statistic import hopkins, hopkins_test

N, D = 100, 2  # smallest reasonable shape for behavioral tests


@pytest.fixture(params=[False, True], ids=["euclidean", "toroidal"])
def toroidal(request):
    return request.param


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture(params=range(10), ids=lambda val: f"seed={val}")
def seed(request):
    return request.param


@pytest.mark.slow
@pytest.mark.parametrize("d", [2, 3, 5], ids=lambda val: f"d={val}")
@pytest.mark.parametrize("edge_correction", [None, "buffer", "toroidal"])
def test_null(d, edge_correction, rng):
    m = N // 10

    # Using buffer zones to correct for edge effects, specify a frame that
    # expectedly contains about N datapoints (0.7^d ~ 1/d for d in {2, 3, 5}).
    n = d * N if edge_correction == "buffer" else N
    frame = (0.15, 0.85) if edge_correction == "buffer" else (0, 1)
    toroidal = edge_correction == "toroidal"

    Xs = rng.uniform(size=(1000, n, d))
    Hs = [hopkins(X, m=m, frame=frame, toroidal=toroidal, rng=rng) for X in Xs]

    # When correcting for edge effects, the empirical distribution of
    # the statistic is closer to the theoretical beta distribution.
    tol = 0.01 if edge_correction else 0.05

    assert np.mean(Hs) == pytest.approx(0.5, abs=tol)
    assert np.std(Hs) == pytest.approx(beta.std(m, m), abs=tol)


def test_regularity(toroidal, seed):
    X = np.array(list(itertools.product(range(N), repeat=D)))
    result = hopkins_test(
        X, toroidal=toroidal, alternative="regular", rng=seed
    )
    assert result.statistic < 0.3
    assert result.pvalue < 0.001

    heuristic = hopkins(X, toroidal=toroidal, power=1, rng=seed)
    assert heuristic != pytest.approx(result.statistic)
    assert heuristic < 0.3


def test_clustering(toroidal, seed):
    rng = np.random.default_rng(seed)
    corners = np.array(list(itertools.product([0, 1], repeat=D)))
    indices = rng.integers(len(corners), size=N)
    X = (corners[indices] + rng.normal(scale=0.1, size=(N, D))) % 1

    result = hopkins_test(X, toroidal=toroidal, rng=seed)
    assert result.statistic > 0.7
    assert result.pvalue < 0.001

    heuristic = hopkins(X, toroidal=toroidal, power=1, rng=seed)
    assert heuristic != pytest.approx(result.statistic)
    assert heuristic > 0.7


def test_extreme_clustering(toroidal):
    X = [[0, 0], [1, 1]] * 2
    assert hopkins(X, toroidal=toroidal) == 1.0


def test_enlarged_bounds(toroidal, rng):
    X = rng.uniform(size=(N, D))

    result = hopkins_test(X, toroidal=toroidal, rng=rng)
    assert 0.4 < result.statistic < 0.6
    assert result.pvalue > 0.1

    result = hopkins_test(X, frame=(0, 1.4), toroidal=toroidal, rng=rng)
    assert result.statistic > 0.7
    assert result.pvalue < 0.001


def test_toroidal_vertices(rng):
    X = list(itertools.product([0, 1], repeat=7))
    assert hopkins(X, toroidal=False, rng=rng) < 0.3
    assert hopkins(X, toroidal=True) == 1.0


def test_toroidal_invariance(rng):
    X = rng.uniform(size=(N, D))
    H1 = hopkins(X, toroidal=False, rng=42)
    with patch("numpy.nextafter", new=lambda x, _: 2 * x):
        H2 = hopkins(X, toroidal=True, rng=42)
    assert H2 == pytest.approx(H1)  # noqa: SIM300


def test_scale_and_shift_invariance(toroidal, rng):
    X = rng.uniform(size=(N, D))
    H1 = hopkins(X, toroidal=toroidal, rng=42)
    H2 = hopkins(2 * X + 1, toroidal=toroidal, rng=42)
    assert H2 == pytest.approx(H1)  # noqa: SIM300


def test_input_immutability(toroidal, rng):
    X = rng.uniform(-1, 1, size=(N, D))
    lower, upper = -np.ones(D), np.ones(D)
    X_copy, lower_copy, upper_copy = X.copy(), lower.copy(), upper.copy()

    hopkins(X, frame=(lower, upper), toroidal=toroidal)
    assert np.array_equal(X, X_copy)
    assert np.array_equal(lower, lower_copy)
    assert np.array_equal(upper, upper_copy)
