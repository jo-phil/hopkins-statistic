import itertools
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import beta

from hopkins_statistic import hopkins, hopkins_test
from hopkins_statistic._sampling import ConvexHull

N, D = 100, 2  # smallest reasonable shape for behavioral tests


@pytest.mark.slow
@pytest.mark.parametrize("d", [2, 3, 5], ids=lambda val: f"d={val}")
@pytest.mark.parametrize("edge_correction", [None, "buffer", "toroidal"])
def test_beta_moments_under_null(d, edge_correction, rng):
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


@pytest.mark.parametrize("d", [2, 3, 5], ids=lambda val: f"d={val}")
def test_beta_moments_under_null_in_convex_hull(d, rng):
    hull = ConvexHull(rng.normal(size=(2**d, d)))
    X = hull.sample(N, rng)

    with patch(  # Avoid recomputing the hull every iteration
        "hopkins_statistic._statistic.resolve_frame", new=lambda _, __: hull
    ):
        Hs = [hopkins(X, m=10, frame="hull", rng=rng) for _ in range(1000)]

    assert np.mean(Hs) == pytest.approx(0.5, abs=0.05)
    assert np.std(Hs) == pytest.approx(beta.std(10, 10), abs=0.05)


@pytest.mark.parametrize("seed", range(10))
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


@pytest.mark.parametrize("rng", range(10), indirect=True)
def test_clustering(toroidal, rng):
    corners = np.array(list(itertools.product([0, 1], repeat=D)))
    indices = rng.integers(len(corners), size=N)
    X = (corners[indices] + rng.normal(scale=0.1, size=(N, D))) % 1

    rng_copy = deepcopy(rng)

    result = hopkins_test(X, toroidal=toroidal, rng=rng)
    assert result.statistic > 0.7
    assert result.pvalue < 0.001

    heuristic = hopkins(X, toroidal=toroidal, power=1, rng=rng_copy)
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


def test_convex_hull_on_rotated_data(rng):
    X = rng.uniform(size=(N, D))

    # Rotate X to be clustered in its bounding box but not in its convex hull
    c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
    R = np.array([[c, -s], [s, c]])
    X = X @ R.T

    assert hopkins(X, rng=42) > 0.7
    assert 0.4 < hopkins(X, frame="hull", rng=42) < 0.6


def test_input_immutability(toroidal, rng):
    X = rng.uniform(-1, 1, size=(N, D))
    lower, upper = -np.ones(D), np.ones(D)
    X_copy, lower_copy, upper_copy = X.copy(), lower.copy(), upper.copy()

    hopkins(X, frame=(lower, upper), toroidal=toroidal)
    assert np.array_equal(X, X_copy)
    assert np.array_equal(lower, lower_copy)
    assert np.array_equal(upper, upper_copy)
