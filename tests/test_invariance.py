from unittest.mock import patch

import pytest

from hopkins_statistic import hopkins


@pytest.fixture
def X(rng):
    return rng.uniform(size=(100, 2))


def test_shift_invariance(X, toroidal):
    H1 = hopkins(X, toroidal=toroidal, rng=42)
    H2 = hopkins(X - 2, toroidal=toroidal, rng=42)
    assert H2 == pytest.approx(H1)  # noqa: SIM300


def test_scale_invariance(X, toroidal):
    H1 = hopkins(X, toroidal=toroidal, rng=42)
    H2 = hopkins(2 * X, toroidal=toroidal, rng=42)
    assert H2 == pytest.approx(H1)  # noqa: SIM300


def test_explicit_bbox_equivalence(hopkins_func, X):
    baseline = hopkins_func(X, frame="bbox", rng=42)
    explicit = hopkins_func(X, frame=(X.min(axis=0), X.max(axis=0)), rng=42)
    assert explicit == baseline


def test_toroidal_non_wrapping_equivalence(X):
    H1 = hopkins(X, toroidal=False, rng=42)
    with patch("numpy.nextafter", new=lambda x, _: 2 * x):
        H2 = hopkins(X, toroidal=True, rng=42)
    assert H2 == pytest.approx(H1)  # noqa: SIM300
