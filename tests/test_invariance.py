from unittest.mock import patch

import pytest

from hopkins_statistic import hopkins_test


@pytest.fixture
def X(rng):
    return rng.uniform(size=(100, 2))


def test_shift_invariance(hopkins_func, X, toroidal):
    baseline = hopkins_func(X, toroidal=toroidal, rng=42)
    shifted = hopkins_func(X - 2, toroidal=toroidal, rng=42)
    assert shifted == pytest.approx(baseline)


def test_scale_invariance(hopkins_func, X, toroidal):
    baseline = hopkins_func(X, toroidal=toroidal, rng=42)
    scaled = hopkins_func(2 * X, toroidal=toroidal, rng=42)
    assert scaled == pytest.approx(baseline)


def test_explicit_bbox_equivalence(hopkins_func, X):
    implicit = hopkins_func(X, frame="bbox", rng=42)
    explicit = hopkins_func(X, frame=(X.min(axis=0), X.max(axis=0)), rng=42)
    assert explicit == implicit


def test_toroidal_non_wrapping_equivalence(hopkins_func, X):
    non_toroidal = hopkins_func(X, toroidal=False, rng=42)
    with patch("numpy.nextafter", new=lambda x, _: 2 * x):
        toroidal = hopkins_func(X, toroidal=True, rng=42)
    assert toroidal == pytest.approx(non_toroidal)


def test_alternative_invariance(X):
    clustered = hopkins_test(X, alternative="clustered", rng=42)
    regular = hopkins_test(X, alternative="regular", rng=42)
    two_sided = hopkins_test(X, alternative="two-sided", rng=42)

    assert clustered.statistic == regular.statistic == two_sided.statistic

    assert clustered.pvalue == pytest.approx(1 - regular.pvalue)
    assert two_sided.pvalue == pytest.approx(
        2 * min(clustered.pvalue, regular.pvalue)
    )
