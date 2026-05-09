from copy import deepcopy

import numpy as np
import pytest

from hopkins_statistic import hopkins, hopkins_test

DEFAULT: dict[str, object] = {}
BUFFER_ZONES = {"m": 0.08, "frame": (0.15, 0.85)}
PERIODIC_BOX = {"m": 20, "frame": (-0.01, 1.01), "toroidal": True}
CONVEX_HULL = {"frame": "hull"}


@pytest.mark.parametrize(
    ("seed", "kwargs", "expected", "expected_pvalue"),
    [
        (17, DEFAULT, 0.4445158774980207, 0.7832376758621498),
        (23, BUFFER_ZONES, 0.40378632307966583, 0.8161614323372398),
        (37, PERIODIC_BOX, 0.5307579555972737, 0.3493765152524343),
        (42, CONVEX_HULL, 0.3956635128966125, 0.9311725659007652)
    ],
    ids=["default", "buffer_zones", "periodic_box", "convex_hull"],
)
def test_regression_under_null(seed, kwargs, expected, expected_pvalue):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(250, 2))

    rng_copy = deepcopy(rng)

    assert hopkins(X, rng=rng, **kwargs) == pytest.approx(expected, abs=1e-15)

    result = hopkins_test(X, rng=rng_copy, **kwargs)
    assert result.statistic == pytest.approx(expected, abs=1e-15)
    assert result.pvalue == pytest.approx(expected_pvalue, abs=1e-15)
