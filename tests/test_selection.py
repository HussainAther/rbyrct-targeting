import numpy as np

from rbyrct_targeting.geometry import Sphere
from rbyrct_targeting.candidates import generate_candidate_rays
from rbyrct_targeting.selection import (
    select_random,
    select_max_path_length,
    select_angular_diversity,
    select_hybrid,
    select_greedy_coverage,
)


def _rows():
    sphere = Sphere(np.array([12.0, 0.0, 5.0]), 1.0)
    return generate_candidate_rays(
        sphere,
        source_nx=7,
        source_nz=7,
        detector_nx=9,
        detector_nz=9,
    )


def test_random_is_seeded():
    rows = _rows()
    a = [r["ray_id"] for r in select_random(rows, 5, seed=123)]
    b = [r["ray_id"] for r in select_random(rows, 5, seed=123)]
    assert a == b


def test_all_selection_methods_return_intersecting_rays():
    rows = _rows()
    methods = [
        select_max_path_length(rows, 5),
        select_angular_diversity(rows, 5),
        select_hybrid(rows, 5),
        select_greedy_coverage(rows, 5),
    ]
    for selected in methods:
        assert len(selected) == 5
        assert all(r["intersects"] for r in selected)
