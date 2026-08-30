import numpy as np

from rbyrct_targeting.geometry import Sphere
from rbyrct_targeting.candidates import generate_candidate_rays


def test_candidate_generation_contains_intersections():
    sphere = Sphere(np.array([12.0, 0.0, 5.0]), 1.0)
    rows = generate_candidate_rays(
        sphere,
        source_nx=5,
        source_nz=5,
        detector_nx=7,
        detector_nz=7,
    )
    assert rows
    assert any(r["intersects"] for r in rows)
    assert all(r["source_y_mm"] == -110.0 for r in rows)
    assert all(r["detector_y_mm"] == 110.0 for r in rows)
