from __future__ import annotations

import numpy as np

from .geometry import Ray, Sphere, ray_sphere_metrics


def _grid_points(y_mm: float, x_extent_mm: float, z_extent_mm: float, nx: int, nz: int):
    xs = np.linspace(-x_extent_mm, x_extent_mm, nx)
    zs = np.linspace(-z_extent_mm, z_extent_mm, nz)
    return [np.array([x, y_mm, z], dtype=float) for z in zs for x in xs]


def _ideal_detector_for_target(source, target, detector_y_mm: float):
    """
    Return the detector-plane point collinear with source and target.
    """
    denom = target[1] - source[1]
    if abs(denom) < 1e-12:
        raise ValueError("Target and source cannot share Y when projecting to detector plane.")
    t = (detector_y_mm - source[1]) / denom
    return source + t * (target - source)


def generate_candidate_rays(
    sphere: Sphere,
    source_y_mm: float = -110.0,
    detector_y_mm: float = 110.0,
    source_x_extent_mm: float = 45.0,
    source_z_extent_mm: float = 45.0,
    detector_x_extent_mm: float = 45.0,
    detector_z_extent_mm: float = 45.0,
    source_nx: int = 9,
    source_nz: int = 9,
    detector_nx: int = 13,
    detector_nz: int = 13,
    keep_only_intersections: bool = False,
    target_centered: bool = True,
    detector_offset_extent_mm: float = 3.0,
):
    """
    Generate source-to-detector candidate rays.

    target_centered=True:
        For each source, calculate the detector-plane point collinear with the
        lesion center, then sample a local detector offset grid around that point.
        This intentionally creates a useful targeting benchmark with both hits
        and near-misses around a small lesion.

    target_centered=False:
        Use the full source x detector Cartesian grid.
    """
    sources = _grid_points(
        source_y_mm, source_x_extent_mm, source_z_extent_mm, source_nx, source_nz
    )

    rows = []
    rid = 0

    if target_centered:
        dxs = np.linspace(-detector_offset_extent_mm, detector_offset_extent_mm, detector_nx)
        dzs = np.linspace(-detector_offset_extent_mm, detector_offset_extent_mm, detector_nz)

        for s in sources:
            ideal = _ideal_detector_for_target(s, sphere.center, detector_y_mm)
            for dz in dzs:
                for dx in dxs:
                    d = ideal + np.array([dx, 0.0, dz], dtype=float)

                    # Detector aperture constraint.
                    if abs(d[0]) > detector_x_extent_mm or abs(d[2]) > detector_z_extent_mm:
                        rid += 1
                        continue

                    ray = Ray(s, d, ray_id=rid)
                    metrics = ray_sphere_metrics(ray, sphere)
                    if keep_only_intersections and not metrics["intersects"]:
                        rid += 1
                        continue

                    rows.append({
                        "ray_id": rid,
                        "source_x_mm": float(s[0]),
                        "source_y_mm": float(s[1]),
                        "source_z_mm": float(s[2]),
                        "detector_x_mm": float(d[0]),
                        "detector_y_mm": float(d[1]),
                        "detector_z_mm": float(d[2]),
                        "ideal_detector_x_mm": float(ideal[0]),
                        "ideal_detector_z_mm": float(ideal[2]),
                        "detector_offset_x_mm": float(dx),
                        "detector_offset_z_mm": float(dz),
                        **metrics,
                    })
                    rid += 1
    else:
        detectors = _grid_points(
            detector_y_mm, detector_x_extent_mm, detector_z_extent_mm, detector_nx, detector_nz
        )
        for s in sources:
            for d in detectors:
                ray = Ray(s, d, ray_id=rid)
                metrics = ray_sphere_metrics(ray, sphere)
                if keep_only_intersections and not metrics["intersects"]:
                    rid += 1
                    continue

                rows.append({
                    "ray_id": rid,
                    "source_x_mm": float(s[0]),
                    "source_y_mm": float(s[1]),
                    "source_z_mm": float(s[2]),
                    "detector_x_mm": float(d[0]),
                    "detector_y_mm": float(d[1]),
                    "detector_z_mm": float(d[2]),
                    **metrics,
                })
                rid += 1

    return rows
