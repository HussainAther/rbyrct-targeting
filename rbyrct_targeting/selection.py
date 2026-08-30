from __future__ import annotations

import math
import numpy as np

from .geometry import Ray, angular_separation_deg


def _row_to_ray(row):
    return Ray(
        np.array([row["source_x_mm"], row["source_y_mm"], row["source_z_mm"]], dtype=float),
        np.array([row["detector_x_mm"], row["detector_y_mm"], row["detector_z_mm"]], dtype=float),
        ray_id=int(row["ray_id"]),
    )


def _valid(rows):
    return [r for r in rows if r["intersects"]]


def select_random(rows, k: int, seed: int = 0):
    valid = _valid(rows)
    if k > len(valid):
        raise ValueError("k exceeds number of intersecting candidate rays.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(valid), size=k, replace=False)
    return [valid[i] for i in idx]


def select_max_path_length(rows, k: int):
    valid = _valid(rows)
    return sorted(valid, key=lambda r: (-r["path_length_mm"], r["ray_id"]))[:k]


def select_angular_diversity(rows, k: int):
    valid = _valid(rows)
    if not valid:
        return []
    if k > len(valid):
        raise ValueError("k exceeds number of intersecting candidate rays.")

    # Seed with strongest central/chord ray.
    selected = [max(valid, key=lambda r: (r["path_length_mm"], -r["ray_id"]))]
    remaining = {r["ray_id"]: r for r in valid if r["ray_id"] != selected[0]["ray_id"]}

    while len(selected) < k:
        best = None
        best_score = None
        for row in remaining.values():
            ray = _row_to_ray(row)
            min_sep = min(angular_separation_deg(ray, _row_to_ray(s)) for s in selected)
            score = (min_sep, row["path_length_mm"], -row["ray_id"])
            if best is None or score > best_score:
                best = row
                best_score = score
        selected.append(best)
        remaining.pop(best["ray_id"])
    return selected


def select_hybrid(rows, k: int, path_weight: float = 0.6, angular_weight: float = 0.4):
    valid = _valid(rows)
    if k > len(valid):
        raise ValueError("k exceeds number of intersecting candidate rays.")
    if not valid:
        return []

    max_path = max(r["path_length_mm"] for r in valid) or 1.0
    selected = [max(valid, key=lambda r: (r["path_length_mm"], -r["ray_id"]))]
    remaining = {r["ray_id"]: r for r in valid if r["ray_id"] != selected[0]["ray_id"]}

    while len(selected) < k:
        best = None
        best_score = None
        for row in remaining.values():
            ray = _row_to_ray(row)
            min_sep = min(angular_separation_deg(ray, _row_to_ray(s)) for s in selected)
            angular_norm = min(min_sep / 90.0, 1.0)
            path_norm = row["path_length_mm"] / max_path
            score = path_weight * path_norm + angular_weight * angular_norm
            tie = (score, row["path_length_mm"], -row["ray_id"])
            if best is None or tie > best_score:
                best = row
                best_score = tie
        selected.append(best)
        remaining.pop(best["ray_id"])
    return selected


def select_greedy_coverage(rows, k: int, polar_bins: int = 6, azimuth_bins: int = 12):
    """
    Greedy angular-bin coverage with path length as a tie-breaker.
    This is a simple baseline, not a physical dose/coverage optimizer.
    """
    valid = _valid(rows)
    if k > len(valid):
        raise ValueError("k exceeds number of intersecting candidate rays.")

    def bin_id(row):
        az = (row["azimuth_deg"] + 180.0) % 360.0
        el = max(-90.0, min(90.0, row["elevation_deg"]))
        ai = min(azimuth_bins - 1, int(az / 360.0 * azimuth_bins))
        pi = min(polar_bins - 1, int((el + 90.0) / 180.0 * polar_bins))
        return pi, ai

    selected = []
    covered = set()
    remaining = list(valid)

    while remaining and len(selected) < k:
        best = max(
            remaining,
            key=lambda r: (
                1 if bin_id(r) not in covered else 0,
                r["path_length_mm"],
                -r["ray_id"],
            ),
        )
        selected.append(best)
        covered.add(bin_id(best))
        remaining.remove(best)

    return selected


def summarize_selection(rows):
    if not rows:
        return {
            "count": 0,
            "mean_path_length_mm": 0.0,
            "min_path_length_mm": 0.0,
            "max_path_length_mm": 0.0,
            "mean_pairwise_angular_separation_deg": 0.0,
            "min_pairwise_angular_separation_deg": 0.0,
        }

    paths = np.array([r["path_length_mm"] for r in rows], dtype=float)
    rays = [_row_to_ray(r) for r in rows]
    seps = []
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            seps.append(angular_separation_deg(rays[i], rays[j]))

    return {
        "count": len(rows),
        "mean_path_length_mm": float(paths.mean()),
        "min_path_length_mm": float(paths.min()),
        "max_path_length_mm": float(paths.max()),
        "mean_pairwise_angular_separation_deg": float(np.mean(seps)) if seps else 0.0,
        "min_pairwise_angular_separation_deg": float(np.min(seps)) if seps else 0.0,
    }
