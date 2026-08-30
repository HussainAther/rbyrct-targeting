from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

EPS = 1e-12


def _vec3(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"Expected 3-vector, got shape {arr.shape}")
    return arr


@dataclass(frozen=True)
class Sphere:
    center: np.ndarray
    radius_mm: float

    def __post_init__(self):
        object.__setattr__(self, "center", _vec3(self.center))
        if self.radius_mm <= 0:
            raise ValueError("Sphere radius must be positive.")


@dataclass(frozen=True)
class Ray:
    source: np.ndarray
    detector: np.ndarray
    ray_id: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "source", _vec3(self.source))
        object.__setattr__(self, "detector", _vec3(self.detector))
        if np.linalg.norm(self.detector - self.source) <= EPS:
            raise ValueError("Ray source and detector cannot coincide.")

    @property
    def vector(self) -> np.ndarray:
        return self.detector - self.source

    @property
    def length_mm(self) -> float:
        return float(np.linalg.norm(self.vector))

    @property
    def direction(self) -> np.ndarray:
        return self.vector / self.length_mm


def ray_sphere_metrics(ray: Ray, sphere: Sphere) -> dict:
    """
    Compute segment/sphere intersection metrics.

    Parameterization:
        p(t) = source + t * (detector-source), t in [0,1]
    """
    s = ray.source
    v = ray.vector
    c = sphere.center
    r = sphere.radius_mm

    vv = float(np.dot(v, v))
    sc = s - c

    t_closest = float(-np.dot(sc, v) / vv)
    t_clamped = min(1.0, max(0.0, t_closest))
    p_closest = s + t_clamped * v
    closest_distance = float(np.linalg.norm(p_closest - c))

    a = vv
    b = 2.0 * float(np.dot(sc, v))
    cc = float(np.dot(sc, sc) - r * r)
    disc = b * b - 4.0 * a * cc

    intersects = False
    path_length = 0.0
    t_enter = None
    t_exit = None
    entry = None
    exit_ = None

    if disc >= -EPS:
        disc = max(0.0, disc)
        sqrt_disc = math.sqrt(disc)
        roots = sorted(((-b - sqrt_disc) / (2.0 * a),
                        (-b + sqrt_disc) / (2.0 * a)))
        lo = max(0.0, roots[0])
        hi = min(1.0, roots[1])
        if hi >= lo and hi >= 0.0 and lo <= 1.0:
            intersects = True
            t_enter, t_exit = float(lo), float(hi)
            entry = s + t_enter * v
            exit_ = s + t_exit * v
            path_length = float(np.linalg.norm(exit_ - entry))

    d = ray.direction
    azimuth_deg = float(np.degrees(np.arctan2(d[0], d[1])))
    elevation_deg = float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1]))))

    return {
        "ray_id": ray.ray_id,
        "intersects": bool(intersects),
        "closest_distance_mm": closest_distance,
        "path_length_mm": path_length,
        "t_enter": t_enter,
        "t_exit": t_exit,
        "entry_mm": None if entry is None else entry.tolist(),
        "exit_mm": None if exit_ is None else exit_.tolist(),
        "direction": d.tolist(),
        "azimuth_deg": azimuth_deg,
        "elevation_deg": elevation_deg,
    }


def angular_separation_deg(ray_a: Ray, ray_b: Ray) -> float:
    da = ray_a.direction
    db = ray_b.direction
    cosang = float(np.clip(np.dot(da, db), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))
