import numpy as np
from rbyrct_targeting.geometry import Sphere, Ray, ray_sphere_metrics


def test_central_ray_has_diameter_path_length():
    sphere = Sphere(np.array([0.0, 0.0, 0.0]), 1.0)
    ray = Ray(np.array([0.0, -5.0, 0.0]), np.array([0.0, 5.0, 0.0]))
    m = ray_sphere_metrics(ray, sphere)
    assert m["intersects"]
    assert abs(m["path_length_mm"] - 2.0) < 1e-12


def test_tangent_ray_zero_chord():
    sphere = Sphere(np.array([0.0, 0.0, 0.0]), 1.0)
    ray = Ray(np.array([1.0, -5.0, 0.0]), np.array([1.0, 5.0, 0.0]))
    m = ray_sphere_metrics(ray, sphere)
    assert m["intersects"]
    assert abs(m["path_length_mm"]) < 1e-10


def test_miss():
    sphere = Sphere(np.array([0.0, 0.0, 0.0]), 1.0)
    ray = Ray(np.array([2.0, -5.0, 0.0]), np.array([2.0, 5.0, 0.0]))
    m = ray_sphere_metrics(ray, sphere)
    assert not m["intersects"]
    assert m["path_length_mm"] == 0.0
