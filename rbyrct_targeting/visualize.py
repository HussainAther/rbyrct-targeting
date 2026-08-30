from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .geometry import Sphere


def plot_selection(sphere: Sphere, rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Sphere wireframe
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x = sphere.center[0] + sphere.radius_mm * np.outer(np.cos(u), np.sin(v))
    y = sphere.center[1] + sphere.radius_mm * np.outer(np.sin(u), np.sin(v))
    z = sphere.center[2] + sphere.radius_mm * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, linewidth=0.4, alpha=0.5)

    for row in rows:
        s = np.array([row["source_x_mm"], row["source_y_mm"], row["source_z_mm"]])
        d = np.array([row["detector_x_mm"], row["detector_y_mm"], row["detector_z_mm"]])
        ax.plot([s[0], d[0]], [s[1], d[1]], [s[2], d[2]], linewidth=1.0)

    ax.scatter([sphere.center[0]], [sphere.center[1]], [sphere.center[2]], s=30)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("RBYRCT targeting diagnostic")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
