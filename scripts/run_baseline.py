from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rbyrct_targeting.geometry import Sphere
from rbyrct_targeting.candidates import generate_candidate_rays
from rbyrct_targeting.selection import (
    select_random,
    select_max_path_length,
    select_angular_diversity,
    select_hybrid,
    select_greedy_coverage,
    summarize_selection,
)
from rbyrct_targeting.io_utils import write_json, write_csv
from rbyrct_targeting.visualize import plot_selection


def main():
    cfg = json.loads((ROOT / "configs" / "baseline.json").read_text())
    lesion = cfg["lesion"]
    geom = cfg["geometry"]
    sel = cfg["selection"]

    sphere = Sphere(
        center=lesion["center_mm"],
        radius_mm=lesion["diameter_mm"] / 2.0,
    )

    t0 = time.perf_counter()
    candidates = generate_candidate_rays(sphere=sphere, **geom)
    generation_s = time.perf_counter() - t0

    intersecting = [r for r in candidates if r["intersects"]]
    k = min(sel["k"], len(intersecting))

    methods = {
        "random": select_random(candidates, k, seed=sel["seed"]),
        "max_path_length": select_max_path_length(candidates, k),
        "angular_diversity": select_angular_diversity(candidates, k),
        "hybrid": select_hybrid(candidates, k),
        "greedy_coverage": select_greedy_coverage(candidates, k),
    }

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)

    write_csv(out / "candidate_rays.csv", candidates)
    write_json(out / "candidate_rays.json", candidates)

    summaries = {}
    for name, rows in methods.items():
        write_csv(out / f"selected_{name}.csv", rows)
        write_json(out / f"selected_{name}.json", rows)
        summaries[name] = summarize_selection(rows)

    report = {
        "candidate_ray_count": len(candidates),
        "intersecting_ray_count": len(intersecting),
        "intersection_fraction": len(intersecting) / len(candidates) if candidates else 0.0,
        "generation_wall_time_s": generation_s,
        "selection_k": k,
        "summaries": summaries,
    }
    write_json(out / "benchmark_summary.json", report)

    plot_selection(sphere, methods["hybrid"], out / "hybrid_selection.png")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
