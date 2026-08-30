from __future__ import annotations

import csv
import json
from pathlib import Path


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    scalar_keys = []
    for k, v in rows[0].items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            scalar_keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=scalar_keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in scalar_keys})
