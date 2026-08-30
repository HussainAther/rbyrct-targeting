# rbyrct-targeting

A standalone research repository for geometric ray targeting in steerable ray-by-ray X-ray CT (RBYRCT).

## Research question

Given a known 3D target lesion, source geometry, detector geometry, and a candidate ray set:

> Which rays should be selected to interrogate the target efficiently under geometric constraints?

This repository intentionally focuses on **targeting**, not reconstruction.

## Baseline lesion

The default example uses a spherical lesion:

- center: `[12, 0, 5]` mm
- diameter: `2.0` mm

These values mirror the validated target-localization geometry used in the related RBYRCT cinematic work, but this repository remains independent.

## Implemented components

- 3D vector and ray representation
- Analytic ray-sphere intersection
- Candidate ray generation between source and detector planes, including a target-centered near-miss/hit sampling mode
- Per-ray metrics:
  - lesion intersection
  - closest approach
  - lesion chord/path length
  - direction
  - detector endpoint
  - angular descriptors
- Selection baselines:
  1. random
  2. maximum lesion path length
  3. angular diversity
  4. hybrid path-length + angular-diversity
  5. greedy coverage
- Deterministic seeded experiments
- JSON/CSV outputs
- Diagnostic 3D visualization
- Pytest validation

## Geometry convention

Canonical unit: **millimeter**

Default coordinates are right-handed:

- `+X`: lateral
- `+Y`: source-to-detector
- `+Z`: superior

Default planes:

- source plane: `Y = -110 mm`
- detector plane: `Y = +110 mm`

## Setup

```powershell
cd C:\Users\shuss\Downloads\rbyrct\rbyrct-targeting
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

## Run tests

```powershell
pytest -q
```

## Run baseline experiment

```powershell
python scripts\run_baseline.py
```

Outputs are written to `outputs/`.

## GPU note

The current baseline is intentionally CPU-first because the geometry workload is small and transparent. A future candidate-scale benchmark can add CuPy acceleration without changing the geometry contract.

## Scientific scope

This is a geometric targeting benchmark. It does **not** model calibrated X-ray spectra, tissue attenuation, scatter, detector response, dose, or full RBYRCT hardware physics.
