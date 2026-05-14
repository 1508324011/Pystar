# Changelog

This file tracks notable updates to `PyStar`.

## 2026-05-13

### MATLAB provider release-contract policy

- Promoted MATLAB provider release-contract policy from blanket `debug_only` downgrade to artifact-contract-based validation for `image_warp`: MATLAB preprocessing/registration/spot-finding/extraction provenance can now remain `valid` when runtime manifests, transform artifacts, `flow_3d` sidecars, scope metadata, field semantics, and schema contracts validate.
- Kept fail-loud behavior and the strict extraction gate: `coordinate_mapping` remains diagnostic/debug-only, and `image_warp` extraction still requires `release_gate.status == "valid"` with no MATLAB-to-native silent fallback.

## 2026-04-29

### Native Max3D and configuration refresh

- Synced the current PyStar runtime modules from `pystar-next`, including the native Max3D regional-max spot-finding baseline behind the existing `algorithm: peak_local_max` compatibility token.
- Refreshed `config/experiment_config.yaml` so the pipeline section follows the current Exp2-derived Python-native reference parameters while preserving the existing example layout for dataset, codebook, round structure, and site paths.
- Added `config/README.md` with parameter explanations for the example config and documented the then-current MATLAB provider validation posture; as of 2026-05-13, MATLAB provider release validity is artifact-contract based for `image_warp`.

## 2026-04-21

### Python-native release prep

- Synced the current Python-native pipeline modules from `pystar-next` into the legacy publishing repository while preserving the legacy repo layout.
- Added required shared support files for the synced runtime (`pystar/extraction_utils.py`, `pystar/tiling.py`, `pystar/matlab_*`, `matlab_runtime/`, `sitecustomize.py`, and `scripts/check_matlab_engine.py`).
- Updated `config/experiment_config.yaml`, `scripts/run_pystar.sh`, `pyproject.toml`, and `README.md` around the Python-native default workflow and MATLAB provider seam. The MATLAB provider release policy was later promoted to artifact-contract-based `image_warp` validation on 2026-05-13.

## 2026-03-05

### Leica sync update (`4b08282`)

- Synced core Leica-related updates into `pystar` modules: `infrastructure.py`, `decoding.py`, `mining.py`, `registration.py`, `spot_finding.py`, `visualization.py`, and `io.py`.
- Added `pystar/decoding_rules.py` for rule-driven decoding flow support.
- Confirmed package health after sync using `py_compile`, `python -m build --no-isolation`, and module import smoke checks.
