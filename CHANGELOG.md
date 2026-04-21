# Changelog

This file tracks notable updates to `PyStar`.

## 2026-04-21

### Python-native release prep

- Synced the current Python-native pipeline modules from `pystar-next` into the legacy publishing repository while preserving the legacy repo layout.
- Added required shared support files for the synced runtime (`pystar/extraction_utils.py`, `pystar/tiling.py`, `pystar/matlab_*`, `matlab_runtime/`, `sitecustomize.py`, and `scripts/check_matlab_engine.py`).
- Updated `config/experiment_config.yaml`, `scripts/run_pystar.sh`, `pyproject.toml`, and `README.md` so the supported workflow is the Python-native pipeline, while MATLAB invocation remains explicitly experimental and unsupported.

## 2026-03-05

### Leica sync update (`4b08282`)

- Synced core Leica-related updates into `pystar` modules: `infrastructure.py`, `decoding.py`, `mining.py`, `registration.py`, `spot_finding.py`, `visualization.py`, and `io.py`.
- Added `pystar/decoding_rules.py` for rule-driven decoding flow support.
- Confirmed package health after sync using `py_compile`, `python -m build --no-isolation`, and module import smoke checks.
