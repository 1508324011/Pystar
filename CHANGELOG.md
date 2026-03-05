# Changelog

This file tracks notable updates to `PyStar`.

## 2026-03-05

### Leica sync update (`4b08282`)

- Synced core Leica-related updates into `pystar` modules: `infrastructure.py`, `decoding.py`, `mining.py`, `registration.py`, `spot_finding.py`, `visualization.py`, and `io.py`.
- Added `pystar/decoding_rules.py` for rule-driven decoding flow support.
- Confirmed package health after sync using `py_compile`, `python -m build --no-isolation`, and module import smoke checks.
