"""Shared Stage18C MATLAB tile worker lifecycle telemetry contract.

This module is intentionally tiny: the executor emits these fields and the
registration diagnostics sidecar validates/summarizes the same field set.  Keep
the names in one place so Stage18C telemetry cannot drift between the executor
seam and the persistence owner.
"""

from __future__ import annotations


WORKER_LIFECYCLE_MS_FIELDS = (
    "backend_construct_ms",
    "matlab_session_start_or_attach_ms",
    "runtime_validation_ms",
    "matlab_addpath_or_bootstrap_ms",
    "input_staging_ms",
    "matlab_call_ms",
    "mat_output_load_ms",
    "result_validation_ms",
    "backend_close_ms",
    "total_tile_wall_ms",
)


WORKER_LIFECYCLE_REQUIRED_FIELDS = (
    "worker_process_pid",
    "tile_index",
    *WORKER_LIFECYCLE_MS_FIELDS,
)
