# pystar/mining.py
import os
import time
from contextlib import contextmanager
from json import loads
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from importlib import import_module
from pathlib import Path
from ._artifact_schemas import (
    SpotTableSchema,
    build_intensity_matrix_spec,
    build_intensity_matrix_metadata_payload,
    build_spot_row_lineage,
    intensity_matrix_metadata_expected_description,
    intensity_matrix_metadata_path,
    spot_row_lineage_from_intensity_metadata_payload,
    validate_intensity_matrix,
    validate_intensity_matrix_consumer_contract,
    validate_intensity_matrix_metadata_payload,
    validate_spot_row_lineage_consumer_contract,
    validate_spot_table,
    wrap_payload_read_error,
    wrap_table_read_error,
)
from .infrastructure import ExperimentConfig
from .extraction_utils import (
    RoundExtractionTransformPlan,
    _CoordinateMappingSamplingPlan,
    _ImageWarpSamplingPlan,
    _build_coordinate_mapping_sampling_plan,
    _build_image_warp_sampling_plan,
    build_round_extraction_transform_plan,
    coords_within_transform_scope,
    get_transform_scope,
    map_spot_coordinates,
    require_coords_within_transform_scope,
    warp_volume_to_reference,
)
from .io import (
    ImageLoader,
    get_matlab_stage_contract,
    load_transform_manifest,
    materialize_round_transform_entry,
    validate_scope_contract,
)
from .io import get_fov_output_structure
from .matlab_engine_bootstrap import (
    MatlabSharedSessionOwner,
    merge_matlab_session_lifecycle_summaries,
    summarize_matlab_boundary_traces,
)
from .matlab_extraction import MATLABExtractionBackend
from .serialization import write_backend_metadata
# visualization 模块保留引用，按需导入即可


EXTRACTION_HOT_PATH_PROFILE_SCHEMA_NAME = "pystar_extraction_hot_path_profile"
EXTRACTION_HOT_PATH_PROFILE_SCHEMA_VERSION = 1
_EXTRACTION_PROFILE_BUCKETS = (
    "spot_table_processing",
    "transform_manifest_load",
    "round_transform_materialization",
    "round_transform_plan_build",
    "image_read",
    "coordinate_or_warp_preparation",
    "interpolation_or_sampling",
    "write_outputs",
)

_SUPPORTED_EXTRACTION_PROVIDERS = ("native", "matlab")
_SUPPORTED_TRANSFORM_APPLICATION_MODES = ("coordinate_mapping", "image_warp")
_SUPPORTED_EXTRACTION_ROUTES = frozenset(
    (provider, mode)
    for provider in _SUPPORTED_EXTRACTION_PROVIDERS
    for mode in _SUPPORTED_TRANSFORM_APPLICATION_MODES
)


def _expected_values(values: tuple[str, ...]) -> str:
    return ", ".join(repr(value) for value in values)


def _validate_extraction_string(value: object, *, field_name: str, expected: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a string; expected one of: {_expected_values(expected)}"
        )
    normalized = value
    if normalized not in expected:
        raise ValueError(
            f"Unsupported {field_name}: {value!r}. Expected one of: {_expected_values(expected)}"
        )
    return normalized


@dataclass(frozen=True)
class _ExtractionRoute:
    """Internal provider/mode decision for one signal-extraction run."""

    provider: str
    transform_application_mode: str

    @property
    def is_native(self) -> bool:
        return self.provider == "native"

    @property
    def is_matlab(self) -> bool:
        return self.provider == "matlab"

    @property
    def is_coordinate_mapping(self) -> bool:
        return self.transform_application_mode == "coordinate_mapping"

    @property
    def is_image_warp(self) -> bool:
        return self.transform_application_mode == "image_warp"

    @property
    def uses_native_sampling_plan(self) -> bool:
        return self.is_native

    @property
    def operation_label(self) -> str:
        if self.is_image_warp:
            return "image_warp extraction"
        return "coordinate_mapping extraction"


def _resolve_extraction_route(config: ExperimentConfig) -> _ExtractionRoute:
    """Resolve and validate extraction provider/mode once for SignalMiner."""

    extraction_cfg = getattr(getattr(config, "pipeline", None), "extraction", None)
    return _build_extraction_route(
        provider_value=getattr(extraction_cfg, "provider", None),
        transform_application_mode_value=getattr(extraction_cfg, "transform_application_mode", None),
    )


def _build_extraction_route(
    *,
    provider_value: object,
    transform_application_mode_value: object,
) -> _ExtractionRoute:
    """Build a validated internal route from raw provider/mode values."""

    provider = _validate_extraction_string(
        provider_value,
        field_name="extraction provider",
        expected=_SUPPORTED_EXTRACTION_PROVIDERS,
    )
    transform_application_mode = _validate_extraction_string(
        transform_application_mode_value,
        field_name="transform application mode",
        expected=_SUPPORTED_TRANSFORM_APPLICATION_MODES,
    )
    if (provider, transform_application_mode) not in _SUPPORTED_EXTRACTION_ROUTES:
        message = (
            f"Unsupported extraction route: provider={provider!r}, "
            f"transform_application_mode={transform_application_mode!r}"
        )
        raise ValueError(
            message
        )
    return _ExtractionRoute(
        provider=provider,
        transform_application_mode=transform_application_mode,
    )


class _ExtractionHotPathProfiler:
    """Default-off extraction profiling guardrail for future optimization work."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._events: list[dict[str, Any]] = []

    @contextmanager
    def record(self, bucket: str, **details: Any) -> Iterator[None]:
        if bucket not in _EXTRACTION_PROFILE_BUCKETS:
            raise ValueError(f"Unknown extraction profiling bucket: {bucket!r}")
        if not self.enabled:
            yield
            return

        started = time.perf_counter()
        try:
            yield
        finally:
            self._events.append(
                {
                    "bucket": bucket,
                    "elapsed_wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    "details": {key: value for key, value in details.items() if value is not None},
                }
            )

    def build_payload(self, *, fov_id: int) -> dict[str, Any]:
        buckets: dict[str, dict[str, Any]] = {
            bucket: {"count": 0, "total_elapsed_wall_ms": 0.0}
            for bucket in _EXTRACTION_PROFILE_BUCKETS
        }
        for event in self._events:
            bucket = str(event["bucket"])
            elapsed = float(event["elapsed_wall_ms"])
            buckets[bucket]["count"] = int(buckets[bucket]["count"]) + 1
            buckets[bucket]["total_elapsed_wall_ms"] = round(
                float(buckets[bucket]["total_elapsed_wall_ms"]) + elapsed,
                3,
            )

        return {
            "schema_name": EXTRACTION_HOT_PATH_PROFILE_SCHEMA_NAME,
            "schema_version": EXTRACTION_HOT_PATH_PROFILE_SCHEMA_VERSION,
            "fov_id": int(fov_id),
            "enabled": self.enabled,
            "buckets": buckets,
            "events": list(self._events),
        }


def _extraction_profile_enabled(config: ExperimentConfig) -> bool:
    extraction_cfg = getattr(getattr(config, "pipeline", None), "extraction", None)
    config_flag = any(
        bool(getattr(extraction_cfg, attr, False))
        for attr in ("profile_hot_path", "profiling_enabled", "enable_hot_path_profiling")
    )
    env_flag = os.environ.get("PYSTAR_EXTRACTION_PROFILE", "").strip().lower() in {"1", "true", "yes", "on"}
    return bool(config_flag or env_flag)

class SignalMiner:
    """Extract per-round sequencing-channel intensities at detected spots.

    Mining is the bridge between geometry and decoding. It reads reference-frame
    spot coordinates from `spots/spots_fov_<id>.csv`, replays the registration
    transform for every imaging round/channel, and writes an intensity tensor of
    shape `(N_spots, N_rounds, N_seq_channels)` to `extraction/`.

    All spot coordinates are `z, y, x` pixels in the reference round. Depending
    on `pipeline.extraction.transform_application_mode`, PyStar either maps
    those coordinates into the moving image (`coordinate_mapping`) or first
    warps the moving image into reference space (`image_warp`) and then samples
    the original coordinates. Native and MATLAB extraction providers share the
    same transform and scope checks so provider differences isolate integration
    semantics rather than contract handling.
    """

    def __init__(self, config: ExperimentConfig, matlab_session_owner: Optional[MatlabSharedSessionOwner] = None):
        self.cfg = config
        self.loader = ImageLoader(config)
        self._matlab_session_owner = matlab_session_owner
        self._matlab_backend: Optional[MATLABExtractionBackend] = None
        self._last_extraction_profile: dict[str, Any] | None = None

    def close(self) -> None:
        """Release the optional MATLAB extraction backend session."""
        if self._matlab_backend is None:
            return
        try:
            self._matlab_backend.close()
        finally:
            self._matlab_backend = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def _get_matlab_backend(self) -> MATLABExtractionBackend:
        if self._matlab_backend is None:
            self._matlab_backend = MATLABExtractionBackend(
                self.cfg,
                matlab_session_owner=self._matlab_session_owner,
            )
        return self._matlab_backend

    def _expected_field_semantics(self) -> dict[str, str]:
        return self.cfg.pipeline.field_semantics.as_dict()
        
    def _load_transforms(self, fov_id: int) -> dict[Any, Any]:
        base_dir = Path(self.cfg.pipeline.output.directory)
        return load_transform_manifest(
            base_dir,
            fov_id,
            load_provenance=True,
            hydrate_flow_3d=False,
        )

    def _materialize_round_transform(
        self,
        fov_id: int,
        round_id: int,
        transform_data: Mapping[str, Any],
        *,
        hydrate_flow_3d: bool = True,
    ) -> dict[str, Any]:
        base_dir = Path(self.cfg.pipeline.output.directory)
        return materialize_round_transform_entry(
            base_dir,
            fov_id,
            round_id,
            transform_data,
            hydrate_flow_3d=hydrate_flow_3d,
        )

    def _build_round_transform_plan(
        self,
        fov_id: int,
        round_id: int,
        transform_data: Mapping[str, Any],
        *,
        hydrate_flow_3d: bool = True,
        profiler: _ExtractionHotPathProfiler | None = None,
    ) -> RoundExtractionTransformPlan:
        profiler = profiler or _ExtractionHotPathProfiler(enabled=False)
        with profiler.record(
            "round_transform_materialization",
            round_id=int(round_id),
            hydrate_flow_3d=bool(hydrate_flow_3d),
        ):
            materialized = self._materialize_round_transform(
                fov_id,
                round_id,
                transform_data,
                hydrate_flow_3d=hydrate_flow_3d,
            )
        with profiler.record(
            "round_transform_plan_build",
            round_id=int(round_id),
            hydrate_flow_3d=bool(hydrate_flow_3d),
        ):
            return build_round_extraction_transform_plan(
                fov_id=fov_id,
                round_id=round_id,
                transform_data=materialized,
                source_transform_data=transform_data,
            )

    def _validate_scope_contract(self, fov_id: int, transforms: dict[Any, Any]) -> dict[str, Any]:
        provenance = transforms.get('_provenance')
        if not isinstance(provenance, dict):
            raise ValueError(
                f"FOV {fov_id} transform manifest is missing _provenance; explicit scope metadata is required before extraction"
            )

        contract = provenance.get('release_contract')
        if not isinstance(contract, dict):
            raise ValueError(
                f"FOV {fov_id} transform manifest is missing release_contract; explicit scope metadata is required before extraction"
            )

        scope_contract = validate_scope_contract(
            contract,
            expected_scope_mode=self.cfg.pipeline.scope_mode,
        )
        print(
            f" [Miner] Scope contract: requested={scope_contract['requested_scope_mode']} | delivered={scope_contract['delivered_coverage']} | status={scope_contract['scope_status']}"
        )

        if not scope_contract['scope_valid']:
            raise ValueError(
                f"FOV {fov_id} scope contract mismatch: requested {scope_contract['requested_scope_mode']!r} but registration delivered {scope_contract['delivered_coverage']!r}; scope_status={scope_contract['scope_status']!r}. Extraction will not proceed."
            )

        if scope_contract['scope_status'] != 'valid':
            raise ValueError(
                f"FOV {fov_id} scope contract is not extraction-legal: scope_status={scope_contract['scope_status']!r}"
            )

        return contract

    def _resolve_scope_metadata(
        self,
        fov_id: int,
        transforms: Mapping[Any, Any],
        contract: dict[str, Any],
    ) -> dict[str, Any] | None:
        delivered_coverage = contract['delivered_coverage']
        resolved_scope: dict[str, Any] | None = None
        missing_rounds: list[int] = []

        for round_id, transform_data in transforms.items():
            if not isinstance(round_id, int):
                continue
            if isinstance(transform_data, RoundExtractionTransformPlan):
                scope_metadata = transform_data.scope_public_payload()
            elif isinstance(transform_data, dict) and 'global_shift_3d' in transform_data:
                scope_metadata = get_transform_scope(transform_data)
            else:
                continue
            if scope_metadata is None:
                missing_rounds.append(int(round_id))
                continue
            if scope_metadata['coverage_mode'] != delivered_coverage:
                raise ValueError(
                    f"FOV {fov_id} round {round_id} scope metadata reports "
                    f"{scope_metadata['coverage_mode']!r}, but release_contract delivered_coverage is "
                    f"{delivered_coverage!r}"
                )

            normalized_scope = dict(scope_metadata)
            if resolved_scope is None:
                resolved_scope = normalized_scope
                continue
            if normalized_scope != resolved_scope:
                raise ValueError(
                    f"FOV {fov_id} transform manifest mixes inconsistent round _scope metadata; round {round_id} differs from earlier rounds"
                )

        if delivered_coverage == 'tile_local':
            if missing_rounds:
                raise ValueError(
                    f"FOV {fov_id} tile_local manifest is missing per-round _scope metadata for rounds {sorted(missing_rounds)}"
                )
            if resolved_scope is None:
                raise ValueError(
                    f"FOV {fov_id} tile_local manifest does not contain any persisted _scope metadata"
                )

        return resolved_scope

    def _validate_image_warp_contract(self, fov_id: int, transforms: dict[Any, Any]) -> None:
        provenance = transforms.get('_provenance')
        if not isinstance(provenance, dict):
            raise ValueError(
                f"FOV {fov_id} transform manifest is missing _provenance; image_warp extraction requires explicit runtime metadata"
            )

        contract = provenance.get('release_contract')
        if not isinstance(contract, dict):
            raise ValueError(
                f"FOV {fov_id} transform manifest is missing release_contract; image_warp extraction requires explicit runtime metadata"
            )

        release_gate = contract.get('release_gate')
        if not isinstance(release_gate, dict):
            raise ValueError(
                f"FOV {fov_id} transform manifest is missing release_gate metadata; image_warp extraction cannot validate legality"
            )

        status = release_gate.get('status')
        if status != 'valid':
            reasons = release_gate.get('reasons') or []
            raise ValueError(
                f"FOV {fov_id} transform contract is not a valid image_warp Phase 1 RC artifact: status={status!r}, reasons={reasons}"
            )

    def _validate_round_transform_for_mode(
        self,
        round_id: int,
        transform_data: dict[str, Any] | RoundExtractionTransformPlan,
        transform_application_mode: str,
    ) -> None:
        if transform_application_mode != 'image_warp':
            return

        if isinstance(transform_data, RoundExtractionTransformPlan):
            data = transform_data.legacy_transform_data()
        else:
            data = transform_data

        if isinstance(data.get('flow_2d'), np.ndarray):
            raise ValueError(
                f"Round {round_id} delivered flow_2d, but image_warp mainline only supports flow_3d"
            )

        is_reference_round = bool(data.get('is_reference_round', False))
        if data.get('flow_3d') is None and not is_reference_round:
            raise ValueError(
                f"Round {round_id} is missing flow_3d. image_warp is the Phase 1 RC mainline and does not silently downgrade."
            )

    def _extract_native_intensities_for_channel(
        self,
        *,
        img_vol: Any,
        ref_coords: Any,
        transform_data: dict[str, Any] | RoundExtractionTransformPlan,
        box_size: tuple[int, int, int],
        route: _ExtractionRoute,
        round_id: int,
        channel_id: int,
        profiler: _ExtractionHotPathProfiler,
        coordinate_mapping_sampling_plan: _CoordinateMappingSamplingPlan | None,
        image_warp_sampling_plan: _ImageWarpSamplingPlan | None,
    ) -> tuple[Any, None]:
        expected_semantics = self._expected_field_semantics()

        if route.is_coordinate_mapping:
            if coordinate_mapping_sampling_plan is None:
                with profiler.record("coordinate_or_warp_preparation", round_id=round_id, channel_id=channel_id):
                    sampling_plan = _build_coordinate_mapping_sampling_plan(
                        img_shape=tuple(img_vol.shape),
                        ref_coords=ref_coords,
                        transform_data=transform_data,
                        box_size=box_size,
                        expected_field_semantics=expected_semantics,
                    )
            else:
                sampling_plan = coordinate_mapping_sampling_plan
            with profiler.record("interpolation_or_sampling", round_id=round_id, channel_id=channel_id):
                return sampling_plan.sample(img_vol), None

        if route.is_image_warp:
            if image_warp_sampling_plan is None:
                with profiler.record("coordinate_or_warp_preparation", round_id=round_id, channel_id=channel_id):
                    sampling_plan = _build_image_warp_sampling_plan(
                        img_shape=tuple(img_vol.shape),
                        ref_coords=ref_coords,
                        transform_data=transform_data,
                        box_size=box_size,
                        expected_field_semantics=expected_semantics,
                    )
            else:
                sampling_plan = image_warp_sampling_plan
            with profiler.record("interpolation_or_sampling", round_id=round_id, channel_id=channel_id):
                return sampling_plan.sample(img_vol), None

        raise ValueError(f'Unsupported transform application mode: {route.transform_application_mode}')

    def _extract_matlab_intensities_for_channel(
        self,
        *,
        img_vol: Any,
        ref_coords: Any,
        transform_data: dict[str, Any] | RoundExtractionTransformPlan,
        box_size: tuple[int, int, int],
        route: _ExtractionRoute,
        fov_id: int,
        round_id: int,
        channel_id: int,
        profiler: _ExtractionHotPathProfiler,
    ) -> tuple[Any, Optional[dict[str, Any]]]:
        expected_semantics = self._expected_field_semantics()
        backend = self._get_matlab_backend()

        if route.is_coordinate_mapping:
            with profiler.record("coordinate_or_warp_preparation", round_id=round_id, channel_id=channel_id):
                target_coords = map_spot_coordinates(
                    ref_coords,
                    transform_data,
                    expected_field_semantics=expected_semantics,
                )
            with profiler.record("interpolation_or_sampling", round_id=round_id, channel_id=channel_id):
                result = backend.extract_intensities(
                    img_vol,
                    target_coords,
                    fov_id=fov_id,
                    round_id=round_id,
                    channel_id=channel_id,
                    box_size=box_size,
                    transform_application_mode=route.transform_application_mode,
                )
            return result['intensities'], result.get('backend_metadata')

        if route.is_image_warp:
            with profiler.record("coordinate_or_warp_preparation", round_id=round_id, channel_id=channel_id):
                warped_volume = warp_volume_to_reference(
                    img_vol,
                    transform_data,
                    expected_field_semantics=expected_semantics,
                )
            with profiler.record("interpolation_or_sampling", round_id=round_id, channel_id=channel_id):
                result = backend.extract_intensities(
                    warped_volume,
                    ref_coords,
                    fov_id=fov_id,
                    round_id=round_id,
                    channel_id=channel_id,
                    box_size=box_size,
                    transform_application_mode=route.transform_application_mode,
                )
            return result['intensities'], result.get('backend_metadata')

        raise ValueError(f'Unsupported transform application mode: {route.transform_application_mode}')

    def _extract_intensities_for_channel(
        self,
        *,
        img_vol: Any,
        ref_coords: Any,
        transform_data: dict[str, Any] | RoundExtractionTransformPlan,
        box_size: tuple[int, int, int],
        transform_application_mode: str,
        fov_id: int,
        round_id: int,
        channel_id: int,
        profiler: _ExtractionHotPathProfiler | None = None,
        coordinate_mapping_sampling_plan: _CoordinateMappingSamplingPlan | None = None,
        image_warp_sampling_plan: _ImageWarpSamplingPlan | None = None,
        route: _ExtractionRoute | None = None,
    ) -> tuple[Any, Optional[dict[str, Any]]]:
        """Extract one `(N_spots,)` intensity vector for one round/channel.

        `img_vol` is a cleaned moving-round volume. `ref_coords` are already
        filtered to any legal transform scope. The returned metadata is `None`
        for native extraction and a MATLAB boundary/provenance record for MATLAB
        extraction.
        """
        profiler = profiler or _ExtractionHotPathProfiler(enabled=False)
        resolved_route = route or _resolve_extraction_route(self.cfg)
        if transform_application_mode != resolved_route.transform_application_mode:
            message = (
                "transform_application_mode does not match resolved extraction route: "
                f"got {transform_application_mode!r}, "
                f"expected {resolved_route.transform_application_mode!r}"
            )
            raise ValueError(
                message
            )

        _ = require_coords_within_transform_scope(
            ref_coords,
            transform_data,
            operation=resolved_route.operation_label,
        )

        if resolved_route.is_native:
            return self._extract_native_intensities_for_channel(
                img_vol=img_vol,
                ref_coords=ref_coords,
                transform_data=transform_data,
                box_size=box_size,
                route=resolved_route,
                round_id=round_id,
                channel_id=channel_id,
                profiler=profiler,
                coordinate_mapping_sampling_plan=coordinate_mapping_sampling_plan,
                image_warp_sampling_plan=image_warp_sampling_plan,
            )

        if resolved_route.is_matlab:
            return self._extract_matlab_intensities_for_channel(
                img_vol=img_vol,
                ref_coords=ref_coords,
                transform_data=transform_data,
                box_size=box_size,
                route=resolved_route,
                fov_id=fov_id,
                round_id=round_id,
                channel_id=channel_id,
                profiler=profiler,
            )

        raise ValueError(f"Unsupported extraction provider: {resolved_route.provider!r}")

    def _require_persisted_spot_row_lineage(
        self,
        metadata_payload: dict[str, Any],
        *,
        fov_id: int,
        metadata_path: Path,
        context: str,
    ) -> Any:
        persisted_lineage = spot_row_lineage_from_intensity_metadata_payload(
            metadata_payload,
            fov_id=fov_id,
            path=metadata_path,
            context=context,
        )
        if persisted_lineage is None:
            raise ValueError(
                f"FOV {fov_id} intensity metadata sidecar at {metadata_path} is missing spot_row_lineage during "
                f"{context}; newly persisted mining artifacts must carry explicit row-lineage metadata"
            )
        return persisted_lineage

    def mine_fov(self, fov_id: int):
        """Run signal extraction for every configured round/channel in one FOV.

        The method validates transform release contracts before touching image
        data, keeps tile-local coordinates inside the delivered region, and
        leaves out-of-scope rows as zeros in the final tensor. That makes scope
        effects explicit in the saved matrix instead of silently extrapolating
        deformation fields.
        """
        print(f"[{'='*20} Mining FOV {fov_id} {'='*20}]")
        profiler = _ExtractionHotPathProfiler(enabled=_extraction_profile_enabled(self.cfg))
        base_dir = Path(self.cfg.pipeline.output.directory)
        paths = get_fov_output_structure(base_dir, fov_id)
        extraction_route = _resolve_extraction_route(self.cfg)
        transform_application_mode = extraction_route.transform_application_mode
        extraction_provider = extraction_route.provider
        # 1. Load Metadata & Transforms
        spots_path = paths["spots"] / f"spots_fov_{fov_id}.csv"
        spot_expected = SpotTableSchema().expected_description()
        with profiler.record("spot_table_processing"):
            try:
                raw_spots_df = pd.read_csv(spots_path)
            except Exception as exc:
                raise wrap_table_read_error(
                    exc,
                    "spot table",
                    fov_id=fov_id,
                    path=spots_path,
                    context="mining load",
                    expected=spot_expected,
                ) from exc
            spots_df = validate_spot_table(
                raw_spots_df,
                fov_id=fov_id,
                path=spots_path,
                context="mining load",
            )
            spot_row_lineage = build_spot_row_lineage(
                spots_df,
                fov_id=fov_id,
                path=spots_path,
                context="mining row-lineage build",
            )
        with profiler.record("transform_manifest_load"):
            transforms = self._load_transforms(fov_id)
        
        ref_coords = spots_df[['z', 'y', 'x']].values.astype(np.float32)
        n_spots = len(ref_coords)

        # Filters channels
        roles = self.cfg.dataset.channel_roles
        all_channels = sorted(list(roles.keys()))
        channels = [c for c in all_channels if roles.get(c) == 'seq']
        
        print(f" [Miner] Channels to extract: {channels}")
        
        rounds = sorted(list(self.cfg.dataset.round_structure.keys()))
        round_transform_plans: dict[int, RoundExtractionTransformPlan] = {}
        for r_id in rounds:
            if r_id not in transforms:
                raise KeyError(f"Missing transform entry for round {r_id} in FOV {fov_id} transform manifest")
            transform_data = transforms[r_id]
            if not isinstance(transform_data, Mapping):
                raise ValueError(f"FOV {fov_id} transform manifest round {r_id} must be a mapping")
            round_transform_plans[r_id] = self._build_round_transform_plan(
                fov_id,
                r_id,
                transform_data,
                hydrate_flow_3d=False,
                profiler=profiler,
            )
        matrix_spec = build_intensity_matrix_spec(
            fov_id=fov_id,
            n_spots=n_spots,
            rounds=rounds,
            channels=channels,
        )
        out_name = paths["extraction"] / f"intensity_matrix_fov_{fov_id}.npy"
        staged_out_name = out_name.with_name(f"{out_name.name}.tmp")
        if staged_out_name.exists():
            staged_out_name.unlink()

        # Let a staged .npy memmap own the full intensity tensor during
        # extraction, then atomically publish the canonical artifact only
        # after validation. This preserves the public path/format contract
        # while avoiding a long-lived eager RAM allocation for large FOVs.
        intensity_matrix = np.lib.format.open_memmap(
            staged_out_name,
            mode="w+",
            dtype=np.float32,
            shape=matrix_spec.expected_shape,
        )
        intensity_matrix[...] = np.float32(0.0)
        intensity_matrix.flush()
        
        # Box Size
        box_vals = self.cfg.pipeline.extraction.integration_box
        box_size: tuple[int, int, int] = (int(box_vals[0]), int(box_vals[1]), int(box_vals[2]))
        backend_records: list[dict[str, Any]] = []
        scope_contract = self._validate_scope_contract(fov_id, transforms)
        scope_metadata = self._resolve_scope_metadata(fov_id, round_transform_plans, scope_contract)
        scope_transform = None if scope_metadata is None else {'_scope': scope_metadata}
        in_scope_mask = coords_within_transform_scope(ref_coords, scope_transform)
        in_scope_coords = ref_coords[in_scope_mask]
        if scope_metadata is not None and scope_metadata.get('coverage_mode') == 'tile_local':
            in_scope_count = int(in_scope_mask.sum())
            if in_scope_count == 0:
                raise ValueError(
                    f"FOV {fov_id} tile_local scope excludes every detected spot; extraction cannot proceed"
                )
            print(
                f" [Miner] Tile-local scope keeps {in_scope_count}/{n_spots} detected spots inside delivered coverage"
            )
        print(f" [Miner] Extraction provider: {extraction_provider}")
        if extraction_route.is_image_warp:
            self._validate_image_warp_contract(fov_id, transforms)

        # 2. Main Loop
        # 优化点：外层循环是 Round，内层是 Channel。
        # 我们在这里引入 tqdm 显示总进度
        total_steps = len(rounds) * len(channels)
        
        with tqdm(total=total_steps, desc="Extracting Signals") as pbar:
            for r_idx, r_id in enumerate(rounds):
                # Pre-calculate coordinates for this round ONCE
                transform_data = transforms[r_id]
                if not isinstance(transform_data, Mapping):
                    raise ValueError(f"FOV {fov_id} transform manifest round {r_id} must be a mapping")
                round_plan = self._build_round_transform_plan(
                    fov_id,
                    r_id,
                    transform_data,
                    hydrate_flow_3d=True,
                    profiler=profiler,
                )
                self._validate_round_transform_for_mode(r_id, round_plan, transform_application_mode)
                coordinate_mapping_sampling_plans: dict[tuple[int, int, int], _CoordinateMappingSamplingPlan] = {}
                image_warp_sampling_plans: dict[tuple[int, int, int], _ImageWarpSamplingPlan] = {}

                current_round_channels = self.cfg.dataset.round_structure[r_id]
                
                for c_idx, c_id in enumerate(channels):
                    if c_id not in current_round_channels:
                        pbar.update(1)
                        continue

                    # Load Image - 这是主要的 IO 开销
                    # 确保是 clean data
                    with profiler.record("image_read", round_id=r_id, channel_id=c_id):
                        img_vol = self.loader.load_clean_image(fov_id, r_id, c_id)
                    coordinate_mapping_sampling_plan: _CoordinateMappingSamplingPlan | None = None
                    image_warp_sampling_plan: _ImageWarpSamplingPlan | None = None
                    if extraction_route.uses_native_sampling_plan:
                        image_shape = tuple(int(dim) for dim in img_vol.shape)
                        if len(image_shape) != 3:
                            raise ValueError(f"img_vol must be 3D, got shape {img_vol.shape}")
                        plan_key = (image_shape[0], image_shape[1], image_shape[2])
                        if extraction_route.is_coordinate_mapping:
                            coordinate_mapping_sampling_plan = coordinate_mapping_sampling_plans.get(plan_key)
                            if coordinate_mapping_sampling_plan is None:
                                with profiler.record("coordinate_or_warp_preparation", round_id=r_id, channel_id=c_id):
                                    coordinate_mapping_sampling_plan = _build_coordinate_mapping_sampling_plan(
                                        img_shape=plan_key,
                                        ref_coords=in_scope_coords,
                                        transform_data=round_plan,
                                        box_size=box_size,
                                        expected_field_semantics=self._expected_field_semantics(),
                                    )
                                coordinate_mapping_sampling_plans[plan_key] = coordinate_mapping_sampling_plan
                        elif extraction_route.is_image_warp:
                            image_warp_sampling_plan = image_warp_sampling_plans.get(plan_key)
                            if image_warp_sampling_plan is None:
                                with profiler.record("coordinate_or_warp_preparation", round_id=r_id, channel_id=c_id):
                                    image_warp_sampling_plan = _build_image_warp_sampling_plan(
                                        img_shape=plan_key,
                                        ref_coords=in_scope_coords,
                                        transform_data=round_plan,
                                        box_size=box_size,
                                        expected_field_semantics=self._expected_field_semantics(),
                                    )
                                image_warp_sampling_plans[plan_key] = image_warp_sampling_plan

                    vals, backend_metadata = self._extract_intensities_for_channel(
                        img_vol=img_vol,
                        ref_coords=in_scope_coords,
                        transform_data=round_plan,
                        box_size=box_size,
                        transform_application_mode=transform_application_mode,
                        fov_id=fov_id,
                        round_id=r_id,
                        channel_id=c_id,
                        profiler=profiler,
                        coordinate_mapping_sampling_plan=coordinate_mapping_sampling_plan,
                        image_warp_sampling_plan=image_warp_sampling_plan,
                        route=extraction_route,
                    )
                    if isinstance(backend_metadata, dict):
                        backend_records.append(backend_metadata)
                    
                    intensity_matrix[in_scope_mask, r_idx, c_idx] = vals
                    
                    # 显式删除引用，帮助 GC 
                    del img_vol
                    pbar.update(1)

                del round_plan

        # 4. Save
        metadata_path = intensity_matrix_metadata_path(out_name)
        persistence_started = time.perf_counter()
        intensity_matrix.flush()
        intensity_matrix = validate_intensity_matrix(
            intensity_matrix,
            matrix_spec,
            path=out_name,
            context="mining save",
        )
        metadata_payload = build_intensity_matrix_metadata_payload(matrix_spec, spot_row_lineage=spot_row_lineage)
        persisted_spec = validate_intensity_matrix_metadata_payload(
            metadata_payload,
            fov_id=fov_id,
            path=metadata_path,
            context="mining metadata save",
        )
        validate_intensity_matrix_consumer_contract(
            persisted_spec,
            matrix_spec,
            path=metadata_path,
            context="mining metadata save",
            matrix_path=out_name,
        )
        persisted_lineage = self._require_persisted_spot_row_lineage(
            metadata_payload,
            fov_id=fov_id,
            metadata_path=metadata_path,
            context="mining metadata save",
        )
        validate_spot_row_lineage_consumer_contract(
            persisted_lineage,
            spot_row_lineage,
            fov_id=fov_id,
            path=metadata_path,
            context="mining metadata save",
            spot_path=spots_path,
        )
        with profiler.record("write_outputs", artifact="intensity_matrix"):
            if isinstance(intensity_matrix, np.memmap):
                intensity_matrix.flush()
                mmap_handle = getattr(intensity_matrix, "_mmap", None)
                if mmap_handle is not None:
                    mmap_handle.close()
            os.replace(staged_out_name, out_name)
            intensity_matrix = np.load(out_name, allow_pickle=False, mmap_mode="r")
            write_backend_metadata(metadata_path, metadata_payload)
        metadata_expected = intensity_matrix_metadata_expected_description()
        try:
            persisted_metadata = loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise wrap_payload_read_error(
                exc,
                "intensity matrix metadata sidecar",
                fov_id=fov_id,
                path=metadata_path,
                context="mining metadata save",
                expected=metadata_expected,
            ) from exc
        persisted_spec = validate_intensity_matrix_metadata_payload(
            persisted_metadata,
            fov_id=fov_id,
            path=metadata_path,
            context="mining metadata save",
        )
        validate_intensity_matrix_consumer_contract(
            persisted_spec,
            matrix_spec,
            path=metadata_path,
            context="mining metadata save",
            matrix_path=out_name,
        )
        reloaded_lineage = self._require_persisted_spot_row_lineage(
            persisted_metadata,
            fov_id=fov_id,
            metadata_path=metadata_path,
            context="mining metadata save",
        )
        validate_spot_row_lineage_consumer_contract(
            reloaded_lineage,
            spot_row_lineage,
            fov_id=fov_id,
            path=metadata_path,
            context="mining metadata save",
            spot_path=spots_path,
        )
        _ = validate_intensity_matrix(
            intensity_matrix,
            persisted_spec,
            path=out_name,
            context="mining save against metadata sidecar",
        )
        if extraction_route.is_matlab and backend_records:
            boundary_traces = [
                trace
                for record in backend_records
                if isinstance(record, dict)
                for trace in [record.get("boundary_instrumentation")]
                if isinstance(trace, dict)
            ]
            session_summaries = [
                summary
                for record in backend_records
                if isinstance(record, dict)
                for summary in [record.get("session_lifecycle_summary")]
                if isinstance(summary, dict)
            ]
            boundary_summary = summarize_matlab_boundary_traces(boundary_traces) if boundary_traces else None
            persistence_ms = round((time.perf_counter() - persistence_started) * 1000.0, 3)
            if boundary_summary is not None:
                boundary_summary["fov_canonical_persistence_ms"] = persistence_ms
            write_backend_metadata(
                paths["qc"] / f"extraction_backend_fov_{fov_id}.json",
                {
                    "provider": extraction_provider,
                    "matlab_stage_contract": get_matlab_stage_contract(self.cfg, "extraction"),
                    "fov_id": int(fov_id),
                    "transform_application_mode": transform_application_mode,
                    "records": backend_records,
                    "boundary_instrumentation_summary": boundary_summary,
                    "session_lifecycle_summary": merge_matlab_session_lifecycle_summaries(session_summaries) if session_summaries else None,
                },
            )
        self._last_extraction_profile = profiler.build_payload(fov_id=fov_id)
        if profiler.enabled:
            write_backend_metadata(
                paths["qc"] / f"extraction_hot_path_profile_fov_{fov_id}.json",
                self._last_extraction_profile,
            )
        print(f" [Miner] Saved extraction matrix to {out_name.name} | Shape: {intensity_matrix.shape}")
        
        # 5. QC (Optional visualization code kept minimal here for speed)
        self._generate_qc(intensity_matrix, spots_df, rounds, channels, fov_id)

    def _generate_qc(self, matrix, spots_df, rounds, channels, fov_id):
        # 剥离出来的 QC 逻辑，保持主流程清晰
        if not self.cfg.pipeline.qc_images_enabled():
            return

        plot_spot_traces = import_module('pystar.visualization').plot_spot_traces
            
        print(f" [QC] Generating extraction QC plots...")
        base_dir = Path(self.cfg.pipeline.output.directory)
        paths = get_fov_output_structure(base_dir, fov_id)
        qc_dir = paths["qc"]
            
        # Trace Plots
        n_spots = int(len(matrix))
        if n_spots == 0:
            print(" [QC] No spots available for extraction trace QC; skipping trace plot and writing empty debug CSV")
            self._save_debug_csv(matrix, spots_df, rounds, channels, fov_id)
            return

        total_intensity = matrix.sum(axis=(1, 2))
        top_count = min(5, n_spots)
        random_count = min(5, n_spots)
        top_indices = np.argsort(total_intensity)[-top_count:]
        random_indices = np.random.choice(n_spots, random_count, replace=False)
        selected_indices = np.unique(np.concatenate([top_indices, random_indices])).astype(np.int64, copy=False)

        plot_spot_traces(
            matrix, selected_indices, 
            rounds, channels,
            output_path=qc_dir / f"spot_traces_fov_{fov_id}.png"
        )
        # Debug CSV
        self._save_debug_csv(matrix, spots_df, rounds, channels, fov_id)

    def _save_debug_csv(self, matrix, spots_df, rounds, channels, fov_id):
        n_debug = min(100, len(spots_df))
        cols: list[str] = []
        for r in rounds:
            for c in channels:
                cols.append(f"R{r}_C{c}")
        flat_mat = matrix[:n_debug].reshape(n_debug, len(cols))
        df_debug = spots_df.iloc[:n_debug].copy()
        df_vals = pd.DataFrame(flat_mat, columns=pd.Index(cols), index=df_debug.index)
        final = pd.concat([df_debug, df_vals], axis=1)
        base_dir = Path(self.cfg.pipeline.output.directory)
        paths = get_fov_output_structure(base_dir, fov_id)
        final.to_csv(paths["extraction"] / f"debug_intensities_fov_{fov_id}.csv", index=False)
