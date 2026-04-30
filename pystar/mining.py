# pystar/mining.py
import json
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from importlib import import_module
from pathlib import Path
from .infrastructure import ExperimentConfig
from .extraction_utils import (
    coords_within_transform_scope,
    extract_box_sum_integer,
    extract_signal_volume,
    get_transform_scope,
    map_spot_coordinates,
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
    merge_matlab_session_lifecycle_summaries,
    summarize_matlab_boundary_traces,
)
from .matlab_extraction import MATLABExtractionBackend
# visualization 模块保留引用，按需导入即可


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _write_backend_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")

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

    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.loader = ImageLoader(config)
        self._matlab_backend: Optional[MATLABExtractionBackend] = None

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
            self._matlab_backend = MATLABExtractionBackend(self.cfg)
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

    def _materialize_round_transform(self, fov_id: int, round_id: int, transform_data: dict[str, Any]) -> dict[str, Any]:
        base_dir = Path(self.cfg.pipeline.output.directory)
        return materialize_round_transform_entry(base_dir, fov_id, round_id, transform_data)

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
        transforms: dict[Any, Any],
        contract: dict[str, Any],
    ) -> dict[str, Any] | None:
        delivered_coverage = contract['delivered_coverage']
        resolved_scope: dict[str, Any] | None = None
        missing_rounds: list[int] = []

        for round_id, transform_data in transforms.items():
            if not isinstance(round_id, int):
                continue
            if not isinstance(transform_data, dict) or 'global_shift_3d' not in transform_data:
                continue

            scope_metadata = get_transform_scope(transform_data)
            if scope_metadata is None:
                missing_rounds.append(int(round_id))
                continue
            if scope_metadata['coverage_mode'] != delivered_coverage:
                raise ValueError(
                    f"FOV {fov_id} round {round_id} scope metadata reports {scope_metadata['coverage_mode']!r}, "
                    f"but release_contract delivered_coverage is {delivered_coverage!r}"
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
        transform_data: dict[str, Any],
        transform_application_mode: str,
    ) -> None:
        if transform_application_mode != 'image_warp':
            return

        if isinstance(transform_data.get('flow_2d'), np.ndarray):
            raise ValueError(
                f"Round {round_id} delivered flow_2d, but image_warp mainline only supports flow_3d"
            )

        is_reference_round = bool(transform_data.get('is_reference_round', False))
        if transform_data.get('flow_3d') is None and not is_reference_round:
            raise ValueError(
                f"Round {round_id} is missing flow_3d. image_warp is the Phase 1 RC mainline and does not silently downgrade."
            )

    def _extract_intensities_for_channel(
        self,
        *,
        img_vol: Any,
        ref_coords: Any,
        transform_data: dict[str, Any],
        box_size: tuple[int, int, int],
        transform_application_mode: str,
        fov_id: int,
        round_id: int,
        channel_id: int,
    ) -> tuple[Any, Optional[dict[str, Any]]]:
        """Extract one `(N_spots,)` intensity vector for one round/channel.

        `img_vol` is a cleaned moving-round volume. `ref_coords` are already
        filtered to any legal transform scope. The returned metadata is `None`
        for native extraction and a MATLAB boundary/provenance record for MATLAB
        extraction.
        """
        expected_semantics = self._expected_field_semantics()
        provider = self.cfg.pipeline.extraction.provider

        if provider == 'native':
            if transform_application_mode == 'coordinate_mapping':
                target_coords = map_spot_coordinates(
                    ref_coords,
                    transform_data,
                    expected_field_semantics=expected_semantics,
                )
                return extract_box_sum_integer(img_vol, target_coords, box_size), None

            return (
                extract_signal_volume(
                    img_vol,
                    ref_coords,
                    transform_data,
                    box_size,
                    transform_application_mode,
                    expected_field_semantics=expected_semantics,
                ),
                None,
            )

        backend = self._get_matlab_backend()
        if transform_application_mode == 'coordinate_mapping':
            target_coords = map_spot_coordinates(
                ref_coords,
                transform_data,
                expected_field_semantics=expected_semantics,
            )
            result = backend.extract_intensities(
                img_vol,
                target_coords,
                fov_id=fov_id,
                round_id=round_id,
                channel_id=channel_id,
                box_size=box_size,
                transform_application_mode=transform_application_mode,
            )
            return result['intensities'], result.get('backend_metadata')

        warped_volume = warp_volume_to_reference(
            img_vol,
            transform_data,
            expected_field_semantics=expected_semantics,
        )
        result = backend.extract_intensities(
            warped_volume,
            ref_coords,
            fov_id=fov_id,
            round_id=round_id,
            channel_id=channel_id,
            box_size=box_size,
            transform_application_mode=transform_application_mode,
        )
        return result['intensities'], result.get('backend_metadata')

    def mine_fov(self, fov_id: int):
        """Run signal extraction for every configured round/channel in one FOV.

        The method validates transform release contracts before touching image
        data, keeps tile-local coordinates inside the delivered region, and
        leaves out-of-scope rows as zeros in the final tensor. That makes scope
        effects explicit in the saved matrix instead of silently extrapolating
        deformation fields.
        """
        print(f"[{'='*20} Mining FOV {fov_id} {'='*20}]")
        base_dir = Path(self.cfg.pipeline.output.directory)
        paths = get_fov_output_structure(base_dir, fov_id)
        # 1. Load Metadata & Transforms
        spots_df = pd.read_csv(paths["spots"] / f"spots_fov_{fov_id}.csv")
        transforms = self._load_transforms(fov_id)
        
        ref_coords = spots_df[['z', 'y', 'x']].values.astype(np.float32)
        n_spots = len(ref_coords)

        # Filters channels
        roles = self.cfg.dataset.channel_roles
        all_channels = sorted(list(roles.keys()))
        channels = [c for c in all_channels if roles.get(c) == 'seq']
        
        print(f" [Miner] Channels to extract: {channels}")
        
        rounds = sorted(list(self.cfg.dataset.round_structure.keys()))
        
        # Pre-allocate
        intensity_matrix = np.zeros((n_spots, len(rounds), len(channels)), dtype=np.float32)
        
        # Box Size
        box_vals = self.cfg.pipeline.extraction.integration_box
        box_size: tuple[int, int, int] = (int(box_vals[0]), int(box_vals[1]), int(box_vals[2]))
        transform_application_mode = self.cfg.pipeline.extraction.transform_application_mode
        extraction_provider = self.cfg.pipeline.extraction.provider
        backend_records: list[dict[str, Any]] = []
        scope_contract = self._validate_scope_contract(fov_id, transforms)
        scope_metadata = self._resolve_scope_metadata(fov_id, transforms, scope_contract)
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
        if transform_application_mode == 'image_warp':
            self._validate_image_warp_contract(fov_id, transforms)

        # 2. Main Loop
        # 优化点：外层循环是 Round，内层是 Channel。
        # 我们在这里引入 tqdm 显示总进度
        total_steps = len(rounds) * len(channels)
        
        with tqdm(total=total_steps, desc="Extracting Signals") as pbar:
            for r_idx, r_id in enumerate(rounds):
                # Pre-calculate coordinates for this round ONCE
                if r_id not in transforms:
                    raise KeyError(f"Missing transform entry for round {r_id} in FOV {fov_id} transform manifest")

                trans_data = self._materialize_round_transform(fov_id, r_id, transforms[r_id])
                self._validate_round_transform_for_mode(r_id, trans_data, transform_application_mode)

                current_round_channels = self.cfg.dataset.round_structure[r_id]
                
                for c_idx, c_id in enumerate(channels):
                    if c_id not in current_round_channels:
                        pbar.update(1)
                        continue

                    # Load Image - 这是主要的 IO 开销
                    # 确保是 clean data
                    img_vol = self.loader.load_clean_image(fov_id, r_id, c_id) 

                    vals, backend_metadata = self._extract_intensities_for_channel(
                        img_vol=img_vol,
                        ref_coords=in_scope_coords,
                        transform_data=trans_data,
                        box_size=box_size,
                        transform_application_mode=transform_application_mode,
                        fov_id=fov_id,
                        round_id=r_id,
                        channel_id=c_id,
                    )
                    if isinstance(backend_metadata, dict):
                        backend_records.append(backend_metadata)
                    
                    intensity_matrix[in_scope_mask, r_idx, c_idx] = vals
                    
                    # 显式删除引用，帮助 GC 
                    del img_vol
                    pbar.update(1)

                del trans_data

        # 4. Save
        out_name = paths["extraction"] / f"intensity_matrix_fov_{fov_id}.npy"
        persistence_started = time.perf_counter()
        np.save(out_name, intensity_matrix)
        if extraction_provider == 'matlab' and backend_records:
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
            _write_backend_metadata(
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
        total_intensity = matrix.sum(axis=(1, 2))
        top_indices = np.argsort(total_intensity)[-5:] 
        random_indices = np.random.choice(len(matrix), 5, replace=False)
        selected_indices = np.concatenate([top_indices, random_indices])
            
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
        flat_mat = matrix[:n_debug].reshape(n_debug, -1)
        df_debug = spots_df.iloc[:n_debug].copy()
        df_vals = pd.DataFrame(flat_mat, columns=pd.Index(cols), index=df_debug.index)
        final = pd.concat([df_debug, df_vals], axis=1)
        base_dir = Path(self.cfg.pipeline.output.directory)
        paths = get_fov_output_structure(base_dir, fov_id)
        final.to_csv(paths["extraction"] / f"debug_intensities_fov_{fov_id}.csv", index=False)
