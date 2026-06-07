"""Canonical per-FOV output path helpers for the PyStar I/O facade.

This module owns the concrete path construction logic for the stable
``pystar.io`` facade. Userspace should keep importing these helpers from
``pystar.io``; this private module exists only to reduce ``io.py`` coupling
without changing any path or filename contract.
"""

from pathlib import Path


def _build_fov_output_paths(base_dir: Path, fov_id: int) -> dict[str, Path]:
    """Return canonical per-FOV output paths without mutating the filesystem."""

    fov_root = base_dir / f"Position{fov_id}" / "output_pystar"
    return {
        "root": fov_root,
        "transforms": fov_root / "transforms",
        "spots": fov_root / "spots",
        "extraction": fov_root / "extraction",
        "decoded": fov_root / "decoded",
        "qc": fov_root / "qc_reports",
        "cleaned": fov_root / "clean_data",
    }


def get_fov_output_structure(base_dir: Path, fov_id: int) -> dict[str, Path]:
    """
    统一管理目录结构的逻辑。
    Good Taste: 如果你想改文件夹名，只改这里一行，全项目生效。
    """
    dirs = _build_fov_output_paths(base_dir, fov_id)

    for p in dirs.values():
        try:
            p.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            # Benchmark/replay bundles may intentionally replace canonical stage
            # directories (for example clean_data/) with symlinks to frozen bundle
            # inputs. Treat an existing symlink-to-directory as satisfying the
            # path contract instead of crashing while touching unrelated sibling
            # paths such as transforms/.
            if p.is_symlink() and p.exists() and p.is_dir():
                continue
            raise

    return dirs


def get_transform_manifest_path(base_dir: Path, fov_id: int) -> Path:
    paths = _build_fov_output_paths(base_dir, fov_id)
    return paths["transforms"] / f"transforms_fov_{fov_id}.npy"


def get_provenance_summary_path(base_dir: Path, fov_id: int) -> Path:
    paths = _build_fov_output_paths(base_dir, fov_id)
    return paths["qc"] / "provenance_summary.md"


def get_flow_3d_sidecar_filename(fov_id: int, round_id: int) -> str:
    return f"transforms_fov_{fov_id}_round_{round_id}_flow_3d.npy"
