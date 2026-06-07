from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pystar._performance_summary import parse_fov_ids, write_performance_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Stage14 per-FOV PyStar performance telemetry sidecars into "
            "run-level JSON and Markdown summaries."
        )
    )
    _ = parser.add_argument(
        "--base-dir",
        required=True,
        type=Path,
        help="Pipeline output base directory containing Position*/output_pystar/ trees.",
    )
    _ = parser.add_argument(
        "--fovs",
        default=None,
        help="Optional comma/range FOV list to include, e.g. '1-4,9'. Missing telemetry is reported as absent.",
    )
    _ = parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON summary path. Defaults to <base-dir>/performance_summary.json.",
    )
    _ = parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output Markdown summary path. Defaults to <base-dir>/performance_summary.md.",
    )
    args = parser.parse_args()

    fov_ids = parse_fov_ids(cast(str | None, args.fovs))
    json_path, markdown_path = write_performance_summary(
        base_dir=cast(Path, args.base_dir),
        fov_ids=fov_ids if fov_ids else None,
        output_json_path=cast(Path | None, args.output_json),
        output_markdown_path=cast(Path | None, args.output_md),
    )
    print(f"Performance telemetry JSON summary: {json_path}")
    print(f"Performance telemetry Markdown summary: {markdown_path}")


if __name__ == "__main__":
    main()
