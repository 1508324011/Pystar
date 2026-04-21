from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pystar.matlab_engine_bootstrap import (
    close_matlab_engine_best_effort,
    load_matlab_engine_module,
    probe_matlab_engine_environment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect MATLAB Engine availability for the active Python environment")
    _ = parser.add_argument(
        "--start-engine-smoke",
        action="store_true",
        help="Start MATLAB Engine, evaluate version, and quit for a minimal runtime smoke test.",
    )
    args = parser.parse_args()

    status = probe_matlab_engine_environment()
    if not status.get("available"):
        print(json.dumps(status, indent=2, ensure_ascii=False))
        raise SystemExit(1)

    if args.start_engine_smoke:
        matlab_engine = load_matlab_engine_module(
            consumer="scripts/check_matlab_engine.py --start-engine-smoke",
        )
        eng = None
        engine_smoke: dict[str, object] = {"started": False}
        try:
            eng = matlab_engine.start_matlab("-nodesktop")
            engine_smoke["started"] = True
            engine_smoke["version"] = eng.eval("version", nargout=1)
        finally:
            if eng is not None:
                teardown_warning = close_matlab_engine_best_effort(
                    eng,
                    consumer="scripts/check_matlab_engine.py --start-engine-smoke",
                )
                if teardown_warning is not None:
                    engine_smoke["teardown_warning"] = teardown_warning
        status["engine_smoke"] = engine_smoke
    else:
        status["engine_smoke"] = {"started": False}

    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
