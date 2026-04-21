from __future__ import annotations

import os


if os.environ.get("PYSTAR_DISABLE_MATLAB_ENGINE_BOOTSTRAP") != "1":
    try:
        from pystar.matlab_engine_bootstrap import configure_matlab_engine_environment
    except Exception:
        pass
    else:
        try:
            _ = configure_matlab_engine_environment()
        except Exception:
            pass
