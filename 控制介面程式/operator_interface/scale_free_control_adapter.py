"""Load the single scale-free controller core owned by parrot_stimulate."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def authoritative_core_path() -> Path:
    workspace = Path(__file__).resolve().parents[2]
    return (
        workspace
        / "模擬器"
        / "parrot_stimulate"
        / "src"
        / "anafi_pcmd_sim"
        / "scale_free_control.py"
    )


_PATH = authoritative_core_path()
_SPEC = importlib.util.spec_from_file_location("_sfm_scale_free_control", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load authoritative scale-free control core: {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

ScaleFreeConfig = _MODULE.ScaleFreeConfig
ScaleFreeSample = _MODULE.ScaleFreeSample
ScaleFreeDecision = _MODULE.ScaleFreeDecision
SpeedLimitChange = _MODULE.SpeedLimitChange
decide_scale_free = _MODULE.decide_scale_free
command_is_fresh = _MODULE.command_is_fresh
validate_speed_limit_change = _MODULE.validate_speed_limit_change

__all__ = [
    "ScaleFreeConfig",
    "ScaleFreeSample",
    "ScaleFreeDecision",
    "SpeedLimitChange",
    "decide_scale_free",
    "command_is_fresh",
    "validate_speed_limit_change",
    "authoritative_core_path",
]
