#!/usr/bin/env python3
"""Resolve the repository workspace layout.

Physical layout (no symlinks):

  <workspace>/
    控制介面程式/   operator_interface, site_profiles, launchers
    定位演算法/     deploy_code, flight_control, configs, validation
    地圖檔/         maps, bundles, mission_routes
    模擬器/         optional validation videos
    執行環境/       optional runtime caches
    outputs/        flight_logs, benchmarks
    .venv/

Set SFM_WORKSPACE_ROOT to override discovery.
"""
from __future__ import annotations

import os
from pathlib import Path

_MARKER_DIRS = ("控制介面程式", "定位演算法", "地圖檔")


def find_workspace(start: Path | None = None) -> Path:
    env = os.environ.get("SFM_WORKSPACE_ROOT", "").strip()
    if env:
        root = Path(env).expanduser().resolve()
        if root.is_dir():
            return root
        raise SystemExit(f"SFM_WORKSPACE_ROOT is not a directory: {root}")

    here = (start or Path(__file__)).resolve()
    for p in [here if here.is_dir() else here.parent, *here.parents]:
        if all((p / name).is_dir() for name in _MARKER_DIRS):
            return p
    raise SystemExit(
        f"could not locate workspace root from {here}; "
        "expected siblings 控制介面程式/定位演算法/地圖檔, "
        "or set SFM_WORKSPACE_ROOT"
    )


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    @property
    def control(self) -> Path:
        return self.root / "控制介面程式"

    @property
    def algorithms(self) -> Path:
        return self.root / "定位演算法"

    @property
    def maps(self) -> Path:
        return self.root / "地圖檔"

    @property
    def sim(self) -> Path:
        return self.root / "模擬器"

    @property
    def runtime(self) -> Path:
        return self.root / "執行環境"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def operator_interface(self) -> Path:
        return self.control / "operator_interface"

    @property
    def site_profiles(self) -> Path:
        return self.control / "site_profiles"

    @property
    def deploy_code(self) -> Path:
        return self.algorithms / "deploy_code" / "sfm_glomap_deploy"

    @property
    def flight_control(self) -> Path:
        return self.algorithms / "flight_control"

    @property
    def configs(self) -> Path:
        return self.algorithms / "configs"

    @property
    def validation(self) -> Path:
        return self.algorithms / "validation"

    @property
    def map_ply_dir(self) -> Path:
        return self.maps / "maps"

    @property
    def bundles(self) -> Path:
        return self.maps / "bundles"

    @property
    def mission_routes(self) -> Path:
        return self.maps / "mission_routes"

    @property
    def site_packages(self) -> Path:
        return self.maps / "場域"

    @property
    def flight_logs(self) -> Path:
        return self.outputs / "flight_logs"

    @property
    def torch_hub_cache(self) -> Path:
        return self.runtime / "torch_hub_cache"

    @property
    def venv_python(self) -> Path:
        return self.root / ".venv" / "bin" / "python"


def workspace_from_file(file_path: str | Path) -> Workspace:
    return Workspace(find_workspace(Path(file_path).resolve()))
