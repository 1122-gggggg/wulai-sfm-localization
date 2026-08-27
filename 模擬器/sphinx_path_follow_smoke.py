#!/usr/bin/env python3
"""Compatibility entrypoint for the authoritative Sphinx smoke test."""
from __future__ import annotations

import runpy
from pathlib import Path


TARGET = (
    Path(__file__).resolve().parents[1]
    / "定位演算法"
    / "flight_control"
    / "sphinx_path_follow_smoke.py"
)


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
