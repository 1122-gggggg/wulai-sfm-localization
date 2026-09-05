"""SFM_EDM_ESEKF_DISABLE forces the ESEKF / KLT-3D-aware paths off.

The toggle exists so a live recording can be replayed both ways for A/B
evaluation (定位演算法/validation/benchmark_esekf_live_replay.py). Every
in-tracker ESEKF call site guards on ``getattr(self, "esekf", None) is not
None``; this test pins that ``_init_esekf`` honours the env var.
"""

from __future__ import annotations

from production_edm_tracker import EDMConfig, ProductionEDMTracker


def _bare_tracker() -> ProductionEDMTracker:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig()
    tracker.pose_guided = None
    return tracker


def test_default_builds_esekf(monkeypatch):
    monkeypatch.delenv("SFM_EDM_ESEKF_DISABLE", raising=False)
    tracker = _bare_tracker()
    tracker._init_esekf()
    assert tracker.esekf is not None
    assert tracker.esekf_disabled_by_env is False


def test_disable_env_forces_esekf_none(monkeypatch):
    monkeypatch.setenv("SFM_EDM_ESEKF_DISABLE", "1")
    tracker = _bare_tracker()
    tracker._init_esekf()
    assert tracker.esekf is None
    assert tracker.esekf_disabled_by_env is True


def test_non_one_value_keeps_esekf(monkeypatch):
    monkeypatch.setenv("SFM_EDM_ESEKF_DISABLE", "0")
    tracker = _bare_tracker()
    tracker._init_esekf()
    assert tracker.esekf is not None
    assert tracker.esekf_disabled_by_env is False
