from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import simulator_preflight


def test_preflight_accepts_a_complete_profile_and_video(
    tmp_path: Path, monkeypatch
) -> None:
    root = Path(__file__).resolve().parents[2]
    profile = root / "控制介面程式/site_profiles/river_site_edm.json"
    video = tmp_path / "replay.mp4"
    video.write_bytes(b"not-a-real-video")
    monkeypatch.setattr(
        simulator_preflight,
        "_check_video",
        lambda _video, _failures: {"duration_s": "1"},
    )
    report = simulator_preflight.run_preflight(
        root=root, profile_path=profile, video_path=video, check_runtime=False
    )
    assert report["ok"]
    assert report["profile"]["site_id"] == "river_site_edm"
    collision_monitor = report["runtime"]["collision_monitor"]
    assert collision_monitor["status"] == "available_non_production"
    assert collision_monitor["available"] is True
    assert collision_monitor["production_safety"] is False
    assert collision_monitor["collision_protection_claim"] is False


def test_preflight_reports_missing_profile_and_video(tmp_path: Path) -> None:
    report = simulator_preflight.run_preflight(
        root=tmp_path,
        profile_path=tmp_path / "missing.json",
        video_path=tmp_path / "missing.mp4",
        check_runtime=False,
    )
    assert not report["ok"]
    assert any("site profile invalid" in item for item in report["failures"])
    assert any("video is missing" in item for item in report["failures"])
    assert any("scale-free control core" in item for item in report["failures"])


def test_preflight_requires_authoritative_scale_free_core(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path
    profile = (
        Path(__file__).resolve().parents[2]
        / "控制介面程式/site_profiles/river_site_edm.json"
    )
    video = tmp_path / "replay.mp4"
    video.write_bytes(b"not-a-real-video")
    monkeypatch.setattr(
        simulator_preflight,
        "_check_video",
        lambda _video, _failures: {"duration_s": "1"},
    )
    report = simulator_preflight.run_preflight(
        root=root, profile_path=profile, video_path=video, check_runtime=False
    )
    assert any("scale-free control core" in item for item in report["failures"])


def test_preflight_rejects_runtime_environment_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[2]
    profile = root / "控制介面程式/site_profiles/river_site_edm.json"
    monkeypatch.setenv("SFM_WORKSPACE_ROOT", str(tmp_path))
    failures: list[str] = []
    runtime: dict[str, object] = {}
    simulator_preflight._check_environment_contract(root, profile, failures, runtime)
    assert any("SFM_WORKSPACE_ROOT" in item for item in failures)


def test_preflight_does_not_treat_system_python_as_the_venv(
    monkeypatch, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[2]
    profile = root / "控制介面程式/site_profiles/river_site_edm.json"
    venv_python = tmp_path / ".venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    system_python = Path(simulator_preflight.sys.executable).resolve()
    venv_python.symlink_to(system_python)
    monkeypatch.setattr(simulator_preflight.sys, "executable", str(venv_python))
    monkeypatch.setenv("SFM_UI_PYTHON", str(system_python))
    failures: list[str] = []

    simulator_preflight._check_environment_contract(root, profile, failures, {})

    assert any("SFM_UI_PYTHON" in item for item in failures)


def test_preflight_rejects_system_site_packages_venv(
    monkeypatch, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[2]
    profile = root / "控制介面程式/site_profiles/river_site_edm.json"
    (tmp_path / "pyvenv.cfg").write_text(
        "include-system-site-packages = true\n"
    )
    monkeypatch.setattr(simulator_preflight.sys, "prefix", str(tmp_path))
    failures: list[str] = []
    runtime: dict[str, object] = {}

    simulator_preflight._check_environment_contract(
        root, profile, failures, runtime
    )

    assert runtime["system_site_packages"] is True
    assert any("include-system-site-packages=true" in item for item in failures)


def test_collision_monitor_is_unavailable_without_hash_locked_scipy(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        simulator_preflight.importlib,
        "import_module",
        lambda name: SimpleNamespace(cKDTree=object())
        if name == "scipy.spatial"
        else (_ for _ in ()).throw(AssertionError(name)),
    )
    failures: list[str] = []

    status = simulator_preflight._collision_monitor_status(
        tmp_path, failures, production_required=False
    )

    assert status["runtime_import"] is True
    assert status["hash_locked"] is False
    assert status["available"] is False
    assert status["status"] == "unavailable"
    assert status["production_safety"] is False
    assert status["collision_protection_claim"] is False
    assert not failures


def test_collision_monitor_requirement_fails_closed_without_lock(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        simulator_preflight.importlib,
        "import_module",
        lambda name: SimpleNamespace(cKDTree=None)
        if name == "scipy.spatial"
        else (_ for _ in ()).throw(AssertionError(name)),
    )
    failures: list[str] = []

    status = simulator_preflight._collision_monitor_status(
        tmp_path, failures, production_required=True
    )

    assert status["status"] == "unavailable"
    assert status["required_for_production"] is True
    assert any("fail-closed" in item for item in failures)
