from __future__ import annotations

from pathlib import Path

import simulator_preflight


def test_preflight_accepts_a_complete_profile_and_video(
    tmp_path: Path, monkeypatch
) -> None:
    root = Path(__file__).resolve().parents[1]
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
        Path(__file__).resolve().parents[1]
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
    root = Path(__file__).resolve().parents[1]
    profile = root / "控制介面程式/site_profiles/river_site_edm.json"
    monkeypatch.setenv("SFM_WORKSPACE_ROOT", str(tmp_path))
    failures: list[str] = []
    runtime: dict[str, object] = {}
    simulator_preflight._check_environment_contract(root, profile, failures, runtime)
    assert any("SFM_WORKSPACE_ROOT" in item for item in failures)


def test_preflight_does_not_treat_system_python_as_the_venv(
    monkeypatch, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[1]
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
