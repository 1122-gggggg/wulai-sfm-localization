from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

import simulator_preflight


def test_preflight_accepts_a_complete_profile_and_video(
    tmp_path: Path, monkeypatch
) -> None:
    root = Path(__file__).resolve().parents[2]
    profile = root / "地圖檔/場域/river_site/site_profile.json"
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


def test_preflight_allows_omitted_video_for_live_runtime_checks(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    profile = root / "地圖檔" / "場域" / "river_site" / "site_profile.json"
    if not profile.is_file():
        pytest.skip("river site profile is not mounted")
    report = simulator_preflight.run_preflight(
        root=root,
        profile_path=profile,
        video_path=None,
        check_runtime=False,
    )
    assert report["ok"]
    assert report["video"]["path"] is None


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


def test_preflight_rejects_route_digest_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    root = Path(__file__).resolve().parents[2]
    route = tmp_path / "route.json"
    route.write_text("{}", encoding="utf-8")
    video = tmp_path / "replay.mp4"
    video.write_bytes(b"not-a-real-video")
    profile = SimpleNamespace(
        site_id="test-site",
        source=tmp_path / "site.json",
        map_ply=tmp_path / "map.ply",
        localization_bundle=tmp_path / "bundle.pt",
        route_json=route,
        asset_sha256=SimpleNamespace(route_json="0" * 64),
    )
    monkeypatch.setattr(
        simulator_preflight,
        "_check_video",
        lambda _video, _failures: {"duration_s": "1"},
    )
    monkeypatch.setattr(
        "site_profile.load_site_profile", lambda _path: profile
    )

    report = simulator_preflight.run_preflight(
        root=root,
        profile_path=tmp_path / "site.json",
        video_path=video,
        check_runtime=False,
    )

    assert not report["ok"]
    assert any(
        "route_json SHA-256 mismatch" in item
        for item in report["failures"]
    )


def test_preflight_requires_authoritative_scale_free_core(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path
    profile = (
        Path(__file__).resolve().parents[2]
        / "地圖檔/場域/river_site/site_profile.json"
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
    profile = root / "地圖檔/場域/river_site/site_profile.json"
    monkeypatch.setenv("SFM_WORKSPACE_ROOT", str(tmp_path))
    failures: list[str] = []
    runtime: dict[str, object] = {}
    simulator_preflight._check_environment_contract(root, profile, failures, runtime)
    assert any("SFM_WORKSPACE_ROOT" in item for item in failures)


def test_preflight_does_not_treat_system_python_as_the_venv(
    monkeypatch, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[2]
    profile = root / "地圖檔/場域/river_site/site_profile.json"
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
    profile = root / "地圖檔/場域/river_site/site_profile.json"
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


def test_runtime_containment_accepts_digest_bound_site_localizer_profile(
    tmp_path: Path,
) -> None:
    localizer_profile = tmp_path / "地圖檔/場域/site/releases/v1/profile.json"
    localizer_profile.parent.mkdir(parents=True)
    localizer_profile.write_text("{}", encoding="utf-8")
    profile = SimpleNamespace(
        localizer_deploy_dir=None,
        localizer_profile=localizer_profile,
    )
    failures: list[str] = []

    simulator_preflight._check_profile_runtime_containment(
        tmp_path,
        profile,
        failures,
    )

    assert failures == []


def test_runtime_containment_rejects_localizer_profile_outside_config_roots(
    tmp_path: Path,
) -> None:
    localizer_profile = tmp_path / "untrusted/profile.json"
    localizer_profile.parent.mkdir(parents=True)
    localizer_profile.write_text("{}", encoding="utf-8")
    profile = SimpleNamespace(
        localizer_deploy_dir=None,
        localizer_profile=localizer_profile,
    )
    failures: list[str] = []

    simulator_preflight._check_profile_runtime_containment(
        tmp_path,
        profile,
        failures,
    )

    assert any("outside approved config roots" in error for error in failures)


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
