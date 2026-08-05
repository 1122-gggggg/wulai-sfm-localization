import hashlib
import json
import math
import subprocess
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from anafi_pcmd_sim.models import TruePosition
from anafi_pcmd_sim.route import Waypoint
from anafi_pcmd_sim.simulator import (
    FirmwareSourceError,
    SphinxFinalApproachDisplacement,
    SphinxWindDisplacements,
    _move_drone_relative,
    build_sphinx_command,
    build_ue_command,
    resolve_firmware_source,
)


def test_sphinx_command_uses_anafi_pc_firmware_and_a_zero_yaw_spawn(tmp_path: Path) -> None:
    firmware = tmp_path / "anafi-pc.ext2.zip"
    firmware.write_bytes(b"firmware")
    command = build_sphinx_command(tmp_path, firmware_source=firmware)

    assert command[0] == "sphinx"
    assert "--datalog-rate=50" in command
    assert any(str(firmware) in item for item in command)
    assert any("::pose=0 0 0.2 0 0 0" in item for item in command)


def test_sphinx_command_accepts_a_pinned_local_firmware_path(tmp_path: Path) -> None:
    firmware = tmp_path / "anafi-pc.ext2.zip"
    firmware.write_bytes(b"firmware")
    command = build_sphinx_command(tmp_path, firmware_source=firmware)

    assert any(str(firmware) in item for item in command)


def test_sphinx_command_accepts_a_route_spawn_pose(tmp_path: Path) -> None:
    pose = "1.25 -2.5 0.2 0 0 1.5708"
    command = build_sphinx_command(tmp_path, spawn_pose=pose)

    assert any(f"::pose={pose}" in item for item in command)


def test_sphinx_command_can_disable_the_unused_front_camera(tmp_path: Path) -> None:
    command = build_sphinx_command(tmp_path, disable_front_camera=True)

    assert any("::with_front_cam=0" in item for item in command)


def test_explicit_firmware_source_overrides_any_local_cache(tmp_path: Path) -> None:
    alternative = tmp_path / "alternate.ext2.zip"
    alternative.write_bytes(b"alternative")

    assert resolve_firmware_source(alternative) == str(alternative)


def test_project_firmware_cache_is_verified_and_used(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    firmware = tmp_path / "anafi-pc.ext2.zip"
    firmware.write_bytes(b"firmware")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact": firmware.name,
                "bytes": firmware.stat().st_size,
                "sha256": hashlib.sha256(firmware.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("anafi_pcmd_sim.simulator.DEFAULT_FIRMWARE_CACHE", firmware)
    monkeypatch.setattr("anafi_pcmd_sim.simulator.DEFAULT_FIRMWARE_MANIFEST", manifest)

    assert resolve_firmware_source() == str(firmware)


def test_project_firmware_cache_rejects_a_sha_mismatch(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    firmware = tmp_path / "anafi-pc.ext2.zip"
    firmware.write_bytes(b"firmware")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact": firmware.name,
                "bytes": firmware.stat().st_size,
                "sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("anafi_pcmd_sim.simulator.DEFAULT_FIRMWARE_CACHE", firmware)
    monkeypatch.setattr("anafi_pcmd_sim.simulator.DEFAULT_FIRMWARE_MANIFEST", manifest)

    with pytest.raises(FirmwareSourceError, match="SHA-256"):
        resolve_firmware_source()


def test_missing_firmware_does_not_fall_back_to_a_remote_latest_url(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    missing = tmp_path / "missing.ext2.zip"
    monkeypatch.setattr("anafi_pcmd_sim.simulator.DEFAULT_FIRMWARE_CACHE", missing)

    with pytest.raises(FirmwareSourceError, match="verified local ANAFI PC firmware"):
        resolve_firmware_source()


def test_ue_command_is_headless_by_default() -> None:
    assert build_ue_command() == ["parrot-ue4-empty", "-RenderOffScreen"]


def test_ue_command_can_open_a_window() -> None:
    assert build_ue_command(show_window=True) == ["parrot-ue4-empty", "-quality=low"]


def test_relative_displacement_uses_the_installed_pysphinx_api(
    monkeypatch: MonkeyPatch,
) -> None:
    captured: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append((command, kwargs["env"]))
        return subprocess.CompletedProcess(command, 0, stdout="true\n", stderr="")

    monkeypatch.setattr("anafi_pcmd_sim.simulator.subprocess.run", fake_run)
    _move_drone_relative(0.25, -0.1)

    command, environment = captured[0]
    assert command[0] == "/usr/bin/python3.10"
    assert command[-3:] == ["0.25", "-0.1", "0.0"]
    assert "/opt/parrot-sphinx/usr/lib" in environment["LD_LIBRARY_PATH"]


def test_wind_displacement_is_seeded_logged_and_never_exceeds_configured_limit(
    monkeypatch: MonkeyPatch,
) -> None:
    offsets: list[tuple[float, float, float]] = []
    monkeypatch.setattr(
        "anafi_pcmd_sim.simulator._move_drone_relative",
        lambda x, y, z=0.0: offsets.append((x, y, z)),
    )
    disturbances = SphinxWindDisplacements(
        seed=42,
        maximum_displacement_m=0.4,
        interval_s=20.0,
    )
    disturbances._apply_once()

    assert len(offsets) == 1
    assert len(disturbances.events) == 1
    assert disturbances.events[0]["displacement_m"] <= 0.4
    assert math.hypot(offsets[0][0], offsets[0][1]) == pytest.approx(
        disturbances.events[0]["displacement_m"]
    )


def test_final_approach_wind_applies_once_at_the_maximum_bound(
    monkeypatch: MonkeyPatch,
) -> None:
    offsets: list[tuple[float, float, float]] = []
    monkeypatch.setattr(
        "anafi_pcmd_sim.simulator._move_drone_relative",
        lambda x, y, z=0.0: offsets.append((x, y, z)),
    )
    disturbance = SphinxFinalApproachDisplacement(displacement_m=0.4)
    target = Waypoint(index=9, x_m=1.0, y_m=0.0, z_m=1.0)
    position = TruePosition(timestamp_s=1.0, x_m=0.5, y_m=0.0, z_m=1.0)

    disturbance.maybe_apply(position, target, False)
    disturbance.maybe_apply(position, target, True)
    disturbance.maybe_apply(position, target, True)

    assert offsets == [pytest.approx((-0.4, 0.0, 0.0))]
    assert disturbance.events[0]["displacement_m"] == pytest.approx(0.4)
