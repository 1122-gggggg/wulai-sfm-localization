from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

CONTROL_ROOT = Path(__file__).resolve().parents[1]
SIMULATED_LAUNCHER = CONTROL_ROOT / "影片模擬串流" / "啟動.sh"
REAL_LAUNCHER = CONTROL_ROOT / "真機串流" / "啟動.sh"
SITE_PROFILE = CONTROL_ROOT / "site_profiles" / "river_site_edm.json"
P119_VALIDATOR = CONTROL_ROOT / "validate_p119_source.py"


def launch(script: Path, *args: str, **extra_env: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({
        "SFM_LAUNCH_DRY_RUN": "1",
        "SFM_MAX_PERFORMANCE": "0",
        "SFM_SITE_PROFILE": str(SITE_PROFILE),
        "SFM_UI_PYTHON": sys.executable,
        **extra_env,
    })
    return subprocess.run(
        [str(script), *args],
        cwd=script.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )


def test_simulated_launcher_is_file_only_and_never_live(tmp_path: Path) -> None:
    video = tmp_path / "replay.mp4"
    video.write_bytes(b"test")

    result = launch(SIMULATED_LAUNCHER, str(video))

    assert result.returncode == 0, result.stderr
    command = next(
        line for line in result.stdout.splitlines() if "dry-run command:" in line
    )
    tokens = command.split()
    assert "--interface simulated-stream" in command
    assert f"--video {video}" in command
    assert "--live" not in tokens
    assert "real-flight" not in command


def test_simulated_launcher_discovers_the_only_imported_video(tmp_path: Path) -> None:
    video_dir = tmp_path / "imported_videos"
    video_dir.mkdir()
    video = video_dir / "portable_replay.mp4"
    video.write_bytes(b"test")

    result = launch(SIMULATED_LAUNCHER, VIDEO_DIR=str(video_dir))

    assert result.returncode == 0, result.stderr
    assert "river_site_edm.json" in result.stdout
    assert str(video) in result.stdout
    assert "H264 main 5000 kbps" in result.stdout
    assert "延遲 280 ms、丟包 0 %" in result.stdout


def test_simulated_launcher_rejects_multiple_imported_videos(tmp_path: Path) -> None:
    video_dir = tmp_path / "imported_videos"
    video_dir.mkdir()
    for name in ("one.mp4", "two.mp4"):
        (video_dir / name).write_bytes(b"test")

    result = launch(SIMULATED_LAUNCHER, VIDEO_DIR=str(video_dir))

    assert result.returncode == 2
    assert "發現多部影片" in result.stderr


def test_p119_hash_is_identical_in_launcher_and_integrity_validator() -> None:
    launcher = SIMULATED_LAUNCHER.read_text(encoding="utf-8")
    validator = P119_VALIDATOR.read_text(encoding="utf-8")
    launcher_hash = re.search(r'P119_SHA256="([0-9a-f]{64})"', launcher)
    validator_hash = re.search(r'EXPECTED_SHA256 = "([0-9a-f]{64})"', validator)

    assert launcher_hash is not None
    assert validator_hash is not None
    assert launcher_hash.group(1) == validator_hash.group(1)


@pytest.mark.parametrize(
    ("preset", "loss"),
    [("nominal", "0"), ("loss-1", "1"), ("loss-3", "3"), ("loss-5", "5")],
)
def test_simulated_launcher_link_presets(preset: str, loss: str, tmp_path: Path) -> None:
    video = tmp_path / "replay.mp4"
    video.write_bytes(b"test")

    result = launch(
        SIMULATED_LAUNCHER,
        str(video),
        ANAFI_LINK_PRESET=preset,
    )

    assert result.returncode == 0, result.stderr
    assert f"延遲 280 ms、丟包 {loss} %" in result.stdout


def test_simulated_launcher_rejects_unknown_link_preset(tmp_path: Path) -> None:
    video = tmp_path / "replay.mp4"
    video.write_bytes(b"test")

    result = launch(
        SIMULATED_LAUNCHER,
        str(video),
        ANAFI_LINK_PRESET="invented",
    )

    assert result.returncode == 2
    assert "ANAFI_LINK_PRESET" in result.stderr


def test_simulated_launcher_rejects_system_python_in_portable_mode(
    tmp_path: Path,
) -> None:
    video = tmp_path / "replay.mp4"
    video.write_bytes(b"test")
    system_python = str(Path(sys.executable).resolve())

    result = launch(
        SIMULATED_LAUNCHER,
        str(video),
        SFM_LAUNCH_DRY_RUN="0",
        SFM_UI_PYTHON=system_python,
        SFM_LOCALIZER_PYTHON=system_python,
    )

    assert result.returncode == 2
    assert "portable mode" in result.stderr


@pytest.mark.parametrize(
    "forbidden", ("--live", "--interface=real-flight", "--video=/tmp/other.mp4")
)
def test_simulated_launcher_rejects_cross_interface_overrides(
    tmp_path: Path, forbidden: str,
) -> None:
    video = tmp_path / "replay.mp4"
    video.write_bytes(b"test")

    result = launch(SIMULATED_LAUNCHER, str(video), forbidden)

    assert result.returncode == 2
    assert "拒絕跨接口參數" in result.stderr


@pytest.mark.parametrize(
    "args",
    (
        ("--localizer-python", "/usr/bin/python3.10"),
        ("--localizer-worker", "/tmp/unvalidated_worker.py"),
        ("--localizer-backend", "xfeat"),
        ("--localizer-deploy-dir", "/tmp/unvalidated_deploy"),
        ("--localizer-profile", "/tmp/unvalidated_profile.json"),
    ),
)
def test_simulated_launcher_rejects_runtime_overrides(
    tmp_path: Path, args: tuple[str, str]
) -> None:
    video = tmp_path / "replay.mp4"
    video.write_bytes(b"test")

    result = launch(SIMULATED_LAUNCHER, str(video), *args)

    assert result.returncode == 2
    assert "portable 入口不接受額外" in result.stderr


def test_real_launcher_is_olympe_only_and_has_no_video_argument() -> None:
    result = launch(REAL_LAUNCHER)

    assert result.returncode == 0, result.stderr
    command = next(
        line for line in result.stdout.splitlines() if "dry-run command:" in line
    )
    assert "--interface real-flight" in command
    assert "--video" not in command
    assert "simulated-stream" not in command
