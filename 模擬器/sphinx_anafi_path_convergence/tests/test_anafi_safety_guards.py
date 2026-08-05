"""Simulator-only safety guards. These tests run WITHOUT Sphinx/Olympe."""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from telemetry_sources import (BANNER, REAL_ANAFI_IP, SKYCONTROLLER_IP,
                               SPHINX_IP_DEFAULT,
                               check_simulator_ip)

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = EXPERIMENT_DIR.parents[1]
FLIGHT_CONTROL_DIR = WORKSPACE_ROOT / "定位演算法" / "flight_control"
SOURCES = sorted(EXPERIMENT_DIR.glob("*.py"))


def test_default_ip_is_sphinx_sim():
    assert SPHINX_IP_DEFAULT == "10.202.0.1"
    assert check_simulator_ip(SPHINX_IP_DEFAULT) == SPHINX_IP_DEFAULT
    assert check_simulator_ip("10.202.0.5") == "10.202.0.5"


def test_real_anafi_ip_rejected():
    with pytest.raises(SystemExit) as e:
        check_simulator_ip(REAL_ANAFI_IP)
    assert "REAL ANAFI" in str(e.value)


def test_skycontroller_ip_rejected():
    with pytest.raises(SystemExit) as e:
        check_simulator_ip(SKYCONTROLLER_IP)
    assert "SkyController" in str(e.value)


def test_arbitrary_non_sim_ip_rejected():
    with pytest.raises(SystemExit):
        check_simulator_ip("192.168.1.20")


@pytest.mark.parametrize("value", [
    "10.202.invalid",
    "10.202.0.1.invalid",
    "10.202.0.999",
    "10.202.0.1/24",
    "::ffff:10.202.0.1",
])
def test_sim_prefix_spoofs_and_invalid_addresses_are_rejected(value):
    with pytest.raises(SystemExit):
        check_simulator_ip(value)


def test_real_ip_override_was_removed():
    from run_sphinx_anafi_convergence import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--dangerous-allow-real-ip"])


def test_banner_text():
    assert "SPHINX ANAFI SIMULATION ONLY" in BANNER


def test_no_emergency_command_in_experiment():
    """The experiment must never use the Olympe Emergency (motor cut) command."""
    for src in SOURCES:
        text = src.read_text()
        assert not re.search(r"\bEmergency\b\s*\(", text), f"Emergency( used in {src.name}"
        assert "import Emergency" not in text.replace(",", " "), src.name


def test_zero_pcmd_before_landing_in_finally():
    """The Sphinx runner must zero PCMD then land inside `finally`."""
    text = (EXPERIMENT_DIR / "run_sphinx_anafi_convergence.py").read_text()
    fin = text[text.index("finally:"):]
    assert fin.index("send_pcmd(0, 0, 0, 0)") < fin.index("Landing()")
    assert "disconnect()" in fin


def test_trial_loop_ends_with_zero_pcmd():
    from controllers import make_controller  # noqa: F401  (import sanity)
    text = (EXPERIMENT_DIR / "run_sphinx_anafi_convergence.py").read_text()
    body = text[text.index("def run_trial"):text.index("def _ground_truth_augment")]
    assert "send_pcmd(0, 0, 0, 0)" in body


def test_pcmd_clamp_bounds():
    from controllers import clamp_pcmd
    r, p, y, g = clamp_pcmd(1000, -1000, 101, -101)
    assert (r, p, y, g) == (100, -100, 100, -100)


def test_conservative_default_limits():
    from controllers import CtrlParams
    p = CtrlParams()
    assert p.max_pitch <= 15 and p.max_yaw <= 30 and p.max_gaz <= 20 and p.max_roll <= 10


def test_sphinx_smoke_uses_event_freshness_for_pose_and_stream_health():
    text = (FLIGHT_CONTROL_DIR / "sphinx_path_follow_smoke.py").read_text()
    assert "SphinxTelemetrySource" in text
    assert "stream_healthy=source.telemetry_healthy" in text
    assert "stream_healthy=lambda: True" not in text


@pytest.mark.parametrize("option,value", [
    ("--duration", "nan"),
    ("--duration", "0"),
    ("--side", "inf"),
    ("--side", "-1"),
])
def test_sphinx_smoke_rejects_unsafe_numeric_limits_before_connect(option, value):
    script = FLIGHT_CONTROL_DIR / "sphinx_path_follow_smoke.py"
    proc = subprocess.run(
        [sys.executable, str(script), option, value],
        text=True, capture_output=True, timeout=10,
    )
    assert proc.returncode != 0
    assert "must be finite and > 0" in (proc.stdout + proc.stderr)


def test_sphinx_launcher_requires_instantiation_and_has_bounded_kill_cleanup():
    text = (FLIGHT_CONTROL_DIR / "launch_sphinx_anafi_empty.sh").read_text()
    assert "All drones instantiated" in text
    assert "kill -KILL" in text
    assert "CLEANUP_TIMEOUT_S" in text
    assert '== *"latest"*' in text
    assert "explicit revision and firmware image path" in text
    assert "latest is forbidden" in text


@pytest.mark.parametrize(
    "selector",
    [
        "",
        "https://firmware.parrot.com/Versions/anafi/pc/%23latest/images/anafi-pc.ext2.zip",
        "https://firmware.parrot.com/Versions/anafi/pc/latest/images/anafi-pc.ext2.zip",
    ],
)
def test_sphinx_launcher_rejects_missing_or_latest_firmware(selector):
    script = FLIGHT_CONTROL_DIR / "launch_sphinx_anafi_empty.sh"
    proc = subprocess.run(
        ["bash", str(script), "--check"],
        env={**os.environ, "FIRMWARE_URL": selector, "PATH": "/usr/bin:/bin"},
        text=True, capture_output=True, timeout=10,
    )
    assert proc.returncode != 0
    assert "explicit" in (proc.stdout + proc.stderr) or "latest" in (
        proc.stdout + proc.stderr
    )
