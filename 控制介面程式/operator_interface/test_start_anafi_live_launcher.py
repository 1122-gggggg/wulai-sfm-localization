from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("start_anafi_live.sh")
APP = SCRIPT.with_name("flight_operator_app.py")
LAUNCH_ENV = {
    "CTRL",
    "DISPLAY",
    "IP",
    "LOC_BENCH",
    "LOC_BENCH_TRACK",
    "LOC_EVERY_N",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
    "OPENBLAS_NUM_THREADS",
    "SFM_LAUNCH_DRY_RUN",
    "SFM_LOCALIZER_PYTHON",
    "SFM_MAX_ALTITUDE_M",
    "SFM_MAX_DISTANCE_M",
    "SFM_MAX_PERFORMANCE",
    "SFM_CPU_THREADS",
    "SFM_SITE_PROFILE",
    "SFM_UI_PYTHON",
    "VENV",
    "WAYLAND_DISPLAY",
}


def run_launcher(*args: str, **overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in LAUNCH_ENV:
        env.pop(name, None)
    env.update({
        "SFM_LAUNCH_DRY_RUN": "1",
        "SFM_UI_PYTHON": sys.executable,
        **overrides,
    })
    return subprocess.run(
        [str(SCRIPT), *args],
        cwd=SCRIPT.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )


def write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)


def test_direct_ip_derives_drone_and_preserves_benchmark_flags():
    result = run_launcher(
        IP="192.168.42.1",
        LOC_BENCH_TRACK="1",
        LOC_EVERY_N="2",
    )

    assert result.returncode == 0, result.stderr
    assert "--ip 192.168.42.1 --controller drone" in result.stdout
    assert "--no-live-detect" in result.stdout
    assert "--max-altitude-m 50 --max-distance-m 100 --distance-geofence" in result.stdout
    assert "--auto-inspect --boot-lock-ms 0" in result.stdout
    assert "--loc-force-track-bench" in result.stdout
    assert "--loc-every-n-frames 2" in result.stdout


def test_skycontroller_derives_field_ip_without_network_probe():
    result = run_launcher(CTRL="skycontroller3")

    assert result.returncode == 0, result.stderr
    assert "--ip 192.168.53.1 --controller skycontroller3" in result.stdout


def test_firmware_limit_defaults_can_be_overridden_from_environment():
    result = run_launcher(SFM_MAX_ALTITUDE_M="18", SFM_MAX_DISTANCE_M="75")

    assert result.returncode == 0, result.stderr
    assert "--max-altitude-m 18 --max-distance-m 75 --distance-geofence" in result.stdout


def test_inconsistent_ip_controller_pair_fails_before_launch():
    result = run_launcher(IP="192.168.42.1", CTRL="skycontroller3")

    assert result.returncode == 2
    assert "inconsistent IP/controller pair" in result.stderr
    assert "dry-run command" not in result.stdout


def test_max_performance_defaults_to_gamemode_prefix_in_dry_run(tmp_path: Path):
    gamemoderun = tmp_path / "gamemoderun"
    write_executable(gamemoderun, 'exec "$@"\n')

    result = run_launcher(PATH=f"{tmp_path}:{os.environ['PATH']}")

    assert result.returncode == 0, result.stderr
    assert f"{gamemoderun} {sys.executable}" in result.stdout
    assert "sustained CPU thread budget: 4" in result.stdout


def test_max_performance_zero_skips_gamemode_prefix_in_dry_run(tmp_path: Path):
    gamemoderun = tmp_path / "gamemoderun"
    write_executable(gamemoderun, 'exec "$@"\n')

    result = run_launcher(
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        SFM_MAX_PERFORMANCE="0",
    )

    assert result.returncode == 0, result.stderr
    assert str(gamemoderun) not in result.stdout
    assert f"dry-run command: {sys.executable}" in result.stdout


def test_invalid_max_performance_value_fails_before_launch():
    result = run_launcher(SFM_MAX_PERFORMANCE="yes")

    assert result.returncode == 2
    assert "SFM_MAX_PERFORMANCE must be 0 or 1" in result.stderr
    assert "dry-run command" not in result.stdout


def test_invalid_cpu_thread_budget_fails_before_launch():
    result = run_launcher(SFM_CPU_THREADS="0")

    assert result.returncode == 2
    assert "SFM_CPU_THREADS must be a positive integer" in result.stderr
    assert "dry-run command" not in result.stdout


def test_live_max_performance_is_best_effort_and_wraps_app(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "calls.log"
    fake_python = tmp_path / "python"
    write_executable(
        fake_python,
        'printf \'threads %s/%s/%s/%s\\n\' "$OPENCV_FOR_THREADS_NUM" '
        '"$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS" '
        '>> "$CALL_LOG"\nprintf \'python %s\\n\' "$*" >> "$CALL_LOG"\n',
    )
    write_executable(bin_dir / "ping", "exit 0\n")
    write_executable(
        bin_dir / "powerprofilesctl",
        'printf \'powerprofilesctl %s\\n\' "$*" >> "$CALL_LOG"\n'
        'if [[ "$1" == "get" ]]; then echo balanced; exit 0; fi\nexit 1\n',
    )
    write_executable(
        bin_dir / "gsettings",
        'printf \'gsettings %s\\n\' "$*" >> "$CALL_LOG"\n'
        'if [[ "$1" == "get" ]]; then echo true; exit 0; fi\nexit 1\n',
    )
    write_executable(
        bin_dir / "gamemoderun",
        'printf \'gamemoderun\\n\' >> "$CALL_LOG"\nexec "$@"\n',
    )

    result = run_launcher(
        CALL_LOG=str(call_log),
        CTRL="drone",
        IP="192.168.42.1",
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        SFM_LAUNCH_DRY_RUN="0",
        SFM_UI_PYTHON=str(fake_python),
    )

    assert result.returncode == 0, result.stderr
    calls = call_log.read_text().splitlines()
    assert "powerprofilesctl set performance" in calls
    assert (
        "gsettings set org.gnome.settings-daemon.plugins.power "
        "power-saver-profile-on-low-battery false"
    ) in calls
    assert "gamemoderun" in calls
    assert "threads 4/4/4/4" in calls
    assert any(
        line.startswith(
            f"python -u {APP} --interface real-flight"
        )
        for line in calls
    )
    assert "continuing without it" in result.stderr


def test_live_reachability_accepts_http_when_drone_drops_icmp(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "calls.log"
    fake_python = tmp_path / "python"
    write_executable(fake_python, 'printf \'python started\\n\' >> "$CALL_LOG"\n')
    write_executable(bin_dir / "ping", "exit 1\n")
    write_executable(
        bin_dir / "curl",
        'printf \'curl %s\\n\' "$*" >> "$CALL_LOG"\nexit 0\n',
    )

    result = run_launcher(
        CALL_LOG=str(call_log),
        CTRL="drone",
        IP="192.168.42.1",
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        SFM_LAUNCH_DRY_RUN="0",
        SFM_MAX_PERFORMANCE="0",
        SFM_UI_PYTHON=str(fake_python),
    )

    assert result.returncode == 0, result.stderr
    calls = call_log.read_text().splitlines()
    assert any(line.startswith("curl ") for line in calls)
    assert "python started" in calls


def test_live_max_performance_restores_session_settings_and_exit_code(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "calls.log"
    power_state = tmp_path / "power.state"
    battery_state = tmp_path / "battery.state"
    fake_python = tmp_path / "python"
    write_executable(
        fake_python,
        'printf \'python sees %s/%s\\n\' "$(< "$POWER_STATE")" '
        '"$(< "$BATTERY_STATE")" >> "$CALL_LOG"\nexit 7\n',
    )
    write_executable(bin_dir / "ping", "exit 0\n")
    write_executable(
        bin_dir / "powerprofilesctl",
        'if [[ "$1" == "get" ]]; then\n'
        '  [[ -f "$POWER_STATE" ]] && cat "$POWER_STATE" || echo balanced\n'
        'else\n'
        '  printf \'power set %s\\n\' "$2" >> "$CALL_LOG"\n'
        '  printf \'%s\\n\' "$2" > "$POWER_STATE"\n'
        'fi\n',
    )
    write_executable(
        bin_dir / "gsettings",
        'if [[ "$1" == "get" ]]; then\n'
        '  [[ -f "$BATTERY_STATE" ]] && cat "$BATTERY_STATE" || echo true\n'
        'else\n'
        '  printf \'battery set %s\\n\' "$4" >> "$CALL_LOG"\n'
        '  printf \'%s\\n\' "$4" > "$BATTERY_STATE"\n'
        'fi\n',
    )
    write_executable(bin_dir / "gamemoderun", 'exec "$@"\n')

    result = run_launcher(
        BATTERY_STATE=str(battery_state),
        CALL_LOG=str(call_log),
        CTRL="drone",
        IP="192.168.42.1",
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        POWER_STATE=str(power_state),
        SFM_LAUNCH_DRY_RUN="0",
        SFM_UI_PYTHON=str(fake_python),
    )

    assert result.returncode == 7
    assert call_log.read_text().splitlines() == [
        "power set performance",
        "battery set false",
        "python sees performance/false",
        "battery set true",
        "power set balanced",
    ]
    assert power_state.read_text().strip() == "balanced"
    assert battery_state.read_text().strip() == "true"


@pytest.mark.parametrize(
    ("signal_number", "signal_label", "expected_returncode"),
    (
        (signal.SIGTERM, "term", 143),
        (signal.SIGHUP, "hup", 129),
    ),
)
def test_live_max_performance_forwards_signal_before_restoring(
    tmp_path: Path,
    signal_number: signal.Signals,
    signal_label: str,
    expected_returncode: int,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "calls.log"
    ready = tmp_path / "ready"
    power_state = tmp_path / "power.state"
    battery_state = tmp_path / "battery.state"
    fake_python = tmp_path / "python"
    write_executable(
        fake_python,
        'trap \'printf "python term\\n" >> "$CALL_LOG"; exit 143\' TERM\n'
        'trap \'printf "python hup\\n" >> "$CALL_LOG"; exit 129\' HUP\n'
        'touch "$READY"\nwhile true; do sleep 0.05; done\n',
    )
    write_executable(bin_dir / "ping", "exit 0\n")
    write_executable(
        bin_dir / "powerprofilesctl",
        'if [[ "$1" == "get" ]]; then\n'
        '  [[ -f "$POWER_STATE" ]] && cat "$POWER_STATE" || echo balanced\n'
        'else\n'
        '  printf \'power set %s\\n\' "$2" >> "$CALL_LOG"\n'
        '  printf \'%s\\n\' "$2" > "$POWER_STATE"\n'
        'fi\n',
    )
    write_executable(
        bin_dir / "gsettings",
        'if [[ "$1" == "get" ]]; then\n'
        '  [[ -f "$BATTERY_STATE" ]] && cat "$BATTERY_STATE" || echo true\n'
        'else\n'
        '  printf \'battery set %s\\n\' "$4" >> "$CALL_LOG"\n'
        '  printf \'%s\\n\' "$4" > "$BATTERY_STATE"\n'
        'fi\n',
    )
    write_executable(bin_dir / "gamemoderun", 'exec "$@"\n')
    env = os.environ.copy()
    for name in LAUNCH_ENV:
        env.pop(name, None)
    env.update({
        "BATTERY_STATE": str(battery_state),
        "CALL_LOG": str(call_log),
        "CTRL": "drone",
        "IP": "192.168.42.1",
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "POWER_STATE": str(power_state),
        "READY": str(ready),
        "SFM_LAUNCH_DRY_RUN": "0",
        "SFM_UI_PYTHON": str(fake_python),
    })
    process = subprocess.Popen(
        [str(SCRIPT)],
        cwd=SCRIPT.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        for _ in range(100):
            if ready.exists():
                break
            time.sleep(0.01)
        assert ready.exists(), "fake UI did not start"
        os.kill(process.pid, signal_number)
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    assert process.returncode == expected_returncode, (stdout, stderr)
    calls = call_log.read_text().splitlines()
    assert calls == [
        "power set performance",
        "battery set false",
        f"python {signal_label}",
        "battery set true",
        "power set balanced",
    ]


def test_live_max_performance_zero_skips_all_tuning(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "calls.log"
    fake_python = tmp_path / "python"
    write_executable(fake_python, 'printf \'python %s\\n\' "$*" >> "$CALL_LOG"\n')
    write_executable(bin_dir / "ping", "exit 0\n")
    for name in ("powerprofilesctl", "gsettings", "gamemoderun"):
        write_executable(
            bin_dir / name,
            f'printf \'{name}\\n\' >> "$CALL_LOG"\nexit 1\n',
        )

    result = run_launcher(
        CALL_LOG=str(call_log),
        CTRL="drone",
        IP="192.168.42.1",
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        SFM_LAUNCH_DRY_RUN="0",
        SFM_MAX_PERFORMANCE="0",
        SFM_UI_PYTHON=str(fake_python),
    )

    assert result.returncode == 0, result.stderr
    assert call_log.read_text().splitlines() == [
        (
            f"python -u {APP} --interface real-flight "
            "--ip 192.168.42.1 "
            "--controller drone --no-live-detect --max-altitude-m 50 "
            "--max-distance-m 100 --distance-geofence --nudge-pct 8 "
            "--nudge-pulse-s 0.20"
        )
    ]


@pytest.mark.parametrize(
    "args",
    (
        ("--video", "/tmp/replay.mp4"),
        ("--video=/tmp/replay.mp4",),
        ("--interface", "simulated-stream"),
        ("--interface=simulated-stream",),
    ),
)
def test_real_launcher_rejects_cross_interface_arguments(args: tuple[str, ...]):
    result = run_launcher(*args)

    assert result.returncode == 2
    assert "rejects cross-interface argument" in result.stderr
    assert "dry-run command" not in result.stdout
