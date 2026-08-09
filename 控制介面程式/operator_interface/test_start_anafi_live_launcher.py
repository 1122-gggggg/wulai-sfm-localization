from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("start_anafi_live.sh")
APP = SCRIPT.with_name("flight_operator_app.py")
PACKAGE_MANIFEST_TOOL = SCRIPT.parents[2] / "tools" / "package_manifest.py"
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
    "SFM_DISTANCE_GEOFENCE",
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


def test_app_help_formats_percent_literals() -> None:
    result = subprocess.run(
        [sys.executable, str(APP), "--help"],
        cwd=SCRIPT.parent,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "at 95% of the confirmed limit" in result.stdout


def test_launcher_help_exits_before_live_setup() -> None:
    result = run_launcher("--help")

    assert result.returncode == 0, result.stderr
    assert "只會啟動 real-flight Olympe 介面" in result.stdout
    assert "checking target" not in result.stdout
    assert "sustained CPU thread budget" not in result.stdout


def write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)


def make_fake_portable_package(tmp_path: Path) -> Path:
    package = tmp_path / "portable"
    for relative in (
        Path("控制介面程式/operator_interface/start_anafi_live.sh"),
        Path("控制介面程式/真機串流/啟動.sh"),
    ):
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SCRIPT.parents[2] / relative, destination)
    display = package / "控制介面程式/operator_interface/resolve_display.sh"
    display.parent.mkdir(parents=True, exist_ok=True)
    display.write_text(
        "configure_operator_display() { export DISPLAY=:0; }\n",
        encoding="utf-8",
    )
    manifest_tool = package / "tools/package_manifest.py"
    manifest_tool.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PACKAGE_MANIFEST_TOOL, manifest_tool)
    (package / "PORTABLE_PACKAGE.json").write_text(
        '{"schema":"sfm-portable-live-runtime/v1",'
        '"package_kind":"live-operator-runtime",'
        '"entrypoint":"一鍵啟動.sh"}\n',
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, str(manifest_tool), "generate", "--root", str(package)],
        check=True,
        capture_output=True,
        text=True,
    )
    return package


def run_portable_launcher(
    launcher: Path, *, dry_run: str, package: Path, **overrides: str
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in LAUNCH_ENV | {
        "SFM_PYTHON",
        "SFM_VENV_DIR",
        "SFM_WORKSPACE_ROOT",
        "SFM_PORTABLE_ALLOW_EXTERNAL_PYTHON",
    }:
        env.pop(name, None)
    env.update(
        {
            "SFM_LAUNCH_DRY_RUN": dry_run,
            "SFM_MAX_PERFORMANCE": "0",
            "SFM_UI_PYTHON": sys.executable,
            "SFM_SITE_PROFILE": str(package / "profile.json"),
            **overrides,
        }
    )
    return subprocess.run(
        [str(launcher)],
        cwd=launcher.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )


@pytest.mark.parametrize(
    "relative_launcher",
    (
        Path("控制介面程式/operator_interface/start_anafi_live.sh"),
        Path("控制介面程式/真機串流/啟動.sh"),
    ),
)
def test_tampered_portable_manifest_fails_before_reachability(
    tmp_path: Path, relative_launcher: Path
) -> None:
    package = make_fake_portable_package(tmp_path)
    manifest = package / "MANIFEST.tsv"
    rows = manifest.read_text(encoding="utf-8").splitlines()
    size, _digest, relative = rows[1].split("\t", 2)
    rows[1] = f"{size}\t{'0' * 64}\t{relative}"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    call_log = tmp_path / "calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_executable(bin_dir / "ping", 'printf \'ping\\n\' >> "$CALL_LOG"\nexit 0\n')

    result = run_portable_launcher(
        package / relative_launcher,
        dry_run="0",
        package=package,
        CALL_LOG=str(call_log),
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        IP="192.168.42.1",
        CTRL="drone",
    )

    assert result.returncode == 1, (result.stdout, result.stderr)
    assert "SHA-256 mismatch" in result.stdout
    assert not call_log.exists(), "manifest verification must precede reachability"
    assert "dry-run command" not in result.stdout


@pytest.mark.parametrize(
    "relative_launcher",
    (
        Path("控制介面程式/operator_interface/start_anafi_live.sh"),
        Path("控制介面程式/真機串流/啟動.sh"),
    ),
)
def test_valid_portable_manifest_reaches_dry_run_without_connecting(
    tmp_path: Path, relative_launcher: Path
) -> None:
    package = make_fake_portable_package(tmp_path)
    result = run_portable_launcher(
        package / relative_launcher,
        dry_run="1",
        package=package,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "portable package manifest OK" in result.stdout
    assert "dry-run command" in result.stdout
    assert "checking target" not in result.stdout


@pytest.mark.parametrize(
    "relative_launcher",
    (
        Path("控制介面程式/operator_interface/start_anafi_live.sh"),
        Path("控制介面程式/真機串流/啟動.sh"),
    ),
)
def test_non_git_runtime_rejects_missing_portable_metadata(
    tmp_path: Path, relative_launcher: Path
) -> None:
    package = make_fake_portable_package(tmp_path)
    (package / "PORTABLE_PACKAGE.json").unlink()

    result = run_portable_launcher(
        package / relative_launcher,
        dry_run="1",
        package=package,
    )

    assert result.returncode == 2
    assert "missing PORTABLE_PACKAGE.json" in result.stderr
    assert "dry-run command" not in result.stdout


@pytest.mark.parametrize(
    "relative_launcher",
    (
        Path("控制介面程式/operator_interface/start_anafi_live.sh"),
        Path("控制介面程式/真機串流/啟動.sh"),
    ),
)
def test_portable_launcher_rejects_unbound_package_local_venv(
    tmp_path: Path, relative_launcher: Path
) -> None:
    package = make_fake_portable_package(tmp_path)
    fake_python = package / ".venv/bin/python"
    fake_python.parent.mkdir(parents=True)
    write_executable(fake_python, "exit 99\n")

    result = run_portable_launcher(
        package / relative_launcher,
        dry_run="1",
        package=package,
        SFM_UI_PYTHON=str(fake_python),
    )

    assert result.returncode == 2
    assert "not bound to this portable manifest" in result.stderr
    assert "dry-run command" not in result.stdout


def test_direct_ip_derives_drone_and_preserves_benchmark_flags():
    result = run_launcher(
        IP="192.168.42.1",
        LOC_BENCH_TRACK="1",
        LOC_EVERY_N="2",
    )

    assert result.returncode == 0, result.stderr
    assert "--ip 192.168.42.1 --controller drone" in result.stdout
    assert "--no-live-detect" in result.stdout
    assert "--max-altitude-m 50 --max-distance-m 100" in result.stdout
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
    assert "--max-altitude-m 18 --max-distance-m 75" in result.stdout


def test_launcher_leaves_the_distance_geofence_to_the_documented_env_switch():
    """The launcher must NOT pin the flag.

    It used to always pass "${SFM_DISTANCE_GEOFENCE_FLAG:---no-distance-geofence}",
    and an explicit CLI flag beats the env-derived argparse default -- so
    SFM_DISTANCE_GEOFENCE, the switch flight_operator_app's --help advertises,
    could never take effect through the supported launch path. The OFF default is
    unchanged: it now comes from env_bool("SFM_DISTANCE_GEOFENCE", False).
    """
    off = run_launcher()
    assert off.returncode == 0, off.stderr
    # "distance-geofence", not "--distance-geofence": the latter does not appear
    # inside "--no-distance-geofence", so it would pass against the very defect
    # this guards -- the launcher pinning the flag off.
    assert "distance-geofence" not in off.stdout

    on = run_launcher(SFM_DISTANCE_GEOFENCE="1")
    assert on.returncode == 0, on.stderr
    # Still not pinned on the command line; the app reads the variable itself,
    # which it inherits from this environment.
    assert "distance-geofence" not in on.stdout.replace("SFM_DISTANCE_GEOFENCE=1", "")
    assert "SFM_DISTANCE_GEOFENCE=1" in on.stdout


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


def test_live_launcher_rejects_system_site_packages_venv(tmp_path: Path):
    venv_dir = tmp_path / "venv"
    bin_dir = venv_dir / "bin"
    bin_dir.mkdir(parents=True)
    fake_python = bin_dir / "python"
    write_executable(fake_python, "exit 0\n")
    (venv_dir / "pyvenv.cfg").write_text(
        "include-system-site-packages = true\n"
    )

    result = run_launcher(
        SFM_LAUNCH_DRY_RUN="0",
        SFM_MAX_PERFORMANCE="0",
        SFM_UI_PYTHON=str(fake_python),
    )

    assert result.returncode == 2
    assert "include-system-site-packages=true" in result.stderr
    assert "checking target" not in result.stdout


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
            "--max-distance-m 100 "
            "--rth-min-altitude-m 20.0 --stream-loss-grace-s 10.0 --nudge-pct 8 "
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
