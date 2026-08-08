from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading


VALIDATION = Path(__file__).resolve().parents[1]


def load_monitor_module():
    path = VALIDATION / "monitor_hardware.py"
    spec = importlib.util.spec_from_file_location("monitor_hardware_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_read_cpu_thermal_zones_reports_each_valid_zone_and_errors(tmp_path):
    monitor = load_monitor_module()
    thermal_root = tmp_path / "thermal"
    thermal_root.mkdir()
    zone0 = thermal_root / "thermal_zone0"
    zone0.mkdir()
    (zone0 / "type").write_text("x86_pkg_temp\n", encoding="utf-8")
    (zone0 / "temp").write_text("71500\n", encoding="utf-8")
    zone1 = thermal_root / "thermal_zone1"
    zone1.mkdir()
    (zone1 / "type").write_text("acpitz\n", encoding="utf-8")
    (zone1 / "temp").write_text("invalid\n", encoding="utf-8")

    result = monitor.read_cpu_thermal_zones(thermal_root)

    assert result["available"] is True
    assert result["max_temp_c"] == 71.5
    assert result["zones"] == [
        {"zone": "thermal_zone0", "type": "x86_pkg_temp", "temp_c": 71.5}
    ]
    assert any("thermal_zone1/temp" in error for error in result["errors"])


def test_read_cpu_thermal_zones_marks_missing_data(tmp_path):
    monitor = load_monitor_module()

    result = monitor.read_cpu_thermal_zones(tmp_path / "missing")

    assert result["available"] is False
    assert result["max_temp_c"] is None
    assert result["zones"] == []
    assert result["errors"]


def test_read_cpu_frequencies_converts_khz_and_summarizes_policies(tmp_path):
    monitor = load_monitor_module()
    for policy, current, maximum in (
        ("policy0", "3400000", "4700000"),
        ("policy1", "2000000", "3600000"),
    ):
        path = tmp_path / "cpufreq" / policy
        path.mkdir(parents=True)
        (path / "scaling_cur_freq").write_text(current, encoding="utf-8")
        (path / "scaling_max_freq").write_text(maximum, encoding="utf-8")

    result = monitor.read_cpu_frequencies(tmp_path)

    assert result["available"] is True
    assert result["policies"][0] == {
        "policy": "policy0",
        "available": True,
        "current_mhz": 3400.0,
        "scaling_max_mhz": 4700.0,
    }
    assert result["current_mhz"] == {"min": 2000.0, "mean": 2700.0, "max": 3400.0}
    assert result["scaling_max_mhz"] == {
        "min": 3600.0,
        "mean": 4150.0,
        "max": 4700.0,
    }


def test_read_cpu_frequencies_marks_missing_data(tmp_path):
    monitor = load_monitor_module()

    result = monitor.read_cpu_frequencies(tmp_path)

    assert result["available"] is False
    assert result["policies"] == []
    assert result["current_mhz"]["mean"] is None
    assert result["errors"]


def test_read_cpu_thermal_throttle_keeps_each_cpu_and_maximum(tmp_path):
    monitor = load_monitor_module()
    values_by_cpu = {
        "cpu0": (2, 10, 3, 20),
        "cpu1": (7, 15, 4, 25),
    }
    for cpu, values in values_by_cpu.items():
        path = tmp_path / cpu / "thermal_throttle"
        path.mkdir(parents=True)
        for filename, value in zip(monitor.CPU_THROTTLE_FILES, values):
            (path / filename).write_text(str(value), encoding="utf-8")

    result = monitor.read_cpu_thermal_throttle(tmp_path)

    assert result["available"] is True
    assert result["cpus"][1]["cpu"] == "cpu1"
    assert result["cpus"][1]["core_throttle_count"] == 7
    assert result["max"] == {
        "core_throttle_count": 7,
        "core_throttle_total_time_ms": 15,
        "package_throttle_count": 4,
        "package_throttle_total_time_ms": 25,
    }


def test_read_cpu_thermal_throttle_marks_missing_data(tmp_path):
    monitor = load_monitor_module()

    result = monitor.read_cpu_thermal_throttle(tmp_path)

    assert result["available"] is False
    assert result["cpus"] == []
    assert result["max"]["package_throttle_count"] is None
    assert result["errors"]


def test_query_nvidia_smi_parses_metrics_and_throttle_reasons():
    monitor = load_monitor_module()
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        stdout = (
            "0, NVIDIA Test GPU, 71, 55, 49.5, 2400, 12001, P0, "
            "0x0004, Active, Not Active, Not Active, Active\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    result = monitor.query_nvidia_smi(timeout_s=1.5, runner=runner)

    assert result["available"] is True
    assert result["error"] is None
    assert result["thermal_throttling_active"] is True
    assert result["power_limit_active"] is True
    assert result["gpus"][0]["temperature_c"] == 71.0
    assert result["gpus"][0]["graphics_clock_mhz"] == 2400.0
    assert result["gpus"][0]["clock_event_reasons"]["hw_thermal_slowdown"] == "Active"
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == 1.5


def test_query_nvidia_smi_falls_back_when_reason_query_is_unsupported():
    monitor = load_monitor_module()
    call_count = 0

    def runner(command, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise subprocess.CalledProcessError(1, command, stderr="unsupported field")
        stdout = "0, NVIDIA Test GPU, 60, 25, 30.0, 1800, 9000, P2\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    result = monitor.query_nvidia_smi(runner=runner)

    assert result["available"] is True
    assert result["reason_error"] == "unsupported field"
    assert result["gpus"][0]["temperature_c"] == 60.0
    assert result["gpus"][0]["clock_event_reasons"]["active_mask"] is None


def test_query_nvidia_smi_marks_missing_command():
    monitor = load_monitor_module()

    def runner(_command, **_kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    result = monitor.query_nvidia_smi(runner=runner)

    assert result["available"] is False
    assert result["gpus"] == []
    assert "nvidia-smi not found" in result["error"]


def sample() -> dict:
    return {
        "schema_version": 1,
        "timestamp_utc": "2026-07-13T00:00:00Z",
        "elapsed_s": 0.1,
        "cpu_thermal": {
            "available": True,
            "max_temp_c": 70.0,
            "zones": [{"zone": "thermal_zone0", "type": "cpu", "temp_c": 70.0}],
            "errors": [],
        },
        "cpu_frequency": {
            "available": True,
            "policies": [{
                "policy": "policy0",
                "available": True,
                "current_mhz": 3400.0,
                "scaling_max_mhz": 4700.0,
            }],
            "current_mhz": {"min": 3400.0, "mean": 3400.0, "max": 3400.0},
            "scaling_max_mhz": {"min": 4700.0, "mean": 4700.0, "max": 4700.0},
            "errors": [],
        },
        "cpu_thermal_throttle": {
            "available": True,
            "cpus": [{
                "cpu": "cpu0",
                "available": True,
                "core_throttle_count": 2,
                "core_throttle_total_time_ms": 10,
                "package_throttle_count": 3,
                "package_throttle_total_time_ms": 20,
            }],
            "max": {
                "core_throttle_count": 2,
                "core_throttle_total_time_ms": 10,
                "package_throttle_count": 3,
                "package_throttle_total_time_ms": 20,
            },
            "errors": [],
        },
        "nvidia": {
            "available": True,
            "error": None,
            "reason_error": None,
            "gpus": [{
                "index": 0,
                "name": "GPU",
                "temperature_c": 65.0,
                "utilization_percent": 90.0,
                "power_w": 50.0,
                "graphics_clock_mhz": 2200.0,
                "memory_clock_mhz": 12000.0,
                "thermal_throttling_active": False,
                "power_limit_active": True,
                "clock_event_reasons": {"sw_power_cap": "Active"},
            }],
            "thermal_throttling_active": False,
            "power_limit_active": True,
        },
    }


def test_run_monitor_writes_jsonl_and_handles_keyboard_interrupt(tmp_path):
    monitor = load_monitor_module()
    output = tmp_path / "hardware.jsonl"
    calls = 0

    def collector(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return sample()

    count = monitor.run_monitor(
        output=output,
        output_format="jsonl",
        interval_s=0.001,
        max_samples=0,
        stop_event=threading.Event(),
        collector=collector,
    )

    assert count == 1
    assert json.loads(output.read_text(encoding="utf-8"))["nvidia"]["available"] is True


def test_run_monitor_writes_flat_csv_summary(tmp_path):
    monitor = load_monitor_module()
    output = tmp_path / "hardware.csv"

    count = monitor.run_monitor(
        output=output,
        output_format="csv",
        interval_s=1.0,
        max_samples=1,
        stop_event=threading.Event(),
        collector=lambda **_kwargs: sample(),
    )

    assert count == 1
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["cpu_max_temp_c"] == "70.0"
    assert row["cpu_current_freq_mean_mhz"] == "3400.0"
    assert row["cpu_package_throttle_count_max"] == "3"
    assert row["gpu_max_temp_c"] == "65.0"
    assert row["gpu_thermal_throttling_active"] == "False"
    assert json.loads(row["gpus_json"])[0]["clock_event_reasons"]["sw_power_cap"] == "Active"


def test_run_monitor_can_write_one_read_only_sample_to_stdout(capsys):
    monitor = load_monitor_module()

    count = monitor.run_monitor(
        output=Path("-"),
        output_format="jsonl",
        interval_s=1.0,
        max_samples=1,
        stop_event=threading.Event(),
        collector=lambda **_kwargs: sample(),
    )

    assert count == 1
    assert json.loads(capsys.readouterr().out)["schema_version"] == 1


def test_thermal_warning_messages_detects_cpu_delta_and_gpu_slowdown():
    monitor = load_monitor_module()
    previous = sample()
    current = sample()
    current_cpu = current["cpu_thermal_throttle"]["cpus"][0]
    current_cpu["core_throttle_count"] = 5
    current_cpu["package_throttle_total_time_ms"] = 31
    current["nvidia"]["thermal_throttling_active"] = True

    warnings = dict(monitor.thermal_warning_messages(previous, current))

    assert warnings["gpu-thermal-throttling"] == "NVIDIA reports active thermal slowdown"
    assert "core count +3" in warnings["cpu-thermal-throttling"]
    assert "package time +11 ms" in warnings["cpu-thermal-throttling"]


def test_run_monitor_rate_limits_repeated_thermal_warnings(tmp_path):
    monitor = load_monitor_module()
    output = tmp_path / "hardware.jsonl"
    calls = 0
    warnings = []

    def collector(**_kwargs):
        nonlocal calls
        current = sample()
        current["cpu_thermal_throttle"]["cpus"][0]["package_throttle_count"] += calls
        current["nvidia"]["thermal_throttling_active"] = True
        calls += 1
        return current

    count = monitor.run_monitor(
        output=output,
        output_format="jsonl",
        interval_s=0.001,
        max_samples=3,
        stop_event=threading.Event(),
        warning_interval_s=60.0,
        collector=collector,
        warning_sink=warnings.append,
    )

    assert count == 3
    assert warnings.count("WARNING: NVIDIA reports active thermal slowdown") == 1
    assert sum("CPU thermal-throttle counters increased" in item for item in warnings) == 1
