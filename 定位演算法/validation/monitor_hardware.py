#!/usr/bin/env python3
"""Log thermal and GPU performance telemetry without changing hardware state.

Examples:
  python3 monitor_hardware.py --output outputs/hardware.jsonl
  python3 monitor_hardware.py --output outputs/hardware.csv --interval 0.5

The monitor is read-only. It emits rate-limited stderr warnings when Linux CPU
thermal-throttle counters increase or NVIDIA reports thermal slowdown.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, TextIO


THERMAL_ROOT = Path("/sys/class/thermal")
CPU_ROOT = Path("/sys/devices/system/cpu")
NVIDIA_CORE_FIELDS = (
    "index",
    "name",
    "temperature.gpu",
    "utilization.gpu",
    "power.draw",
    "clocks.current.graphics",
    "clocks.current.memory",
    "pstate",
)
NVIDIA_REASON_FIELDS = (
    "clocks_event_reasons.active",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
    "clocks_event_reasons.sw_power_cap",
)
NVIDIA_FIELDS = NVIDIA_CORE_FIELDS + NVIDIA_REASON_FIELDS
CSV_FIELDS = (
    "timestamp_utc",
    "elapsed_s",
    "cpu_thermal_available",
    "cpu_max_temp_c",
    "cpu_thermal_zones_json",
    "cpu_thermal_errors_json",
    "cpu_frequency_available",
    "cpu_current_freq_min_mhz",
    "cpu_current_freq_mean_mhz",
    "cpu_current_freq_max_mhz",
    "cpu_scaling_max_freq_min_mhz",
    "cpu_scaling_max_freq_mean_mhz",
    "cpu_scaling_max_freq_max_mhz",
    "cpu_frequency_policies_json",
    "cpu_frequency_errors_json",
    "cpu_throttle_available",
    "cpu_core_throttle_count_max",
    "cpu_core_throttle_total_time_ms_max",
    "cpu_package_throttle_count_max",
    "cpu_package_throttle_total_time_ms_max",
    "cpu_throttle_cpus_json",
    "cpu_throttle_errors_json",
    "nvidia_available",
    "nvidia_error",
    "nvidia_reason_error",
    "gpu_count",
    "gpu_max_temp_c",
    "gpu_max_utilization_percent",
    "gpu_total_power_w",
    "gpu_min_graphics_clock_mhz",
    "gpu_min_memory_clock_mhz",
    "gpu_thermal_throttling_active",
    "gpu_power_limit_active",
    "gpus_json",
)


def read_cpu_thermal_zones(root: Path = THERMAL_ROOT) -> dict[str, Any]:
    """Read Linux thermal-zone temperatures, which are exposed in millidegrees C."""
    zones: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        zone_paths = sorted(root.glob("thermal_zone*"), key=lambda path: path.name)
    except OSError as exc:
        zone_paths = []
        errors.append(f"cannot list {root}: {exc}")

    for zone_path in zone_paths:
        try:
            raw_temp = (zone_path / "temp").read_text(encoding="utf-8").strip()
            temp_c = int(raw_temp) / 1000.0
        except (OSError, ValueError) as exc:
            errors.append(f"{zone_path.name}/temp: {exc}")
            continue

        try:
            zone_type = (zone_path / "type").read_text(encoding="utf-8").strip()
        except OSError as exc:
            zone_type = "unknown"
            errors.append(f"{zone_path.name}/type: {exc}")
        zones.append({"zone": zone_path.name, "type": zone_type, "temp_c": temp_c})

    if not zones and not errors:
        errors.append(f"no readable thermal zones under {root}")
    return {
        "available": bool(zones),
        "max_temp_c": max((zone["temp_c"] for zone in zones), default=None),
        "zones": zones,
        "errors": errors,
    }


def _numeric_suffix(path: Path) -> tuple[str, int]:
    prefix = path.name.rstrip("0123456789")
    suffix = path.name[len(prefix):]
    return prefix, int(suffix) if suffix else -1


def _summary(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values, default=None),
        "mean": round(sum(values) / len(values), 6) if values else None,
        "max": max(values, default=None),
    }


def read_cpu_frequencies(cpu_root: Path = CPU_ROOT) -> dict[str, Any]:
    """Read cpufreq policy clocks. Linux exposes these values in kHz."""
    policies: list[dict[str, Any]] = []
    errors: list[str] = []
    policy_root = cpu_root / "cpufreq"
    try:
        policy_paths = sorted(policy_root.glob("policy*"), key=_numeric_suffix)
    except OSError as exc:
        policy_paths = []
        errors.append(f"cannot list {policy_root}: {exc}")

    for policy_path in policy_paths:
        values: dict[str, float | None] = {}
        for output_key, filename in (
            ("current_mhz", "scaling_cur_freq"),
            ("scaling_max_mhz", "scaling_max_freq"),
        ):
            try:
                raw_khz = (policy_path / filename).read_text(encoding="utf-8").strip()
                values[output_key] = int(raw_khz) / 1000.0
            except (OSError, ValueError) as exc:
                values[output_key] = None
                errors.append(f"{policy_path.name}/{filename}: {exc}")
        policies.append({
            "policy": policy_path.name,
            "available": all(value is not None for value in values.values()),
            **values,
        })

    complete = [policy for policy in policies if policy["available"]]
    if not policies and not errors:
        errors.append(f"no cpufreq policies under {policy_root}")
    return {
        "available": bool(complete),
        "policies": policies,
        "current_mhz": _summary([policy["current_mhz"] for policy in complete]),
        "scaling_max_mhz": _summary([
            policy["scaling_max_mhz"] for policy in complete
        ]),
        "errors": errors,
    }


CPU_THROTTLE_FILES = (
    "core_throttle_count",
    "core_throttle_total_time_ms",
    "package_throttle_count",
    "package_throttle_total_time_ms",
)


def read_cpu_thermal_throttle(cpu_root: Path = CPU_ROOT) -> dict[str, Any]:
    """Read cumulative x86 thermal-throttle counters for each logical CPU."""
    cpus: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        throttle_paths = sorted(
            cpu_root.glob("cpu[0-9]*/thermal_throttle"),
            key=lambda path: _numeric_suffix(path.parent),
        )
    except OSError as exc:
        throttle_paths = []
        errors.append(f"cannot list CPU throttle counters under {cpu_root}: {exc}")

    for throttle_path in throttle_paths:
        values: dict[str, int | None] = {}
        for filename in CPU_THROTTLE_FILES:
            try:
                raw_value = (throttle_path / filename).read_text(encoding="utf-8").strip()
                values[filename] = int(raw_value)
            except (OSError, ValueError) as exc:
                values[filename] = None
                errors.append(f"{throttle_path.parent.name}/{filename}: {exc}")
        cpus.append({
            "cpu": throttle_path.parent.name,
            "available": all(value is not None for value in values.values()),
            **values,
        })

    complete = [cpu for cpu in cpus if cpu["available"]]
    if not throttle_paths and not errors:
        errors.append(f"no thermal_throttle counters under {cpu_root}")
    return {
        "available": bool(complete),
        "cpus": cpus,
        "max": {
            filename: max((cpu[filename] for cpu in complete), default=None)
            for filename in CPU_THROTTLE_FILES
        },
        "errors": errors,
    }


def _optional_float(value: str) -> float | None:
    normalized = value.strip()
    if not normalized or normalized.lower() in {
        "n/a", "[n/a]", "not supported", "[not supported]",
    }:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def _reason_active(value: str | None) -> bool:
    return isinstance(value, str) and value.strip().lower() == "active"


def _parse_nvidia_rows(output: str, fields: tuple[str, ...]) -> tuple[list[dict[str, str]], list[str]]:
    parsed: list[dict[str, str]] = []
    errors: list[str] = []
    for row_number, row in enumerate(csv.reader(output.splitlines()), start=1):
        if not row:
            continue
        if len(row) != len(fields):
            errors.append(
                f"nvidia-smi row {row_number} has {len(row)} columns; expected {len(fields)}"
            )
            continue
        parsed.append({field: value.strip() for field, value in zip(fields, row)})
    return parsed, errors


def _query_nvidia_fields(
    fields: tuple[str, ...],
    *,
    timeout_s: float,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> tuple[list[dict[str, str]], list[str]]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    completed = runner(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return _parse_nvidia_rows(completed.stdout, fields)


def _command_error(exc: BaseException) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        stderr = (exc.stderr or "").strip()
        if stderr:
            return stderr
    return str(exc) or exc.__class__.__name__


def query_nvidia_smi(
    *,
    timeout_s: float = 2.0,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Query all NVIDIA GPUs. If reason fields fail, retain the core metrics."""
    reason_error: str | None = None
    try:
        rows, parse_errors = _query_nvidia_fields(
            NVIDIA_FIELDS, timeout_s=timeout_s, runner=runner,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        reason_error = _command_error(exc)
        try:
            rows, parse_errors = _query_nvidia_fields(
                NVIDIA_CORE_FIELDS, timeout_s=timeout_s, runner=runner,
            )
        except (OSError, subprocess.SubprocessError) as core_exc:
            return {
                "available": False,
                "error": _command_error(core_exc),
                "reason_error": reason_error,
                "gpus": [],
                "thermal_throttling_active": False,
                "power_limit_active": False,
            }

    gpus: list[dict[str, Any]] = []
    for row in rows:
        reasons = {
            "active_mask": row.get("clocks_event_reasons.active"),
            "hw_thermal_slowdown": row.get(
                "clocks_event_reasons.hw_thermal_slowdown"
            ),
            "sw_thermal_slowdown": row.get(
                "clocks_event_reasons.sw_thermal_slowdown"
            ),
            "hw_power_brake_slowdown": row.get(
                "clocks_event_reasons.hw_power_brake_slowdown"
            ),
            "sw_power_cap": row.get("clocks_event_reasons.sw_power_cap"),
        }
        thermal_active = any(_reason_active(reasons[key]) for key in (
            "hw_thermal_slowdown", "sw_thermal_slowdown",
        ))
        power_limit_active = any(_reason_active(reasons[key]) for key in (
            "hw_power_brake_slowdown", "sw_power_cap",
        ))
        index_value = _optional_float(row.get("index", ""))
        gpus.append({
            "index": int(index_value) if index_value is not None else None,
            "name": row.get("name"),
            "temperature_c": _optional_float(row.get("temperature.gpu", "")),
            "utilization_percent": _optional_float(row.get("utilization.gpu", "")),
            "power_w": _optional_float(row.get("power.draw", "")),
            "graphics_clock_mhz": _optional_float(
                row.get("clocks.current.graphics", "")
            ),
            "memory_clock_mhz": _optional_float(row.get("clocks.current.memory", "")),
            "pstate": row.get("pstate"),
            "clock_event_reasons": reasons,
            "thermal_throttling_active": thermal_active,
            "power_limit_active": power_limit_active,
        })

    error = "; ".join(parse_errors) if parse_errors else None
    return {
        "available": True,
        "error": error,
        "reason_error": reason_error,
        "gpus": gpus,
        "thermal_throttling_active": any(
            gpu["thermal_throttling_active"] for gpu in gpus
        ),
        "power_limit_active": any(gpu["power_limit_active"] for gpu in gpus),
    }


def collect_sample(
    *,
    started_monotonic: float,
    thermal_root: Path = THERMAL_ROOT,
    cpu_root: Path = CPU_ROOT,
    nvidia_timeout_s: float = 2.0,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "elapsed_s": round(time.monotonic() - started_monotonic, 6),
        "cpu_thermal": read_cpu_thermal_zones(thermal_root),
        "cpu_frequency": read_cpu_frequencies(cpu_root),
        "cpu_thermal_throttle": read_cpu_thermal_throttle(cpu_root),
        "nvidia": query_nvidia_smi(timeout_s=nvidia_timeout_s),
    }


def _present_values(gpus: list[dict[str, Any]], key: str) -> list[float]:
    return [float(gpu[key]) for gpu in gpus if gpu.get(key) is not None]


def flatten_sample(sample: dict[str, Any]) -> dict[str, Any]:
    cpu = sample["cpu_thermal"]
    cpu_frequency = sample["cpu_frequency"]
    cpu_throttle = sample["cpu_thermal_throttle"]
    nvidia = sample["nvidia"]
    gpus = nvidia["gpus"]
    temperatures = _present_values(gpus, "temperature_c")
    utilizations = _present_values(gpus, "utilization_percent")
    powers = _present_values(gpus, "power_w")
    graphics_clocks = _present_values(gpus, "graphics_clock_mhz")
    memory_clocks = _present_values(gpus, "memory_clock_mhz")
    return {
        "timestamp_utc": sample["timestamp_utc"],
        "elapsed_s": sample["elapsed_s"],
        "cpu_thermal_available": cpu["available"],
        "cpu_max_temp_c": cpu["max_temp_c"],
        "cpu_thermal_zones_json": json.dumps(cpu["zones"], ensure_ascii=False),
        "cpu_thermal_errors_json": json.dumps(cpu["errors"], ensure_ascii=False),
        "cpu_frequency_available": cpu_frequency["available"],
        "cpu_current_freq_min_mhz": cpu_frequency["current_mhz"]["min"],
        "cpu_current_freq_mean_mhz": cpu_frequency["current_mhz"]["mean"],
        "cpu_current_freq_max_mhz": cpu_frequency["current_mhz"]["max"],
        "cpu_scaling_max_freq_min_mhz": cpu_frequency["scaling_max_mhz"]["min"],
        "cpu_scaling_max_freq_mean_mhz": cpu_frequency["scaling_max_mhz"]["mean"],
        "cpu_scaling_max_freq_max_mhz": cpu_frequency["scaling_max_mhz"]["max"],
        "cpu_frequency_policies_json": json.dumps(
            cpu_frequency["policies"], ensure_ascii=False
        ),
        "cpu_frequency_errors_json": json.dumps(
            cpu_frequency["errors"], ensure_ascii=False
        ),
        "cpu_throttle_available": cpu_throttle["available"],
        "cpu_core_throttle_count_max": cpu_throttle["max"]["core_throttle_count"],
        "cpu_core_throttle_total_time_ms_max": cpu_throttle["max"][
            "core_throttle_total_time_ms"
        ],
        "cpu_package_throttle_count_max": cpu_throttle["max"][
            "package_throttle_count"
        ],
        "cpu_package_throttle_total_time_ms_max": cpu_throttle["max"][
            "package_throttle_total_time_ms"
        ],
        "cpu_throttle_cpus_json": json.dumps(cpu_throttle["cpus"], ensure_ascii=False),
        "cpu_throttle_errors_json": json.dumps(
            cpu_throttle["errors"], ensure_ascii=False
        ),
        "nvidia_available": nvidia["available"],
        "nvidia_error": nvidia["error"],
        "nvidia_reason_error": nvidia["reason_error"],
        "gpu_count": len(gpus),
        "gpu_max_temp_c": max(temperatures, default=None),
        "gpu_max_utilization_percent": max(utilizations, default=None),
        "gpu_total_power_w": sum(powers) if powers else None,
        "gpu_min_graphics_clock_mhz": min(graphics_clocks, default=None),
        "gpu_min_memory_clock_mhz": min(memory_clocks, default=None),
        "gpu_thermal_throttling_active": nvidia["thermal_throttling_active"],
        "gpu_power_limit_active": nvidia["power_limit_active"],
        "gpus_json": json.dumps(gpus, ensure_ascii=False),
    }


def thermal_warning_messages(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> list[tuple[str, str]]:
    """Return stable warning keys and messages without adding hardware reads."""
    warnings: list[tuple[str, str]] = []
    cpu_thermal = current["cpu_thermal"]
    cpu_throttle = current["cpu_thermal_throttle"]
    nvidia = current["nvidia"]

    if not cpu_thermal["available"]:
        warnings.append((
            "cpu-temperature-unavailable",
            "CPU temperature telemetry is unavailable",
        ))
    if not cpu_throttle["available"]:
        warnings.append((
            "cpu-throttle-unavailable",
            "CPU thermal-throttle counters are unavailable; CPU throttling cannot be confirmed",
        ))
    if not nvidia["available"]:
        warnings.append((
            "nvidia-unavailable",
            f"NVIDIA telemetry is unavailable: {nvidia.get('error') or 'unknown error'}",
        ))
    elif nvidia.get("reason_error"):
        warnings.append((
            "nvidia-reasons-unavailable",
            "NVIDIA slowdown-reason telemetry is unavailable; core GPU metrics were retained",
        ))
    if nvidia.get("thermal_throttling_active"):
        warnings.append((
            "gpu-thermal-throttling",
            "NVIDIA reports active thermal slowdown",
        ))

    if previous is None or not cpu_throttle["available"]:
        return warnings
    previous_throttle = previous["cpu_thermal_throttle"]
    if not previous_throttle["available"]:
        return warnings

    previous_cpus = {
        cpu["cpu"]: cpu for cpu in previous_throttle["cpus"] if cpu["available"]
    }
    current_cpus = {
        cpu["cpu"]: cpu for cpu in cpu_throttle["cpus"] if cpu["available"]
    }
    deltas: list[str] = []
    for label, field in (
        ("core count", "core_throttle_count"),
        ("core time", "core_throttle_total_time_ms"),
        ("package count", "package_throttle_count"),
        ("package time", "package_throttle_total_time_ms"),
    ):
        increases = [
            current_cpus[cpu_name][field] - previous_cpu[field]
            for cpu_name, previous_cpu in previous_cpus.items()
            if cpu_name in current_cpus
            and current_cpus[cpu_name][field] > previous_cpu[field]
        ]
        if increases:
            suffix = " ms" if field.endswith("_time_ms") else ""
            deltas.append(f"{label} +{max(increases)}{suffix}")
    if deltas:
        warnings.append((
            "cpu-thermal-throttling",
            "CPU thermal-throttle counters increased (" + ", ".join(deltas) + ")",
        ))
    return warnings


class SampleWriter:
    def __init__(self, handle: TextIO, output_format: str) -> None:
        self.handle = handle
        self.output_format = output_format
        self.csv_writer: csv.DictWriter[str] | None = None
        if output_format == "csv":
            self.csv_writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            self.csv_writer.writeheader()
            handle.flush()

    def write(self, sample: dict[str, Any]) -> None:
        if self.csv_writer is not None:
            self.csv_writer.writerow(flatten_sample(sample))
        else:
            self.handle.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
        self.handle.flush()


def run_monitor(
    *,
    output: Path,
    output_format: str,
    interval_s: float,
    max_samples: int,
    stop_event: threading.Event,
    thermal_root: Path = THERMAL_ROOT,
    cpu_root: Path = CPU_ROOT,
    nvidia_timeout_s: float = 2.0,
    warning_interval_s: float = 30.0,
    collector: Callable[..., dict[str, Any]] = collect_sample,
    warning_sink: Callable[[str], None] | None = None,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sample_count = 0
    previous_sample: dict[str, Any] | None = None
    warning_times: dict[str, float] = {}
    if warning_sink is None:
        warning_sink = lambda message: print(message, file=sys.stderr)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = SampleWriter(handle, output_format)
        try:
            while not stop_event.is_set():
                sample = collector(
                    started_monotonic=started,
                    thermal_root=thermal_root,
                    cpu_root=cpu_root,
                    nvidia_timeout_s=nvidia_timeout_s,
                )
                writer.write(sample)
                if warning_interval_s > 0:
                    warning_now = time.monotonic()
                    for warning_key, message in thermal_warning_messages(
                        previous_sample, sample
                    ):
                        last_warning = warning_times.get(warning_key)
                        if (
                            last_warning is None
                            or warning_now - last_warning >= warning_interval_s
                        ):
                            warning_sink(f"WARNING: {message}")
                            warning_times[warning_key] = warning_now
                previous_sample = sample
                sample_count += 1
                if max_samples and sample_count >= max_samples:
                    break
                next_sample_at = started + sample_count * interval_s
                if stop_event.wait(max(0.0, next_sample_at - time.monotonic())):
                    break
        except KeyboardInterrupt:
            stop_event.set()
    return sample_count


def _output_format(path: Path, requested: str | None) -> str:
    if requested:
        return requested
    return "csv" if path.suffix.lower() == ".csv" else "jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("jsonl", "csv"), default=None)
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    parser.add_argument(
        "--samples", type=int, default=0, help="stop after N samples; 0 runs until interrupted"
    )
    parser.add_argument("--nvidia-timeout", type=float, default=2.0)
    parser.add_argument(
        "--warning-interval",
        type=float,
        default=30.0,
        help="minimum seconds between warnings of the same type; 0 disables warnings",
    )
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    if args.samples < 0:
        parser.error("--samples must be zero or greater")
    if args.nvidia_timeout <= 0:
        parser.error("--nvidia-timeout must be greater than zero")
    if args.warning_interval < 0:
        parser.error("--warning-interval must be zero or greater")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    output_format = _output_format(args.output, args.format)
    sample_count = run_monitor(
        output=args.output,
        output_format=output_format,
        interval_s=args.interval,
        max_samples=args.samples,
        stop_event=stop_event,
        nvidia_timeout_s=args.nvidia_timeout,
        warning_interval_s=args.warning_interval,
    )
    print(
        f"hardware monitor wrote {sample_count} sample(s) to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
