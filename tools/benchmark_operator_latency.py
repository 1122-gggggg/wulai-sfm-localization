#!/usr/bin/env python3
"""Offline RGB-submit comparison and bounded simulated operator-window timing.

No aircraft backend is selected; --ui-seconds requires an explicit profile/video.
The microbenchmark compares the old copy with the current pipe submission path
in one process. UI timings include fresh asset validation, not cached receipts.
"""

from __future__ import annotations

import argparse
import functools
import json
import queue
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPERATOR = ROOT / "控制介面程式/operator_interface"


def _pipe_benchmark(client_type, iterations: int) -> dict:
    import numpy as np

    frame = np.arange(1280 * 720 * 3, dtype=np.uint8).tobytes()
    # Match FFmpeg's immutable ndarray view, not an artificial bytes-only input.
    decoded = np.frombuffer(frame, dtype=np.uint8).reshape(720, 1280, 3)
    raw = client_type._frame_memoryview(decoded)
    client = object.__new__(client_type)
    client._lock = threading.Lock()
    client.in_flight = True
    client._coalesce = None
    client._coalesce_drops = 0

    # Production coalescing is the Tk-thread hot path while the detector is busy.
    def previous():
        item = (1, "frame", b"", bytes(raw), {})
        with client._lock:
            return client._store_coalesced(item)

    def current():
        return client._submit_pipe_frame(1, "frame", b"", raw, {})

    samples = {"previous_copy": [], "current_submit": []}
    for index in range(iterations + 10):
        # Alternate order to avoid always giving one implementation warm caches.
        calls = [("previous_copy", previous), ("current_submit", current)]
        if index % 2:
            calls.reverse()
        for label, call in calls:
            start = time.perf_counter_ns()
            call()
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            if index >= 10:
                samples[label].append(elapsed)
    current()
    if client._coalesce[3] != frame:
        raise RuntimeError("pipe submission changed decoded RGB bytes")
    return {
        "frame_bytes": len(frame),
        "iterations": iterations,
        "scope": "CPU pipe queue/coalesce only; no IPC, model, GPU or network",
        "input": "readonly NumPy RGB array over immutable decoder bytes",
        "median_ms": {key: statistics.median(values) for key, values in samples.items()},
        "p95_ms": {
            key: sorted(values)[min(len(values) - 1, int(0.95 * len(values)))]
            for key, values in samples.items()
        },
        "immutable_frame_reused": client._coalesce[3] is frame,
    }


_QUEUE_EMPTY = object()


def _get_nowait(source):
    try:
        return source.get_nowait()
    except queue.Empty:
        return _QUEUE_EMPTY


def _drain_queue(source):
    batch = []
    while (item := _get_nowait(source)) is not _QUEUE_EMPTY:
        batch.append(item)
    return batch


_STARTUP_STAGES = (
    "_resolve_startup_site",
    "_load_startup_route",
    "_load_runtime_map",
    "_operator_session_identity",
    "_create_operator_session",
    "_build_operator_backend",
    "_build_operator_localizer",
    "_prepare_operator_launch_inputs",
    "_build_operator_launch",
)


def _install_stage_profiler(operator_tick):
    stage_samples: dict[str, list[float]] = {}
    original_detail = operator_tick._TickProfile.detail

    def detail(profile, name, elapsed_ms):
        original_detail(profile, name, elapsed_ms)
        bucket = stage_samples.setdefault(name, [])
        if len(bucket) < 18000:
            bucket.append(float(elapsed_ms))

    operator_tick._TickProfile.detail = detail
    return stage_samples, original_detail


def _time_startup_stages(operator_launch, timings):
    for name in _STARTUP_STAGES:
        original = getattr(operator_launch, name)

        def timed(*call_args, _original=original, _name=name, **kwargs):
            start = time.perf_counter()
            try:
                return _original(*call_args, **kwargs)
            finally:
                timings[_name] = (time.perf_counter() - start) * 1000

        setattr(operator_launch, name, functools.wraps(original)(timed))


def _grab_window(ui, path):
    from PIL import ImageGrab

    x, y = ui.winfo_rootx(), ui.winfo_rooty()
    ImageGrab.grab(bbox=(x, y, x + ui.winfo_width(), y + ui.winfo_height())).save(path)


def _observe_window_state(ui, launch, observed, now, smoke_start):
    if ui.winfo_viewable() and not observed["mapped"]:
        observed.update(
            mapped=True,
            startup_to_mapped_ms=(now - smoke_start) * 1000,
            width=ui.winfo_width(),
            height=ui.winfo_height(),
        )
    if launch.localizer is not None and launch.localizer.ready and not observed["worker_ready"]:
        observed.update(worker_ready=True, startup_to_worker_ready_ms=(now - smoke_start) * 1000)
    if getattr(ui, "live_result", None) is not None and not observed["result_received"]:
        observed.update(result_received=True, startup_to_result_ms=(now - smoke_start) * 1000)


def _observe_segmentation(ui, launch, observed, detection_ids):
    detector = getattr(launch, "detector", None)
    if detector is not None and detector.ready:
        observed["segmentation_ready"] = True
    detection = getattr(ui, "detection_result", None) or {}
    if detection.get("task") == "segment" and detection.get("success"):
        detection_ids.add(detection.get("display_seq"))
        observed["segmentation_results"] = len(detection_ids)
        observed["last_mask_count"] = detection.get("mask_count", 0)
        observed["mask_result_received"] |= detection.get("mask_count", 0) > 0


def _window_heartbeat(ui, launch, observed, heartbeat_gaps, detection_ids, smoke_start):
    last = time.perf_counter()

    def heartbeat():
        nonlocal last
        now = time.perf_counter()
        heartbeat_gaps.append((now - last) * 1000)
        last = now
        _observe_window_state(ui, launch, observed, now, smoke_start)
        _observe_segmentation(ui, launch, observed, detection_ids)
        ui.after(20, heartbeat)

    return heartbeat


def _window_close(args, ui, observed):
    def close():
        try:
            pipeline = getattr(ui, "_segmentation_pipeline", None)
            if args.segment and pipeline is not None:
                observed["paired_pipeline"] = pipeline.snapshot(time.monotonic())
            session_logs = getattr(ui, "session_logs", None)
            if session_logs is not None:
                # Lets report_localization_performance.py read exactly this run.
                observed["session_dir"] = str(session_logs.directory)
            if args.capture:
                _grab_window(ui, args.capture)
                observed["capture"] = str(args.capture)
        finally:
            ui._on_close()

    return close


def _window_build(args, original_build, observed, heartbeat_gaps, detection_ids, smoke_start):
    def build(inputs):
        if inputs.args.live or inputs.interface_mode.value != "simulated-stream":
            raise RuntimeError("benchmark refuses real-flight interfaces")
        launch = original_build(inputs)
        ui = launch.app
        if args.window_size:
            ui.geometry(args.window_size)
        heartbeat = _window_heartbeat(
            ui, launch, observed, heartbeat_gaps, detection_ids, smoke_start
        )
        ui.after(0, heartbeat)
        ui.after(round(args.ui_seconds * 1000), _window_close(args, ui, observed))
        return launch

    return build


def _window_argv(args):
    argv = [
        str(OPERATOR / "flight_operator_app.py"),
        "--interface",
        "simulated-stream",
        "--site-profile",
        str(args.site_profile),
        "--video",
        str(args.video),
        "--replay-json",
        "",
    ]
    argv += ["--live-localize", "--auto-inspect"] if args.localize else ["--no-live-localize"]
    if args.segment:
        argv += [
            "--live-detect",
            "--detector-task",
            "segment",
            "--detector-half",
            "--detector-model",
            str(args.segment_model),
            "--detector-python",
            str(args.segment_python),
            "--detect-every-n-frames",
            "1",
        ]
    else:
        argv += ["--no-live-detect"]
    return argv


def _record_window_stats(report, observed, operator_tick, stage_samples, heartbeat_gaps):
    report["render_stages_ms"] = {
        name: {
            "count": len(values),
            "p50": operator_tick._percentile(values),
            "p95": operator_tick._percentile(values, 0.95),
        }
        for name, values in sorted(stage_samples.items())
        if values
    }
    if heartbeat_gaps:
        observed["heartbeat_gap_p95_ms"] = sorted(heartbeat_gaps)[
            min(len(heartbeat_gaps) - 1, int(len(heartbeat_gaps) * 0.95))
        ]
        observed["heartbeat_gap_max_ms"] = max(heartbeat_gaps)


def _check_window_result(args, observed):
    if not observed["mapped"]:
        raise RuntimeError("operator window never became viewable")
    if args.segment:
        paired = observed.get("paired_pipeline", {})
        if not (
            observed["segmentation_ready"]
            and observed["mask_result_received"]
            and paired.get("paired_delivered_count", 0) > 0
            and paired.get("rendered_mask_count", 0) > 0
        ):
            raise RuntimeError(f"segmentation masks/pairs never rendered: {observed}")


def _window_smoke(args, report: dict) -> None:
    smoke_start = time.perf_counter()
    import operator_launch

    report["operator_import_ms"] = (time.perf_counter() - smoke_start) * 1000
    import operator_tick

    stage_samples, original_detail = _install_stage_profiler(operator_tick)
    timings = report["startup_stages_ms"] = {}
    _time_startup_stages(operator_launch, timings)
    observed = report["window"] = {"mapped": False, "worker_ready": False, "result_received": False}
    heartbeat_gaps: list[float] = []
    detection_ids = set()
    observed.update(segmentation_ready=False, mask_result_received=False)
    operator_launch._build_operator_launch = _window_build(
        args,
        operator_launch._build_operator_launch,
        observed,
        heartbeat_gaps,
        detection_ids,
        smoke_start,
    )
    argv = _window_argv(args)
    previous_argv = sys.argv
    sys.argv = argv
    try:
        operator_launch.main()
    finally:
        sys.argv = previous_argv
        operator_tick._TickProfile.detail = original_detail
        _record_window_stats(report, observed, operator_tick, stage_samples, heartbeat_gaps)
    _check_window_result(args, observed)


class _CompletedWorker:
    ready = True
    alive = True

    def __init__(self, completions):
        self._completions = completions

    def poll_results(self):
        return _drain_queue(self._completions)

    def drain_result_notifications(self):
        return None

    def busy(self):
        return False


def _build_controlled_ui(operator, np, args):
    from simulated_backend import DroneBackend

    ui = operator.OperatorApp(
        DroneBackend(),
        np.random.default_rng(0).random((2000, 6)).astype(np.float32) * 10,
        site_id="offline-completions",
    )
    ui.geometry(args.window_size or "1440x900")
    ui.update_idletasks()
    ui.inspecting = True
    ui.boot_lock_done = True
    # No decoder, localizer process, socket, or flight command is involved.
    # Completion data still passes the production validation/publication path.
    ui.submit_current_frame_for_localization = lambda: None
    frame = np.full((720, 1280, 3), (25, 70, 150), dtype=np.uint8)
    ui.video_frame = frame
    ui._video_frame_stamp = time.monotonic()
    return ui


def _instrument_controlled_ui(args, ui, profiler, delays, received, failures, render_times):
    original_metrics = ui.update_localization_metrics

    def metrics(result):
        received.append(int(result["seq"]))
        delays.append(float(result["ui_poll_delay_ms"]))
        original_metrics(result)

    ui.update_localization_metrics = metrics
    ui.report_callback_exception = lambda kind, value, trace: failures.append(repr(value))
    original_render = ui.render_video

    def render(*values):
        start = time.perf_counter()
        if args.render_profile:
            profiler.enable()
        try:
            return original_render(*values)
        finally:
            if args.render_profile:
                profiler.disable()
            render_times.append((time.perf_counter() - start) * 1000)

    ui.render_video = render


def _produce_completions(completions, stop):
    seq = 0
    while not stop.wait(0.05):
        seq += 1
        now = time.monotonic()
        completions.put(
            {
                "seq": seq,
                "display_seq": seq,
                "frame_name": f"controlled-{seq}",
                "success": True,
                "pose": {"x": seq * 0.001, "y": 0.0, "z": 0.0},
                "inliers": 80,
                "reproj_rms": 0.5,
                "mode": "TRACK",
                "wall_ms": 18.61 + seq % 5,
                "core_wall_ms": 18.61,
                "client_submit_mono": now - 0.02,
                "client_response_mono": now,
                "capture_mono": now - 0.02,
            }
        )


def _controlled_heartbeat(ui, heartbeat_gaps):
    last = time.perf_counter()

    def heartbeat():
        nonlocal last
        now = time.perf_counter()
        heartbeat_gaps.append((now - last) * 1000)
        last = now
        # Change source and freshness just as a camera callback would, on Tk.
        ui._video_frame_stamp = time.monotonic()
        ui.backend.state.frame_age_ms = (len(heartbeat_gaps) % 40) * 10.0
        ui.after(20, heartbeat)

    return heartbeat


def _controlled_close(args, ui, stop, report, received, failures):
    def close():
        stop.set()
        if args.capture:
            _grab_window(ui, args.capture)
        report["window"] = {
            "mode": "offline-controlled-completions",
            "mapped": bool(ui.winfo_viewable()),
            "result_received": bool(received),
            "received_count": len(received),
            "ordered_once": received == sorted(set(received)),
            "latest_seq": received[-1] if received else None,
            "history_count": len(ui.history),
            "callback_failures": failures,
            "capture": str(args.capture) if args.capture else None,
        }
        for callback in ui.tk.call("after", "info"):
            ui.after_cancel(callback)
        ui.destroy()

    return close


def _record_controlled_stats(report, operator_tick, delays, heartbeat_gaps, render_times):
    for name, values in (
        ("ui_poll_delay_ms", delays),
        ("heartbeat_gap_ms", heartbeat_gaps),
        ("render_video_ms", render_times),
    ):
        report[name] = (
            {
                "count": len(values),
                "p50": operator_tick._percentile(values),
                "p95": operator_tick._percentile(values, 0.95),
                "max": max(values),
            }
            if values
            else {"count": 0}
        )


def _controlled_window_smoke(args, report: dict) -> None:
    """Exercise real Tk ingestion/rendering with an offline completion producer."""
    import cProfile
    import numpy as np
    import flight_operator_app as operator
    import operator_tick

    if not args.controlled_results or args.localize:
        raise RuntimeError("controlled completions refuse model/live launch paths")

    completions = queue.Queue()
    stop = threading.Event()
    delays, heartbeat_gaps, render_times = [], [], []
    received = []
    failures = []
    profiler = cProfile.Profile()
    ui = _build_controlled_ui(operator, np, args)
    ui.localizer = _CompletedWorker(completions)
    _instrument_controlled_ui(args, ui, profiler, delays, received, failures, render_times)
    producer = threading.Thread(
        target=_produce_completions,
        args=(completions, stop),
        name="offline-completions",
        daemon=True,
    )
    producer.start()
    heartbeat = _controlled_heartbeat(ui, heartbeat_gaps)
    ui.after(0, heartbeat)
    ui.after(
        round(args.ui_seconds * 1000), _controlled_close(args, ui, stop, report, received, failures)
    )
    try:
        ui.mainloop()
    finally:
        stop.set()
        producer.join()
    _record_controlled_stats(report, operator_tick, delays, heartbeat_gaps, render_times)
    if args.render_profile:
        profiler.dump_stats(str(args.render_profile))
    if failures or not received or received != sorted(set(received)):
        raise RuntimeError(f"offline Tk completion smoke failed: {report['window']}")


def _validate_args(parser, args) -> None:
    if args.iterations < 1 or not 0 <= args.ui_seconds <= 600:
        parser.error("iterations must be positive and ui-seconds must be 0..600")
    if (
        args.ui_seconds
        and not args.controlled_results
        and (args.site_profile is None or args.video is None)
    ):
        parser.error("window smoke requires --site-profile and --video")
    if args.localize and not args.ui_seconds:
        parser.error("--localize requires a window smoke")
    if args.segment and (not args.localize or args.controlled_results):
        parser.error("--segment requires --localize and an actual offline window smoke")
    if (args.capture or args.window_size) and not args.ui_seconds:
        parser.error("--capture and --window-size require a window smoke")
    if args.controlled_results and (not args.ui_seconds or args.localize):
        parser.error("--controlled-results requires --ui-seconds and cannot use --localize")
    if args.render_profile and not args.controlled_results:
        parser.error("--render-profile requires --controlled-results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument(
        "--ui-seconds",
        type=float,
        default=0,
        help="keep the simulated window observable this long after construction",
    )
    parser.add_argument("--site-profile", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument(
        "--window-size", help="fixed Tk geometry, e.g. 1440x900, for before/after comparisons"
    )
    parser.add_argument(
        "--capture", type=Path, help="save a screenshot before closing the simulated window"
    )
    parser.add_argument(
        "--localize",
        action="store_true",
        help="also load the real localizer models and feed recorded frames, never flight commands",
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="run real pretrained YOLO26n-seg alongside localization and display masks",
    )
    parser.add_argument(
        "--segment-model", type=Path, default=ROOT / "執行環境/models/yolo26n-seg.pt"
    )
    parser.add_argument("--segment-python", type=Path, default=ROOT / ".venv-yolo/bin/python")
    parser.add_argument(
        "--controlled-results",
        action="store_true",
        help="real Tk with offline 20 Hz completion events, no models/video/hardware",
    )
    parser.add_argument(
        "--render-profile", type=Path, help="save controlled render cProfile statistics"
    )
    parser.add_argument("--report", type=Path, help="save the observed benchmark JSON")
    args = parser.parse_args()
    _validate_args(parser, args)
    sys.path.insert(0, str(ROOT / "控制介面程式"))
    sys.path.insert(0, str(OPERATOR))
    from live_worker_clients import LiveWorkerClient

    report = {"pipe_submit": _pipe_benchmark(LiveWorkerClient, args.iterations)}
    if args.controlled_results:
        _controlled_window_smoke(args, report)
    elif args.ui_seconds:
        _window_smoke(args, report)
    if args.report:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
