"""Same-source pairing, retained mask pixels and render-boundary regressions."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[3] / "控制介面程式" / "operator_interface"
for extra in (HERE, HERE.parent):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from operator_pair_metrics import PairedPresentationMetrics
import flight_operator_app as operator


def submit_pair(pipeline, seq=1, stamp=10.0, frame=None):
    name = f"source-{seq}"
    pipeline.submit("localization", seq, name, stamp, stamp + 0.01)
    pipeline.submit("segmentation", seq, name, stamp, stamp + 0.02, frame=frame)
    return {
        "display_seq": seq, "frame_name": name, "source_frame_stamp_mono": stamp,
        "ui_pair_session": pipeline.session, "success": True, "task": "segment",
        "mask_count": 1, "boxes": [{
            "xyxy": [20, 30, 80, 90], "mask_polygon": [[20, 30], [80, 30], [50, 90]],
            "class_name": "target", "class_id": 1, "confidence": 0.9,
        }],
    }


def test_pair_counts_only_after_both_render_and_ignores_repeated_results():
    pipeline = PairedPresentationMetrics()
    result = submit_pair(pipeline, frame=object())
    assert pipeline.receive("localization", result)
    assert pipeline.receive("segmentation", result)
    assert not pipeline.receive("localization", result)
    assert not pipeline.receive("segmentation", result)
    assert pipeline.snapshot(10.1)["paired_delivered_count"] == 0
    pipeline.rendered("segmentation", result, 10.1)
    assert pipeline.snapshot(10.1)["paired_delivered_count"] == 0
    pipeline.rendered("localization", result, 10.2)
    pipeline.rendered("localization", result, 10.9)
    pipeline.rendered("segmentation", result, 10.9)
    metrics = pipeline.snapshot(10.9)
    assert metrics["paired_delivered_count"] == 1
    assert metrics["segmented_render_count"] == 1
    assert metrics["rendered_mask_count"] == 1
    assert metrics["paired_source_to_render_ms"] == pytest.approx(200)
    assert metrics["paired_submit_to_render_ms"] == pytest.approx(190)
    assert metrics["compute_e2e_fps"] == pytest.approx(1000 / 190)


def test_delivery_rate_is_unique_rendered_pairs_not_source_or_model_fps():
    pipeline = PairedPresentationMetrics()
    for seq in range(3):
        stamp = 10 + seq
        result = submit_pair(pipeline, seq=seq, stamp=stamp, frame=object())
        for kind in ("localization", "segmentation"):
            pipeline.receive(kind, result)
            pipeline.rendered(kind, result, stamp + 0.2)
    assert pipeline.snapshot(12.3)["paired_delivery_fps"] == pytest.approx(1)
    assert pipeline.snapshot(20)["paired_delivery_fps"] == 0
    assert pipeline.snapshot(20)["paired_source_to_render_ms"] is None


def test_source_stamp_name_and_session_must_match_and_hold_retries_do_not_pair():
    pipeline = PairedPresentationMetrics()
    result = submit_pair(pipeline, frame=object())
    for changes in ({"source_frame_stamp_mono": 10.5}, {"frame_name": "other"},
                    {"ui_pair_session": pipeline.session - 1}, {"hold_retry": True}):
        assert not pipeline.receive("segmentation", {**result, **changes})
    assert pipeline.display_frame is None
    assert pipeline.receive("segmentation", result)
    assert not pipeline.receive("localization", {**result, "hold_retry": True})
    pipeline.rendered("segmentation", result, 10.2)
    assert pipeline.delivered_count == 0


def test_failures_deduplicate_separately_and_never_inflate_delivery():
    pipeline = PairedPresentationMetrics()
    result = submit_pair(pipeline, frame=object())
    failure = {**result, "success": False}
    assert pipeline.receive("localization", failure)
    assert not pipeline.receive("localization", failure)
    assert pipeline.receive("segmentation", failure)
    pipeline.rendered("localization", failure, 10.2)
    pipeline.rendered("segmentation", failure, 10.2)
    metrics = pipeline.snapshot(10.2)
    assert metrics["localization_failures"] == metrics["segmentation_failures"] == 1
    assert metrics["paired_delivered_count"] == 0
    assert pipeline.display_frame is None


def test_retained_pixels_are_bounded_and_old_completion_cannot_replace_new_frame():
    pipeline = PairedPresentationMetrics(max_records=3, max_frames=2)
    frames = [object() for _ in range(5)]
    results = [submit_pair(pipeline, seq=seq, stamp=10 + seq, frame=frame)
               for seq, frame in enumerate(frames)]
    assert len(pipeline.records) == 3
    assert len(pipeline.frames) == 2
    assert not pipeline.receive("segmentation", results[0])
    assert pipeline.receive("segmentation", results[4])
    assert pipeline.display_frame is frames[4]
    assert pipeline.receive("segmentation", results[3])
    assert pipeline.display_frame is frames[4]
    assert pipeline.display_result is results[4]


def test_restart_rejects_old_result_even_when_source_identity_is_reused():
    pipeline = PairedPresentationMetrics()
    old = submit_pair(pipeline, frame=object())
    pipeline.receive("segmentation", old)
    pipeline.rendered("segmentation", old, 10.2)
    pipeline.reset()
    new_frame = object()
    new = submit_pair(pipeline, frame=new_frame)
    assert not pipeline.receive("segmentation", old)
    assert pipeline.display_frame is None
    assert pipeline.snapshot(10.3)["rendered_mask_count"] == 0
    assert pipeline.receive("segmentation", new)
    assert pipeline.display_frame is new_frame


class RecordingDraw:
    def __init__(self):
        self.polygons = []
        self.rectangles = []

    def begin(self):
        self.polygons.clear()
        self.rectangles.clear()

    def end(self):
        pass

    def font(self, *args, **kwargs):
        return object()

    def polygon(self, points, **kwargs):
        self.polygons.append((points, kwargs))

    def rectangle(self, coordinates, **kwargs):
        self.rectangles.append((coordinates, kwargs))

    def textlength(self, *args, **kwargs):
        return 30

    def text(self, *args, **kwargs):
        pass


def test_video_renders_masks_on_submitted_pixels_not_newer_live_source(monkeypatch):
    pipeline = PairedPresentationMetrics()
    submitted = np.zeros((720, 1280, 3), dtype=np.uint8)
    live = np.ones((720, 1280, 3), dtype=np.uint8)
    result = submit_pair(pipeline, frame=submitted)
    draw = RecordingDraw()
    sources = []
    presenter = SimpleNamespace(draw=draw, photo=None, has_frame=True)
    presenter.present = lambda source, stamp, width, height: (sources.append((source, stamp)) or (0.5, 7, 11))
    ui = SimpleNamespace(
        detector=SimpleNamespace(task="segment"), _segmentation_pipeline=pipeline,
        _video_presenter=presenter, video_frame=live, _video_frame_stamp=11.0,
        video_display_frame_name="newer", video_display_index=99,
        detection_result=result, inspecting=True, loc_health="OK",
        loc_health_inliers=10, loc_health_reproj=1.0, lost_hold=None,
        _is_live_backend=lambda: False, lost_holding=lambda: False,
        _video_diagnostic_lines=lambda: (),
    )
    ui.draw_detections = lambda *args, **kwargs: operator.OperatorApp.draw_detections(ui, *args, **kwargs)
    monkeypatch.setattr(operator, "draw_video_hud", lambda *args, **kwargs: True)
    monkeypatch.setattr(operator, "draw_video_banner", lambda *args, **kwargs: None)
    operator.OperatorApp.render_video(ui, 640, 360, SimpleNamespace(link_ok=True))
    assert sources[-1][0] is live
    assert not draw.polygons  # Unmatched result must never be painted on the live feed.
    pipeline.receive("segmentation", result)
    operator.OperatorApp.render_video(ui, 640, 360, SimpleNamespace(link_ok=True))
    assert sources[-1][0] is submitted
    assert sources[-1][1] == 10.0
    assert draw.polygons[0][0] == ((17.0, 26.0), (47.0, 26.0), (32.0, 56.0))
    assert draw.polygons[0][1]["stipple"] == "gray50"
    assert pipeline.snapshot(11)["rendered_display_seq"] == 1
    assert pipeline.snapshot(11)["rendered_mask_count"] == 1
    pipeline.reset()
    operator.OperatorApp.render_video(ui, 640, 360, SimpleNamespace(link_ok=True))
    assert sources[-1][0] is live
    assert not draw.polygons
