"""End-to-end FPS: frames whose localization AND detection results both arrived."""

import pytest

import flight_operator_app as app
from operator_pair_metrics import EndToEndFpsTracker
from operator_rendering import build_hud_overlay_data


def _result(seq, name="f", **extra):
    return {"display_seq": seq, "frame_name": f"{name}{seq}", **extra}


def test_a_frame_counts_only_when_both_workers_returned_that_same_frame():
    tracker = EndToEndFpsTracker()
    tracker.expect_pair(1, "f1")
    assert tracker.receive("localization", _result(1), 10.0) is False
    # A detection result for a different frame, or one never registered, never completes it.
    assert tracker.receive("detection", _result(2), 10.1) is False
    assert tracker.completed_count == 0
    assert tracker.receive("detection", _result(1), 10.2) is True
    assert tracker.completed_count == 1
    # Replays of an already-completed frame are ignored.
    assert tracker.receive("detection", _result(1), 10.3) is False
    assert tracker.receive("localization", _result(1), 10.3) is False
    assert tracker.completed_count == 1


def test_each_stage_counts_once_and_retries_and_unknown_kinds_are_ignored():
    tracker = EndToEndFpsTracker()
    tracker.expect_pair(5, "f5")
    assert tracker.receive("localization", _result(5), 1.0) is False
    assert tracker.receive("localization", _result(5), 1.1) is False  # duplicate stage
    assert tracker.receive("detection", _result(5, hold_retry=True), 1.2) is False
    assert tracker.receive("segmentation", _result(5), 1.3) is False
    assert tracker.receive("detection", {"frame_name": "f5"}, 1.4) is False  # no display_seq
    assert tracker.receive("detection", None, 1.5) is False
    assert tracker.completed_count == 0
    assert tracker.receive("detection", _result(5), 1.6) is True


def test_fps_is_the_rolling_pair_completion_rate_and_decays_to_zero():
    tracker = EndToEndFpsTracker(window_s=5.0)
    for seq in range(11):  # one completed frame every 0.2 s -> 5 fps
        tracker.expect_pair(seq, f"f{seq}")
        tracker.receive("localization", _result(seq), seq * 0.2)
        tracker.receive("detection", _result(seq), seq * 0.2 + 0.05)
    assert tracker.fps(2.1) == pytest.approx(5.0)
    assert tracker.fps(60.0) == 0.0  # nothing completed inside the window any more
    tracker.reset()
    assert tracker.fps(2.1) == 0.0 and tracker.completed_count == 0


def test_pending_frames_are_bounded():
    tracker = EndToEndFpsTracker(max_pending=4)
    for seq in range(10):
        tracker.expect_pair(seq, f"f{seq}")
    assert len(tracker.pending) == 4
    assert tracker.receive("localization", _result(0), 1.0) is False  # evicted
    assert (9, "f9") in tracker.pending


class _Detector:
    def __init__(self, *, busy=False, accept=True, task="detect"):
        self._busy, self._accept, self.task = busy, accept, task
        self.submitted = []

    def busy(self):
        return self._busy

    def submit(self, index, name, frame):
        self.submitted.append((index, name, frame))
        return self._accept


def _operator(detector, *, inspecting=True, enabled=True, localizer=True, every=3):
    class FakeOperator:
        _segmentation_enabled = app.OperatorApp._segmentation_enabled
        _detection_follows_localization = app.OperatorApp._detection_follows_localization
        _pair_detection_with_localization = app.OperatorApp._pair_detection_with_localization
        _note_e2e_result = app.OperatorApp._note_e2e_result
        _reset_e2e_fps = app.OperatorApp._reset_e2e_fps
        _pipeline_fps = app.OperatorApp._pipeline_fps
        submit_current_frame_for_detection = app.OperatorApp.submit_current_frame_for_detection

    operator = FakeOperator()
    operator.detector = detector
    operator.detection_enabled = enabled
    operator.inspecting = inspecting
    operator.localizer = object() if localizer else None
    operator._segmentation_pipeline = None
    operator._e2e_fps = EndToEndFpsTracker()
    operator.video_frame = object()
    operator.video_display_index = 9
    operator.video_display_frame_name = "f9"
    operator.last_detect_submitted_index = -1
    operator.detect_every_n_frames = every
    operator.detection_pending_frame_name = ""
    operator.det_status = "OFF"
    operator._video_frame_timing = {}
    operator.loc_fps = 12.0
    return operator


def test_localized_frame_is_also_sent_to_the_detector_and_registered_as_a_pair():
    detector = _Detector()
    operator = _operator(detector)
    operator._pair_detection_with_localization("f9", {})
    assert detector.submitted == [(9, "f9", operator.video_frame)]
    assert (9, "f9") in operator._e2e_fps.pending
    # The independent every-N path steps aside while localization runs (no double submit).
    operator.video_display_index = 12
    operator.submit_current_frame_for_detection()
    assert len(detector.submitted) == 1

    operator._note_e2e_result("localization", _result(9, name="f"), 1.0)
    operator._note_e2e_result("detection", _result(9, name="f"), 1.1)
    assert operator._e2e_fps.completed_count == 1


@pytest.mark.parametrize(
    "detector,kwargs,timing",
    [
        (_Detector(busy=True), {}, {}),                      # busy detector is skipped, never awaited
        (_Detector(accept=False), {}, {}),                   # rejected submit registers nothing
        (_Detector(), {"enabled": False}, {}),               # detection off
        (_Detector(), {}, {"hold_retry": True}),             # lost-hold retry of an old frame
        (_Detector(), {"inspecting": False}, {}),            # localization not running
        (_Detector(task="segment"), {}, {}),                 # segmentation keeps its own pairing
        (None, {}, {}),                                      # no detector loaded
    ],
)
def test_pair_is_not_registered_unless_the_detector_accepted_the_same_frame(
    detector, kwargs, timing,
):
    operator = _operator(detector, **kwargs)
    operator._pair_detection_with_localization("f9", timing)
    assert not operator._e2e_fps.pending


def test_detection_without_localization_keeps_its_own_every_n_cadence():
    detector = _Detector()
    operator = _operator(detector, inspecting=False)
    operator.video_display_index = 10  # not a multiple of 3
    operator.submit_current_frame_for_detection()
    assert detector.submitted == []
    operator.video_display_index = 9
    operator.submit_current_frame_for_detection()
    assert len(detector.submitted) == 1
    assert not operator._e2e_fps.pending  # nothing to pair with


def test_readout_is_end_to_end_only_while_detection_runs_with_the_detector_loaded():
    operator = _operator(_Detector())
    operator._e2e_fps.completed.extend([100.0, 100.5, 101.0])
    label, value = operator._pipeline_fps()
    assert "端到端" in label and value == 0.0  # window long past: honest zero, not loc_fps
    operator.detection_enabled = False
    assert operator._pipeline_fps() == ("定位 FPS", 12.0)
    operator.detection_enabled = True
    operator.detector = _Detector(task="segment")
    assert operator._pipeline_fps() == ("定位 FPS", 12.0)
    operator.detector = None
    assert operator._pipeline_fps() == ("定位 FPS", 12.0)


def test_hud_shows_the_supplied_label_and_defaults_to_localization_fps():
    telemetry = {"olympe_state": "RTH 1/1", "olympe_attitude": "att | 三軸速度 v"}
    plain = build_hud_overlay_data(None, {"fps": 11.0}, telemetry)
    assert any("定位 FPS 11.0" in line for line in plain["diagnostic_lines"])
    labelled = build_hud_overlay_data(
        None, {"fps": 7.5, "fps_label": "端到端 FPS(定位+偵測)"}, telemetry,
    )
    assert any("端到端 FPS(定位+偵測) 7.5" in line for line in labelled["diagnostic_lines"])
    assert not any("定位 FPS 7.5" in line for line in labelled["diagnostic_lines"])
