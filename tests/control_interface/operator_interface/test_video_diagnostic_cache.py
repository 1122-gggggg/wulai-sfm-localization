from __future__ import annotations

import flight_operator_app as app


class _Var:
    """Stand-in with the DedupStringVar contract: ``version`` bumps on a changed set()."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.version = 0
        self.gets = 0

    def get(self) -> str:
        self.gets += 1
        return self.text

    def set(self, text: str) -> None:
        if text != self.text:
            self.text = text
            self.version += 1


def _operator() -> app.OperatorApp:
    operator = app.OperatorApp.__new__(app.OperatorApp)
    operator.current_auto_speed_var = _Var("AUTO PCMD ±30%")
    operator.olympe_altitude_var = _Var("飛控高度 3.0 | AGL 2.5")
    operator.olympe_attitude_var = _Var("飛控融合姿態 yaw 10 | 三軸速度 1.0 0.0 0.0")
    operator.loc_health_inliers = 150
    operator.loc_health_reproj = 1.2
    operator.loc_health = "OK"
    operator.loc_fps = 12.0
    operator.loc_wall_ms = 40.0
    operator.loc_latency_ms = 35.0
    operator.loc_e2e_ms = 60.0
    operator._device_telemetry_lines = ("device line",)
    return operator


def _telemetry_gets(operator: app.OperatorApp) -> int:
    return sum(getattr(operator, name).gets for name in app._VIDEO_TELEMETRY_VARS)


def test_unchanged_inputs_return_the_same_lines_without_reading_the_vars_again() -> None:
    operator = _operator()

    first = operator._video_diagnostic_lines()
    reads = _telemetry_gets(operator)
    second = operator._video_diagnostic_lines()

    assert second == first
    assert reads > 0
    assert _telemetry_gets(operator) == reads


def test_changed_text_or_metric_repaints_with_the_same_lines_as_a_fresh_build() -> None:
    operator = _operator()
    before = operator._video_diagnostic_lines()

    operator.olympe_altitude_var.set("飛控高度 4.0 | AGL 3.5")
    after_text = operator._video_diagnostic_lines()
    assert after_text != before
    assert "飛控高度 4.0 | AGL 3.5" in after_text

    operator.loc_latency_ms = 36.0
    operator.loc_health_inliers = 151
    after_metric = operator._video_diagnostic_lines()
    assert after_metric != after_text

    fresh = _operator()
    fresh.olympe_altitude_var.set("飛控高度 4.0 | AGL 3.5")
    fresh.loc_latency_ms = 36.0
    fresh.loc_health_inliers = 151
    assert after_metric == fresh._video_diagnostic_lines()


def test_vars_without_a_write_count_are_read_every_call() -> None:
    operator = _operator()
    operator.olympe_altitude_var = type(
        "PlainVar", (), {"get": lambda self: operator.plain_text}
    )()
    operator.plain_text = "飛控高度 1.0 | AGL 0.5"
    assert "飛控高度 1.0 | AGL 0.5" in operator._video_diagnostic_lines()

    operator.plain_text = "飛控高度 2.0 | AGL 1.5"
    assert "飛控高度 2.0 | AGL 1.5" in operator._video_diagnostic_lines()
