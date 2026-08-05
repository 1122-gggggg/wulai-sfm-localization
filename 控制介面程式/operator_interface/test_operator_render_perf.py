"""Guards for the operator-UI render and layout optimisations.

Nothing else in the suite constructs a real OperatorApp, so these are the only
tests that would catch a layout regression or a reintroduced per-tick
PhotoImage/label churn. They need a usable X display and are skipped without one.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
for extra in (HERE, HERE.parent):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import flight_operator_app as app  # noqa: E402

tk = pytest.importorskip("tkinter")


def _display_available() -> bool:
    try:
        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


pytestmark = pytest.mark.skipif(
    not _display_available(), reason="operator UI tests need a usable display"
)


@pytest.fixture()
def operator():
    backend = app.DroneBackend()
    points = np.random.default_rng(0).random((2000, 6)).astype(np.float32) * 10.0
    instance = app.OperatorApp(backend, points, tick_ms=10, site_id="test")
    instance.update_idletasks()
    try:
        yield instance
    finally:
        instance.destroy()


def test_dedup_string_var_suppresses_only_no_op_writes(operator) -> None:
    var = app.DedupStringVar(master=operator, value="a")
    writes: list[str] = []
    var.trace_add("write", lambda *_: writes.append(var.get()))

    var.set("a")
    assert writes == [], "writing the current value must not repaint"
    var.set("b")
    var.set("b")
    var.set("c")
    assert writes == ["b", "c"]


def test_dedup_string_var_still_overwrites_operator_typed_text(operator) -> None:
    """The compare reads the live Tcl value, so an Entry edit is not shadowed."""
    var = app.DedupStringVar(master=operator, value="10")
    entry = tk.Entry(operator, textvariable=var)
    entry.delete(0, "end")
    entry.insert(0, "999")               # widget writes the variable directly
    assert var.get() == "999"
    var.set("10")                        # code restores the original value
    assert var.get() == "10"


def test_present_frame_reuses_the_photo_until_the_panel_resizes(operator) -> None:
    state = operator.backend.poll()
    operator._present_frame(
        operator.map_label, "map_photo", operator.render_map(400, 300, state))
    first = operator.map_photo
    operator._present_frame(
        operator.map_label, "map_photo", operator.render_map(400, 300, state))
    assert operator.map_photo is first, "same-size repaint must reuse the PhotoImage"
    operator._present_frame(
        operator.map_label, "map_photo", operator.render_map(420, 300, state))
    assert operator.map_photo is not first, "a resize must allocate a new PhotoImage"


def test_map_detail_drops_while_interacting_and_settles_after(operator) -> None:
    assert operator._map_detail_points() == app.MAP_STATIC_POINTS
    operator._note_map_interaction()
    assert operator._map_detail_points() == app.MAP_INTERACTIVE_POINTS
    assert app.MAP_INTERACTIVE_POINTS < app.MAP_STATIC_POINTS
    # The decimation level is part of the base-image cache key, so settling back
    # to full detail cannot serve the decimated cloud from cache.
    interactive_key = operator.map_base_key(400, 300)
    operator._map_interact_until = 0.0
    assert operator.map_base_key(400, 300) != interactive_key
    operator.on_map_release(None)
    assert operator._map_dirty_key is None


def test_incident_banner_only_occupies_a_row_when_something_is_wrong(operator) -> None:
    """2026-08-03 operator decision: no idle 「安全狀態：正常」 row.

    The localization alert banner was removed entirely in the same pass;
    LOCALIZATION LOST / LOW CONFIDENCE now appear only inside the video panel
    (render_video), which is clipped to that panel and gated by its dirty key.
    """
    assert not hasattr(operator, "loc_alert_banner")
    assert not hasattr(operator, "_update_alert_banner")

    operator._show_incident_banner(None, None)
    assert operator.incident_banner.winfo_manager() == ""

    operator._show_incident_banner("▲ 安全事件：CONTROL LINK LOST", "#b42318")
    assert operator.incident_banner.winfo_manager() == "pack"
    assert "CONTROL LINK LOST" in operator.incident_banner.cget("text")

    operator._show_incident_banner(None, None)
    assert operator.incident_banner.winfo_manager() == ""


def test_flight_actions_are_not_inside_the_scrollable_pane(operator) -> None:
    """原地降落 / 緊急停止 must not be scrollable off screen."""
    wanted = {"懸停", "原地降落", "緊急停止電腦動作", "手動/搖桿 (Esc)"}
    found: dict[str, list] = {label: [] for label in wanted}

    def walk(widget) -> None:
        for child in widget.winfo_children():
            try:
                label = child.cget("text")
            except Exception:
                label = None
            if label in found:
                found[label].append(child)
            walk(child)

    def scrollable(widget) -> bool:
        node = widget
        while node is not None and node is not operator:
            if isinstance(node, tk.Canvas):
                return True
            node = node.master
        return False

    walk(operator)
    # 懸停 deliberately exists twice: the flight row and the nudge pad centre.
    # The requirement is that every action is reachable without scrolling.
    for label, widgets in found.items():
        assert widgets, f"missing flight control: {label}"
        assert any(not scrollable(widget) for widget in widgets), (
            f"every '{label}' button sits inside a scrollable canvas "
            "and can be scrolled off screen"
        )


def test_long_route_draws_a_bounded_number_of_dots(operator, monkeypatch) -> None:
    ellipses = {"count": 0}
    real_draw = app.ImageDraw.Draw

    class CountingDraw:
        def __init__(self, image):
            self._inner = real_draw(image)

        def ellipse(self, *args, **kwargs):
            ellipses["count"] += 1
            return self._inner.ellipse(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(app.ImageDraw, "Draw", CountingDraw)
    rng = np.random.default_rng(1)
    operator.route_pts = [tuple(p) for p in rng.random((2000, 3)) * 5.0]
    operator.route_visible = True
    operator.history = []
    operator.history_health = []
    operator.no_loc_markers = []
    operator.map_base_cache = None

    operator.render_map(400, 300, operator.backend.poll())
    # Route dots plus a small fixed set (pivot marker, camera dot, frustum centre).
    assert ellipses["count"] <= app.ROUTE_DOT_MAX + 8, ellipses["count"]
    assert ellipses["count"] > 0
