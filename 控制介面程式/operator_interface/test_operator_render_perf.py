"""Guards for the operator-UI render and layout optimisations.

Nothing else in the suite constructs a real OperatorApp, so these are the only
tests that would catch a layout regression or a reintroduced per-tick
PhotoImage/label churn. They need a usable X display and are skipped without one.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import gc
import inspect
import math

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
        # Collect while the interpreter is still on the Tk main thread. Deferred
        # Variable.__del__ calls otherwise surface as an unraisable exception in
        # whichever unrelated test happens to trigger the collection.
        instance.destroy()
        gc.collect()


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


def test_controls_use_tabs_without_scrollbars_and_flight_actions_stay_visible(
    operator,
) -> None:
    """All panels use fixed tabs; abort actions remain outside those tabs."""
    wanted = {"懸停", "原地降落", "緊急停止電腦動作", "手動/搖桿 (Esc)"}
    found: dict[str, list] = {label: [] for label in wanted}
    descendants = []

    def walk(widget) -> None:
        for child in widget.winfo_children():
            descendants.append(child)
            try:
                label = child.cget("text")
            except Exception:
                label = None
            if label in found:
                found[label].append(child)
            walk(child)

    def inside_control_tabs(widget) -> bool:
        node = widget
        while node is not None and node is not operator:
            if node is operator.controls_notebook:
                return True
            node = node.master
        return False

    walk(operator)
    assert not any(
        isinstance(widget, (tk.Scrollbar, app.ttk.Scrollbar))
        for widget in descendants
    )
    assert tuple(
        operator.controls_notebook.tab(tab_id, "text")
        for tab_id in operator.controls_notebook.tabs()
    ) == (
        # One flight tab: 定位資訊 merged into it, then 飛行 · 操作 emptied out when
        # the virtual sticks moved beside the flight readouts. Everything else is
        # pre-flight setup and is only reached on the ground.
        "飛行",
        "校正",
        "場域資產",
        "系統紀錄",
    )

    # 懸停 deliberately exists twice: the flight row and the nudge pad centre.
    # At least one instance of every abort action must stay above the tabs.
    for label, widgets in found.items():
        assert widgets, f"missing flight control: {label}"
        assert any(not inside_control_tabs(widget) for widget in widgets), (
            f"every '{label}' button sits inside a selectable tab"
        )


def test_each_control_tab_fits_the_minimum_window_without_clipping(operator) -> None:
    operator.geometry("980x640")
    operator.update_idletasks()
    root_left = operator.winfo_rootx()
    root_top = operator.winfo_rooty()
    root_right = root_left + operator.winfo_width()
    root_bottom = root_top + operator.winfo_height()

    for tab_id in operator.controls_notebook.tabs():
        operator.controls_notebook.select(tab_id)
        operator.update_idletasks()
        stack = [operator.nametowidget(tab_id)]
        while stack:
            parent = stack.pop()
            children = parent.winfo_children()
            stack.extend(children)
            for child in children:
                if not child.winfo_ismapped():
                    continue
                left = child.winfo_rootx()
                top = child.winfo_rooty()
                right = left + child.winfo_width()
                bottom = top + child.winfo_height()
                assert root_left <= left <= right <= root_right
                assert root_top <= top <= bottom <= root_bottom


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


def test_magnetometer_axis_diagram_draws_and_clears(operator) -> None:
    """The diagram must draw for each axis and CLEAR when no axis is requested,
    so the operator never sees a stale rotation they should not perform."""
    canvas = operator.magnetometer_axis_canvas

    for raw in ("xAxis", "yAxis", "zAxis"):
        operator._draw_magnetometer_axis(app.magnetometer_axis_guide(raw))
        operator.update_idletasks()
        items = canvas.find_all()
        assert len(items) >= 5, f"{raw} drew too little: {len(items)} items"
        texts = [canvas.itemcget(i, "text") for i in items
                 if canvas.type(i) == "text"]
        assert any("現在請轉" in t for t in texts), raw

    operator._draw_magnetometer_axis(None)
    operator.update_idletasks()
    texts = [canvas.itemcget(i, "text") for i in canvas.find_all()
             if canvas.type(i) == "text"]
    assert not any("現在請轉" in t for t in texts), "stale rotation still shown"


def test_virtual_stick_drag_clamps_to_the_circle_and_self_centres(operator) -> None:
    """The knob is a circle: a corner drag must not exceed unit magnitude."""
    import types

    stick = operator.stick_right
    seen: list[tuple[float, float]] = []
    stick._on_change = lambda _s, x, y: seen.append((x, y))
    centre = stick.SIZE / 2.0

    stick._on_drag(types.SimpleNamespace(x=centre + stick._radius, y=centre))
    assert seen[-1] == pytest.approx((1.0, 0.0))

    # Far outside, diagonally: clamped onto the circle, direction preserved.
    stick._on_drag(types.SimpleNamespace(x=stick.SIZE * 3, y=-stick.SIZE * 3))
    x, y = seen[-1]
    assert math.hypot(x, y) == pytest.approx(1.0)
    assert x > 0 and y > 0

    # Releasing, or the pointer leaving the widget, always reports zero.
    stick._on_release()
    assert seen[-1] == (0.0, 0.0)
    assert stick.value == (0.0, 0.0)


def test_both_virtual_sticks_are_present_and_start_centred(operator) -> None:
    for stick in (operator.stick_left, operator.stick_right):
        assert stick.value == (0.0, 0.0)
        assert stick.winfo_reqwidth() == stick.SIZE


def test_no_control_tab_is_clipped_by_the_fixed_control_pane(operator) -> None:
    """The pane has pack_propagate(False), so a tall tab is cut off, not scrolled.

    This was a hard-coded 295 px while 飛控與限制 needed 304 px: the bottom of that
    tab was simply unreachable, with nothing on screen to say so.
    """
    notebook = operator.controls_notebook
    pane = notebook.master
    for geometry in ("1440x900", "980x640"):
        operator.geometry(geometry)
        operator.update_idletasks()
        pane_bottom = pane.winfo_rooty() + pane.winfo_height()
        for tab_id in notebook.tabs():
            notebook.select(tab_id)
            operator.update_idletasks()
            stack = [operator.nametowidget(tab_id)]
            while stack:
                parent = stack.pop()
                children = parent.winfo_children()
                stack.extend(children)
                for child in children:
                    if not child.winfo_ismapped():
                        continue
                    bottom = child.winfo_rooty() + child.winfo_height()
                    assert bottom <= pane_bottom, (
                        f"{geometry} {notebook.tab(tab_id, 'text')}: a widget ends "
                        f"{bottom - pane_bottom}px below the control pane"
                    )


def test_each_gravity_guide_phase_fits_the_fixed_control_pane(operator) -> None:
    notebook = operator.controls_notebook
    pane = notebook.master
    calibration_tab = operator._preflight_tabs["compass"]
    notebook.select(calibration_tab)

    for geometry in ("1440x900", "980x640"):
        operator.geometry(geometry)
        for phase in app.PHASES:
            operator.gravity_guide_var.set(
                app.gravity_phase_guidance(phase, sample_count=15, span_deg=90.0)
            )
            operator.update_idletasks()
            pane_bottom = pane.winfo_rooty() + pane.winfo_height()
            stack = [calibration_tab]
            while stack:
                parent = stack.pop()
                children = parent.winfo_children()
                stack.extend(children)
                for child in children:
                    if child.winfo_ismapped():
                        bottom = child.winfo_rooty() + child.winfo_height()
                        assert bottom <= pane_bottom, (
                            f"{geometry} {phase}: gravity guide clips by "
                            f"{bottom - pane_bottom}px"
                        )


def test_shrinking_the_window_takes_height_from_the_map_not_the_controls(operator) -> None:
    """The control pane is fixed-height; the map/video pane is the elastic one."""
    notebook = operator.controls_notebook
    pane = notebook.master

    operator.geometry("1440x900")
    operator.update_idletasks()
    tall_pane, tall_map = pane.winfo_height(), operator.map_label.winfo_height()

    # 520, not 640: at 640 everything still fits, so pack order does not matter
    # and the test would pass with the protection removed.
    operator.geometry("980x520")
    operator.update_idletasks()

    assert pane.winfo_height() == tall_pane, "the control pane lost height"
    assert operator.map_label.winfo_height() < tall_map, (
        "sanity: the map pane should be the one that shrank"
    )


def test_map_view_controls_live_on_the_map(operator) -> None:
    """They change what the map shows, so they belong on the map, bottom-left."""
    labels = {}
    stack = [operator.map_label]
    while stack:
        parent = stack.pop()
        children = parent.winfo_children()
        stack.extend(children)
        for child in children:
            try:
                labels[str(child.cget("text"))] = child
            except Exception:
                continue

    for text in ("重設地圖", "上下翻面", "顯示規劃路徑"):
        assert text in labels, f"{text} is no longer a child of the map canvas"

    operator.update_idletasks()
    map_bottom = operator.map_label.winfo_rooty() + operator.map_label.winfo_height()
    map_left = operator.map_label.winfo_rootx()
    widget = labels["重設地圖"]
    assert widget.winfo_rootx() - map_left < operator.map_label.winfo_width() * 0.5
    assert map_bottom - (widget.winfo_rooty() + widget.winfo_height()) < 60


def test_camera_and_mission_sit_in_the_always_visible_flight_bar(operator) -> None:
    """Buried in a tab, they were unreachable unless that tab happened to be open."""
    # The status line lives on the video now, so the bar is named rather than
    # located through whichever widget happened to be parented to it.
    flight_bar = operator.flight_bar
    titles = set()
    for child in flight_bar.winfo_children():
        try:
            titles.add(str(child.cget("text")))
        except Exception:
            continue

    # 任務控制 merged INTO 飛行模式 on 2026-08-06: one block of flight actions,
    # not two adjacent frames that read as unrelated groups.
    assert {"飛行模式", "鏡頭"} <= titles, f"flight bar holds {sorted(titles)}"
    assert "任務控制" not in titles, "the mission frame is back as a separate block"

    labels = set()
    stack = [flight_bar]
    while stack:
        parent = stack.pop()
        for child in parent.winfo_children():
            stack.append(child)
            try:
                labels.add(str(child.cget("text")))
            except Exception:
                continue
    assert "起飛" in labels, "takeoff must still live in the always-visible bar"
    assert "起飛後錄影" in labels


def test_control_pane_follows_the_selected_tab_height(operator) -> None:
    """Sizing to the tallest tab padded every short tab with dead space."""
    notebook = operator.controls_notebook
    pane = notebook.master
    # The <<NotebookTabChanged>> binding is a virtual event, which update_idletasks
    # does not pump; call the fitter directly and assert the binding separately.
    assert "<<NotebookTabChanged>>" in notebook.bind(), (
        "nothing re-fits the pane when the operator switches tabs"
    )
    heights = {}
    for tab_id in notebook.tabs():
        notebook.select(tab_id)
        operator._fit_control_pane()
        operator.update_idletasks()
        heights[notebook.tab(tab_id, "text")] = pane.winfo_height()

    assert len(set(heights.values())) > 1, (
        f"every tab still gets the same height: {heights}"
    )
    # The shortest tab must not be forced to the tallest tab's height.
    assert heights["系統紀錄"] < max(heights.values())
    assert min(heights.values()) >= app.CONTROL_PANE_MIN_H


def test_localization_readout_overlays_the_video_it_measures(operator) -> None:
    """Reading it used to mean looking away from the picture it describes."""
    operator.update_idletasks()
    found = []
    stack = [operator.video_label]
    while stack:
        parent = stack.pop()
        for child in parent.winfo_children():
            stack.append(child)
            if child is operator.loc_health_label:
                found.append(child)
    assert found, "定位儀表 is no longer a child of the video canvas"

    video_bottom = operator.video_label.winfo_rooty() + operator.video_label.winfo_height()
    video_left = operator.video_label.winfo_rootx()
    overlay = found[0].master
    assert overlay.winfo_rootx() - video_left < operator.video_label.winfo_width() * 0.5
    assert video_bottom - (overlay.winfo_rooty() + overlay.winfo_height()) < 200


def test_virtual_sticks_share_the_flight_tab_with_the_readouts(operator) -> None:
    """飛行 · 操作 held only the sticks once everything else moved out."""
    notebook = operator.controls_notebook
    target = [t for t in notebook.tabs() if notebook.tab(t, "text") == "飛行"]
    assert target, "the flight tab is missing"
    notebook.select(target[0])
    operator.update_idletasks()

    node = operator.stick_left
    while node is not None and node is not operator:
        if str(node) == str(target[0]):
            break
        node = node.master
    else:
        raise AssertionError("the virtual sticks are not inside the flight tab")

    titles = set()
    stack = [operator.nametowidget(target[0])]
    while stack:
        parent = stack.pop()
        for child in parent.winfo_children():
            stack.append(child)
            try:
                titles.add(str(child.cget("text")))
            except Exception:
                continue
    assert "ANAFI / 控制權" in titles


def test_no_control_readout_or_button_appears_twice(operator) -> None:
    """Three copies of battery/gimbal/link, two of the handover buttons, is worse
    than one: they refresh by different paths and disagree under load."""
    notebook = operator.controls_notebook
    duplicated = {"恢復電腦控制", "交回搖桿 (Esc)", "定位鎖定"}
    seen = {label: 0 for label in duplicated}

    for tab_id in list(notebook.tabs()):
        notebook.select(tab_id)
        operator.update_idletasks()
    stack = [operator]
    while stack:
        parent = stack.pop()
        children = parent.winfo_children()
        stack.extend(children)
        for child in children:
            try:
                label = str(child.cget("text"))
            except Exception:
                continue
            if label in seen:
                seen[label] += 1

    assert seen["定位鎖定"] == 0, "定位鎖定 engages automatically after 開始定位"
    assert seen["恢復電腦控制"] <= 1, f"恢復電腦控制 rendered {seen['恢復電腦控制']}x"
    assert seen["交回搖桿 (Esc)"] <= 1, f"交回搖桿 rendered {seen['交回搖桿 (Esc)']}x"


def test_anafi_panel_does_not_restate_the_hud_and_status_bar(operator) -> None:
    """It was a third copy of battery / gimbal / zoom / backlog age / fps."""
    operator.update_idletasks()
    flight = operator.anafi_flight_var.get()
    stream = operator.anafi_stream_var.get()

    for token in ("battery", "gimbal", "zoom"):
        assert token not in flight, f"{token} is already in the status bar / HUD"
    for token in ("backlog age", "fps", "GPS", "link "):
        assert token not in stream, f"{token} is already in the video HUD"


def test_status_bar_states_the_mode_once(operator) -> None:
    """It read "REAL ANAFI | mode=LIVE | LIVE | LIVE | HOVER" on screen.

    video_hud_identity already ends in "mode=<X>", and display_mode is the very
    same operator_mode_label() call, so appending it printed the value twice.
    """
    identity = app.video_hud_identity(True, "LIVE")
    assert identity.endswith("mode=" + app.operator_mode_label(True, "LIVE"))

    source = inspect.getsource(app.OperatorApp.tick)
    assert "video_hud_identity" in source
    assert "{display_mode} | {st.loc}" not in source, (
        "display_mode restates the mode already inside video_hud_identity"
    )


def test_localization_overlay_stays_a_small_corner_of_the_video(operator) -> None:
    """It overlays the picture, so every line it takes is picture lost."""
    operator.geometry("1440x900")
    operator.update_idletasks()
    video = operator.video_label
    overlay = operator.loc_health_label.master

    assert overlay.winfo_rooty() >= video.winfo_rooty(), "overlay spills above the video"
    assert overlay.winfo_height() < video.winfo_height() * 0.45, (
        f"overlay is {overlay.winfo_height()}px of {video.winfo_height()}px"
    )
    # Two-up, not a single stacked column: the column count is what keeps it short.
    # It now carries the flight readouts too, so the row budget grew with it.
    columns, rows = overlay.grid_size()
    assert columns >= 2, f"the readouts are stacked in {columns} column(s)"
    assert rows <= 9, f"{rows} rows of overlay is most of the picture"


def test_localization_facts_appear_only_in_the_video_overlay(operator) -> None:
    """定位 / inliers / 位姿 / 影格 were in the bar AND the overlay at once."""
    operator.update_idletasks()
    bar_text = operator.status.cget("text")
    assert operator.status.master is operator.loc_health_label.master, (
        "the aircraft-state line must share the overlay, not sit in its own block"
    )
    for token in ("定位", "inliers", "位姿", "影格"):
        assert token not in bar_text, f"{token} is already in the overlay: {bar_text}"

    # The separate 位姿/影格 readout is gone; its colour grading moved to the
    # overlay's health label, which is the thing it was grading.
    assert not hasattr(operator, "loc_age_label")
    source = inspect.getsource(app.OperatorApp._update_age_readout)
    assert "loc_health_label" in source
