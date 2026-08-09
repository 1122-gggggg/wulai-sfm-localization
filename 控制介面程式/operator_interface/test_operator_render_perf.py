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
from tkinter import ttk

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


def _parse_window_size(value: object) -> tuple[int, int] | None:
    if isinstance(value, str):
        parts = value.lower().split("x")
        if len(parts) != 2:
            return None
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _window_geometry(size: tuple[int, int]) -> str:
    return f"{size[0]}x{size[1]}"


def _relative_luminance(color: str) -> float:
    value = color.strip().lower()
    if value == "white":
        value = "#ffffff"
    if not value.startswith("#") or len(value) != 7:
        raise ValueError(f"expected an RGB hex colour, got {color!r}")
    channels = [int(value[index:index + 2], 16) / 255.0
                for index in (1, 3, 5)]
    linear = [channel / 12.92 if channel <= 0.04045 else
              ((channel + 0.055) / 1.055) ** 2.4
              for channel in channels]
    return (0.2126 * linear[0] + 0.7152 * linear[1] +
            0.0722 * linear[2])


def _contrast_ratio(foreground: str, background: str) -> float:
    lighter = max(_relative_luminance(foreground), _relative_luminance(background))
    darker = min(_relative_luminance(foreground), _relative_luminance(background))
    return (lighter + 0.05) / (darker + 0.05)


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


def test_window_sizes_come_from_production_constants_and_selftest_has_no_old_minimum(
    operator,
) -> None:
    sizes = {
        name: _parse_window_size(value)
        for name, value in vars(app).items()
        if name.isupper()
    }
    standard_names = [name for name, size in sizes.items() if size == (1440, 900)]
    minimum_names = [name for name, size in sizes.items() if size == (1180, 768)]
    assert standard_names, "production must expose the 1440x900 window constant"
    assert minimum_names, "production must expose the 1180x768 minimum constant"

    assert operator.geometry().split("+", 1)[0] == _window_geometry(app.UI_STANDARD_SIZE)
    assert tuple(map(int, operator.minsize())) == app.UI_MIN_SIZE
    init_source = inspect.getsource(app.OperatorApp.__init__)
    main_source = inspect.getsource(app.main)
    assert any(name in init_source for name in standard_names)
    assert any(name in init_source for name in minimum_names)
    assert any(name in main_source for name in standard_names)
    assert any(name in main_source for name in minimum_names)
    assert f"{980}x{640}" not in main_source


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
    wanted = {"懸停", "原地降落", "手動/搖桿 (Esc)"}
    found: dict[str, list] = {label: [] for label in wanted}
    descendants = []
    all_labels = []

    def walk(widget) -> None:
        for child in widget.winfo_children():
            descendants.append(child)
            try:
                label = child.cget("text")
            except Exception:
                label = None
            if label is not None:
                all_labels.append(str(label))
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
    )
    assert "停止自動並懸停" not in all_labels

    # 懸停 deliberately exists twice: the flight row and the nudge pad centre.
    # At least one instance of every abort action must stay above the tabs.
    for label, widgets in found.items():
        assert widgets, f"missing flight control: {label}"
        assert any(not inside_control_tabs(widget) for widget in widgets), (
            f"every '{label}' button sits inside a selectable tab"
        )


def test_flight_actions_are_keyboard_focusable_and_return_invokes_auto(
    operator,
) -> None:
    for command, button in operator.flight_buttons.items():
        assert str(button.cget("takefocus")).lower() not in {"0", "false"}, command

    auto_button = operator.flight_buttons["start_auto"]
    invoked: list[str] = []
    auto_button.configure(command=lambda: invoked.append("start_auto"))
    auto_button.configure(state="normal")
    auto_button.focus_set()
    operator.update()
    assert operator.focus_get() is auto_button

    auto_button.event_generate("<KeyPress-Return>")
    operator.update()
    assert invoked == ["start_auto"]


def test_space_on_a_focused_action_only_hovers_all_directions(operator, monkeypatch) -> None:
    commands: list[str] = []
    monkeypatch.setattr(operator, "send", lambda command, **_kwargs: commands.append(command))
    operator._nudge_keys_held.update(operator._nudge_key_map)
    operator._nudge_buttons_held.update({"左", "右", "前", "後"})
    operator._stick_vector_active = True

    auto_button = operator.flight_buttons["start_auto"]
    auto_button.focus_set()
    operator.update()
    auto_button.event_generate("<KeyPress-space>")
    operator.update()

    assert commands == ["hover"]
    assert not operator._nudge_keys_held
    assert not operator._nudge_buttons_held
    assert operator.stick_left.value == (0.0, 0.0)
    assert operator.stick_right.value == (0.0, 0.0)


def test_flight_action_normal_and_active_backgrounds_support_white_text(operator) -> None:
    style = ttk.Style(operator)
    seen_styles: set[str] = set()
    for command, button in operator.flight_buttons.items():
        style_name = str(button.cget("style") or button.winfo_class())
        if style_name in seen_styles:
            continue
        seen_styles.add(style_name)
        for state, state_spec in (("normal", ()), ("active", ("active",))):
            background = style.lookup(style_name, "background", state=state_spec)
            foreground = style.lookup(style_name, "foreground", state=state_spec)
            assert foreground.lower() in {"#ffffff", "white"}, (
                f"{command} {style_name} {state} must use white text, got "
                f"{foreground!r}"
            )
            assert _contrast_ratio("#ffffff", background) >= 4.5, (
                f"{command} {style_name} {state} contrast is too low: "
                f"white on {background!r}"
            )


def test_each_control_tab_fits_the_minimum_window_without_clipping(operator) -> None:
    operator.geometry(_window_geometry(app.UI_MIN_SIZE))
    operator.update_idletasks()
    root_left = operator.winfo_rootx()
    root_top = operator.winfo_rooty()
    root_right = root_left + operator.winfo_width()
    root_bottom = root_top + operator.winfo_height()

    for tab_id in operator.controls_notebook.tabs():
        operator.controls_notebook.select(tab_id)
        operator._fit_control_pane()
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
    for geometry in (
        _window_geometry(app.UI_STANDARD_SIZE),
        _window_geometry(app.UI_MIN_SIZE),
    ):
        operator.geometry(geometry)
        operator.update_idletasks()
        for tab_id in notebook.tabs():
            notebook.select(tab_id)
            operator._fit_control_pane()
            operator.update_idletasks()
            pane_bottom = pane.winfo_rooty() + pane.winfo_height()
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


def test_calibration_results_and_each_gravity_phase_fit_the_control_pane(operator) -> None:
    notebook = operator.controls_notebook
    pane = notebook.master
    calibration_tab = operator._preflight_tabs["compass"]
    notebook.select(calibration_tab)
    operator.drone_magnetometer_var.set(app.format_magnetometer_calibration(
        app.DroneState(
            drone_magnetometer_required=0,
            drone_magnetometer_started=False,
            drone_magnetometer_x_done=True,
            drone_magnetometer_y_done=True,
            drone_magnetometer_z_done=True,
            drone_magnetometer_failed=False,
        )
    )["drone"])

    for geometry in (
        _window_geometry(app.UI_STANDARD_SIZE),
        _window_geometry(app.UI_MIN_SIZE),
    ):
        operator.geometry(geometry)
        for phase in app.PHASES:
            operator.gravity_guide_var.set(
                app.gravity_phase_guidance(phase, sample_count=15, span_deg=90.0)
            )
            operator._fit_control_pane()
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


def test_gravity_workflow_preserves_samples_and_gates_phase_advance(
    operator, monkeypatch,
) -> None:
    operator.gravity_start()
    assert operator.gravity_cal.phase == "yaw"
    assert operator.gravity_next_button.instate(["disabled"])

    operator.gravity_next()
    assert operator.gravity_cal.phase == "yaw", "an incomplete phase must not advance"

    for index in range(24):
        operator.gravity_cal.add_sample(
            roll=math.radians(7.0),
            pitch=0.0,
            yaw=index / 24.0 * math.tau,
            t_mono=index * 0.05,
        )
    phase_result = operator.gravity_cal.analyze_phase("yaw")
    assert phase_result.ok is False, "tilt must remain a final quality failure"
    assert any("level tilt" in note for note in phase_result.notes)
    operator._refresh_gravity_controls()
    assert operator.gravity_next_button.instate(["!disabled"])
    operator.gravity_next()
    assert operator.gravity_cal.phase == "pitch"

    count_before = len(operator.gravity_cal.samples)
    monkeypatch.setattr(app.messagebox, "askyesno", lambda *_args, **_kwargs: False)
    operator.gravity_start()
    assert len(operator.gravity_cal.samples) == count_before
    assert operator.gravity_cal.phase == "pitch"


def test_yaw_gravity_check_accepts_stable_real_airframe_offset() -> None:
    calibrator = app.GravityCalibrator()
    calibrator.start()
    for index in range(24):
        calibrator.add_sample(
            roll=0.0,
            pitch=math.radians(5.1),
            yaw=index / 24.0 * math.tau,
            t_mono=index * 0.05,
        )

    result = calibrator.analyze_phase("yaw")
    assert result.ok is True
    assert result.level_tilt_deg == pytest.approx(5.1)

    calibrator.start()
    for index in range(24):
        calibrator.add_sample(
            roll=0.0,
            pitch=math.radians(6.5),
            yaw=index / 24.0 * math.tau,
            t_mono=index * 0.05,
        )

    result = calibrator.analyze_phase("yaw")
    assert result.ok is False
    assert any("level tilt" in note for note in result.notes)


def test_preflight_bar_hides_after_completion_and_returns_when_invalidated(
    operator, monkeypatch,
) -> None:
    assert operator.preflight_bar.winfo_manager() == "pack"
    evidence = {step: (step, "stable") for step in app.PREFLIGHT_GUIDE_STEPS}
    monkeypatch.setattr(
        operator,
        "_preflight_step_evidence",
        lambda step, _state, **_kwargs: (evidence[step], "可確認"),
    )
    for step in app.PREFLIGHT_GUIDE_STEPS:
        assert operator.preflight_guide.current_step == step
        operator.preflight_guide.confirm_current(evidence[step])

    operator._update_preflight_guide(operator.current_state)
    operator.update_idletasks()
    assert operator.preflight_bar.winfo_manager() == ""

    monkeypatch.setattr(
        operator,
        "_preflight_step_evidence",
        lambda step, _state, **_kwargs: (
            (None, "狀態已改變")
            if step == "compass"
            else (evidence[step], "可確認")
        ),
    )
    operator._update_preflight_guide(operator.current_state)
    operator.update_idletasks()
    assert operator.preflight_guide.complete is False
    assert operator.preflight_bar.winfo_manager() == "pack"
    assert operator.preflight_steps_frame.winfo_manager() == "pack"


def test_shrinking_the_window_takes_height_from_the_map_not_the_controls(operator) -> None:
    """The control pane is fixed-height; the map/video pane is the elastic one."""
    notebook = operator.controls_notebook
    pane = notebook.master

    operator.geometry(_window_geometry(app.UI_STANDARD_SIZE))
    operator.update_idletasks()
    tall_pane, tall_map = pane.winfo_height(), operator.map_label.winfo_height()

    # The minimum is shorter than the standard viewport, so the map/video pane
    # must absorb the reduction while the control pane stays fixed.
    operator.geometry(_window_geometry(app.UI_MIN_SIZE))
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
    assert "起飛後錄影" not in labels
    assert not hasattr(operator, "record_on_takeoff_var")
    assert operator.backend.record_on_takeoff is True
    assert operator.record_status_var.get().startswith("錄影:")


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
    assert heights["場域資產"] < max(heights.values())
    assert min(heights.values()) >= app.CONTROL_PANE_MIN_H


def test_localization_readout_is_in_the_always_visible_header(operator) -> None:
    """Localization health stays visible without obscuring the camera picture."""
    operator.update_idletasks()
    assert operator.loc_health_label is operator.status_chips["localization"]
    assert operator.loc_health_label.master is operator.status_strip
    assert not operator.video_label.winfo_children()


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
    assert "飛行限制與目前設定" in titles


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


def test_status_header_separates_identity_link_and_flight_state(operator) -> None:
    state = operator.backend.poll()
    operator._update_flight_header(state)
    texts = {name: widget.cget("text") for name, widget in operator.status_chips.items()}
    assert texts["identity"] == "SIMULATED"
    assert texts["link"].startswith("連線：")
    assert texts["flight"].startswith("飛行：")
    assert all(" | " not in text for text in texts.values())


def test_video_has_no_widget_overlay_and_one_engineering_hud(operator) -> None:
    """Requested engineering telemetry shares the image-FPS HUD."""
    operator.geometry(_window_geometry(app.UI_STANDARD_SIZE))
    operator.update_idletasks()
    assert not operator.video_label.winfo_children()
    source = inspect.getsource(app.OperatorApp.render_video)
    assert "_video_diagnostic_lines" in source
    assert "hud_h = 146" in source
    assert "hud_font = pil_ui_font(11)" in source
    assert "video=" not in source and "loc=" not in source and "det=" not in source


def test_localization_summary_and_diagnostics_are_not_duplicated(operator) -> None:
    """Only health is in the header; estimator internals stay in the video HUD."""
    operator.update_idletasks()
    header_text = operator.loc_health_label.cget("text")
    for token in ("inliers", "reproj", "pose_age", "stream_age"):
        assert token not in header_text
    assert "inliers" in operator.loc_quality_var.get()

    # The separate 位姿/影格 readout is gone; its colour grading moved to the
    # overlay's health label, which is the thing it was grading.
    assert not hasattr(operator, "loc_age_label")
    source = inspect.getsource(app.OperatorApp._update_age_readout)
    assert "loc_health_label" in source


def test_video_hud_contains_only_requested_engineering_metrics(operator) -> None:
    assert "診斷／紀錄" not in {
        operator.controls_notebook.tab(tab_id, "text")
        for tab_id in operator.controls_notebook.tabs()
    }
    joined = "\n".join(operator._video_diagnostic_lines())
    for token in (
        "速度上限", "wall_ms", "core", "e2e", "RTH", "GPS",
        "飛控高度", "AGL", "連接品質", "定位 FPS", "inliers",
        "飛控融合姿態", "三軸速度",
    ):
        assert token in joined
    for removed in (
        "硬體清單", "串流 FPS", "reproj", "地圖覆蓋", "風 ",
        "感測器", "偵測", "pose_age", "stream_age", "水平",
    ):
        assert removed not in joined


def test_flight_tab_only_shows_limit_settings_and_has_no_duplicate_hover(
    operator,
) -> None:
    stack = [operator._flight_tab]
    settings_panel = None
    flight_tab_texts = []
    while stack:
        parent = stack.pop()
        children = parent.winfo_children()
        stack.extend(children)
        for child in children:
            try:
                text = str(child.cget("text"))
            except Exception:
                continue
            flight_tab_texts.append(text)
            if text == "飛行限制與目前設定":
                settings_panel = child

    assert settings_panel is not None
    assert "懸停（全部歸零）" not in flight_tab_texts
    assert operator.flight_buttons["hover"].winfo_manager() == "pack"

    stack = [settings_panel]
    settings_texts = []
    while stack:
        parent = stack.pop()
        children = parent.winfo_children()
        stack.extend(children)
        for child in children:
            try:
                settings_texts.append(str(child.cget("text")))
            except Exception:
                continue
    joined = "\n".join(settings_texts)
    for token in ("目前設定", "高度", "距離", "圍欄", "速度上限"):
        assert token in joined
    for removed in (
        "硬體", "版本", "失聯策略", "起飛資料", "控制器",
        "ground speed", "PCMD", "map Y",
    ):
        assert removed not in joined
