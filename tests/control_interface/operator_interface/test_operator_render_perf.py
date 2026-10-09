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
import math
from tkinter import ttk

import pytest

HERE = (
    Path(__file__).resolve().parents[3]
    / "控制介面程式"
    / "operator_interface"
)
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
    instance = app.OperatorApp(backend, points, site_id="test")
    instance.update_idletasks()
    try:
        yield instance
    finally:
        # Tk keeps ``after`` callbacks in the Tcl event queue even when a test
        # destroys its root.  A later test can then dispatch the stale callback
        # after its Python command has gone away, making otherwise unrelated
        # focus assertions depend on test order.
        for callback_id in instance.tk.call("after", "info"):
            try:
                instance.after_cancel(callback_id)
            except tk.TclError:
                pass
        # Collect while the interpreter is still on the Tk main thread. Deferred
        # Variable.__del__ calls otherwise surface as an unraisable exception in
        # whichever unrelated test happens to trigger the collection.
        instance.destroy()
        gc.collect()


def test_ui_tick_defaults_to_30_hz_and_clamps_both_sides(operator) -> None:
    assert app.build_argument_parser().parse_args([]).tick_ms == 33
    assert operator.tick_ms == 33
    assert operator._tick_period_s == pytest.approx(0.033)
    # The 8 ms default used to saturate the Tcl timer queue (overdue ticks
    # reschedule 1 ms out, and due timers starve the result-notification
    # filehandler); the old behavior stays reachable as an explicit opt-out.
    fast = app.OperatorApp(app.DroneBackend(), operator.map_points, tick_ms=8)
    try:
        assert fast.tick_ms == 8
    finally:
        for callback_id in fast.tk.call("after", "info"):
            try:
                fast.after_cancel(callback_id)
            except tk.TclError:
                pass
        fast.destroy()


def _force_widget_focus(operator, widget) -> None:
    """Give a widget deterministic keyboard focus without a window-manager race."""
    operator.update_idletasks()
    widget.focus_force()
    operator.update()
    assert operator.focus_get() is widget



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


def test_controls_share_one_workspace_and_flight_actions_stay_visible(
    operator,
) -> None:
    """Flight, calibration and assets are visible together; abort actions stay above."""
    wanted = {"懸停", "原地降落", "手動/搖桿 (Esc)", "停止電腦動作"}
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
    ) == ("操作工作區",)
    assert set(operator.control_sections) == {"flight", "calibration", "assets"}
    assert all(section.winfo_ismapped() for section in operator.control_sections.values())
    assert "停止自動並懸停" not in all_labels

    # At least one instance of every abort action must stay above the workspace.
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
    _force_widget_focus(operator, auto_button)

    auto_button.event_generate("<KeyPress-Return>")
    operator.update()
    assert invoked == ["start_auto"]

    stop_button = operator.flight_buttons["emergency_stop"]
    stop_button.configure(command=lambda: invoked.append("emergency_stop"))
    _force_widget_focus(operator, stop_button)
    stop_button.event_generate("<KeyPress-Return>")
    operator.update()
    assert invoked == ["start_auto", "emergency_stop"]


def test_map_has_top_right_zoom_buttons_using_the_existing_zoom_path(operator) -> None:
    operator.update_idletasks()
    controls = operator.map_zoom_controls
    assert controls.winfo_manager() == "place"
    assert controls.winfo_x() + controls.winfo_width() >= operator.map_label.winfo_width() - 10

    for button in (operator.map_zoom_in_button, operator.map_zoom_out_button):
        assert button.winfo_reqwidth() >= 40
        assert button.winfo_reqheight() >= 40
        assert str(button.cget("takefocus")).lower() not in {"0", "false"}
    operator.geometry(_window_geometry(app.UI_MIN_SIZE))
    operator.update_idletasks()
    canvas = operator.map_label
    for button in (operator.map_zoom_in_button, operator.map_zoom_out_button):
        assert canvas.winfo_rooty() <= button.winfo_rooty()
        assert button.winfo_rooty() + button.winfo_height() <= canvas.winfo_rooty() + canvas.winfo_height()

    initial_zoom = operator.map_zoom
    operator.map_zoom_in_button.invoke()
    assert operator.map_zoom == pytest.approx(initial_zoom * 1.12)
    operator.map_zoom_out_button.invoke()
    assert operator.map_zoom == pytest.approx(initial_zoom)


def test_space_on_a_focused_action_only_hovers_all_directions(operator, monkeypatch) -> None:
    commands: list[str] = []
    monkeypatch.setattr(operator, "send", lambda command, **_kwargs: commands.append(command))
    operator._nudge_keys_held.update(operator._nudge_key_map)
    operator._nudge_buttons_held.update({"左", "右", "前", "後"})
    operator._stick_vector_active = True

    auto_button = operator.flight_buttons["start_auto"]
    _force_widget_focus(operator, auto_button)
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


def test_short_viewport_renders_at_canvas_height_without_cropping(operator) -> None:
    operator.geometry(_window_geometry(app.UI_MIN_SIZE))
    operator.update_idletasks()
    requested_heights = (
        operator.map_label.winfo_reqheight(), operator.video_label.winfo_reqheight(),
    )
    operator.tick()
    operator.update_idletasks()
    operator.tick()
    operator.redraw_map_only()
    assert app.VIEWPORT_MIN_H <= operator.video_label.winfo_height() < 220
    assert operator.video_photo.height() <= operator.video_label.winfo_height()
    assert operator.video_photo.width() <= operator.video_label.winfo_width()
    assert operator.map_photo.height() == operator.map_label.winfo_height()
    # Presenting resized images must not feed allocated height back into the
    # Canvas request, otherwise later shrinks steal height from fixed controls.
    assert (
        operator.map_label.winfo_reqheight(), operator.video_label.winfo_reqheight(),
    ) == requested_heights




def test_long_route_items_remain_bounded_and_clear_when_route_shrinks(operator) -> None:
    operator.route_pts = [
        tuple(point) for point in np.random.default_rng(1).random((2000, 3)) * 5.0
    ]
    operator.history = []
    operator.history_health = []
    operator.no_loc_markers = []
    state = operator.backend.poll()
    operator.present_map(400, 300, state)
    assert len(operator.map_label.find_withtag("map-overlay")) <= 2 * app.ROUTE_DOT_MAX + 8

    operator.route_pts[:] = operator.route_pts[:2]
    operator.present_map(400, 300, state)
    assert len(operator.map_label.find_withtag("map-overlay")) <= 10


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


def test_firmware_calibration_results_fit_the_control_pane(operator) -> None:
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
                        f"{geometry}: compass panel clips by "
                        f"{bottom - pane_bottom}px"
                    )


def test_yaw_gravity_check_accepts_stable_real_airframe_offset() -> None:
    from gravity_calibration import GravityCalibrator

    calibrator = GravityCalibrator()
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

    for text in ("重設地圖",):
        assert text in labels, f"{text} is no longer a child of the map canvas"
    assert "顯示規劃路徑" not in labels
    assert "上下翻面" not in labels
    assert not hasattr(operator, "flip_map_vertical")

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
    assert "鏡頭" in labels
    assert "起飛" in labels, "takeoff must still live in the always-visible bar"
    assert "起飛後錄影" in labels
    assert operator.record_on_takeoff_var.get() is False
    assert operator.backend.record_on_takeoff is False
    assert operator.record_status_var.get().startswith("錄影:")


def test_record_on_takeoff_disarmed_by_default_and_operator_can_arm(operator) -> None:
    assert operator.backend.record_on_takeoff is False
    operator.record_on_takeoff_var.set(True)
    operator._on_record_on_takeoff_toggled()
    assert operator.backend.record_on_takeoff is True
    operator.record_on_takeoff_var.set(False)
    operator._on_record_on_takeoff_toggled()
    assert operator.backend.record_on_takeoff is False


def test_preflight_navigation_keeps_all_sections_visible(operator) -> None:
    notebook = operator.controls_notebook
    workspace = notebook.select()
    for step in app.PREFLIGHT_GUIDE_STEPS:
        operator._select_preflight_tab(step)
        operator.update_idletasks()
        assert notebook.select() == workspace
        assert all(section.winfo_ismapped() for section in operator.control_sections.values())
    operator._select_flight_tab()
    assert notebook.select() == workspace
    assert notebook.master.winfo_height() >= app.CONTROL_PANE_MIN_H


@pytest.mark.parametrize("axis", ["xAxis", "yAxis", "zAxis"])
def test_compass_rotation_instructions_fit_the_shared_workspace(operator, axis) -> None:
    operator.geometry(_window_geometry(app.UI_MIN_SIZE))
    operator._draw_magnetometer_axis(app.magnetometer_axis_guide(axis))
    operator.update_idletasks()
    canvas = operator.magnetometer_axis_canvas
    for item in canvas.find_all():
        left, top, right, bottom = canvas.bbox(item)
        assert 0 <= left <= right <= canvas.winfo_width()
        assert 0 <= top <= bottom <= canvas.winfo_height()


@pytest.mark.parametrize("route_count", [2, 5])
def test_imported_route_choices_fit_beside_flight_and_calibration(
    operator, monkeypatch, route_count,
) -> None:
    from types import SimpleNamespace
    import site_assets_panel

    monkeypatch.setattr(site_assets_panel, "describe_site_routes", lambda _root: [
        SimpleNamespace(
            label=f"巡檢航線 {index + 1:02d}", path=Path(f"route_{index + 1:02d}.json"),
            flight_ready=index == 0,
        )
        for index in range(route_count)
    ])
    panel = operator.site_assets_panel
    panel.activate_site("/tmp/layout-profile.json", "/tmp/layout-site")
    operator.geometry(_window_geometry(app.UI_MIN_SIZE))
    operator.update_idletasks()
    operator._fit_control_pane()
    operator.update_idletasks()
    assert operator.video_label.winfo_height() >= app.VIEWPORT_MIN_H
    assert len(panel._permanently_disabled) >= 1
    assert panel.winfo_rooty() + panel.winfo_height() <= operator.winfo_rooty() + operator.winfo_height()
    for child in panel.winfo_children():
        if child.grid_info().get("row") == 5:
            for route_card in child.winfo_children():
                route_card.event_generate("<Enter>")
    operator.update_idletasks()
    for button in panel._buttons:
        assert button.winfo_ismapped()
        assert button.winfo_rootx() >= panel.winfo_rootx()
        assert button.winfo_rootx() + button.winfo_width() <= panel.winfo_rootx() + panel.winfo_width()
        assert button.winfo_rooty() + button.winfo_height() <= operator.winfo_rooty() + operator.winfo_height()
    for command in ("manual", "hover", "land", "emergency_stop"):
        assert operator.flight_buttons[command].winfo_ismapped()


def test_localization_readout_is_in_the_always_visible_header(operator) -> None:
    """Localization health stays visible without obscuring the camera picture."""
    operator.update_idletasks()
    assert operator.loc_health_label is operator.status_chips["localization"]
    assert operator.loc_health_label.master is operator.status_strip
    assert operator.video_label.winfo_children() == [operator.depth_view_button.master]


def test_virtual_sticks_share_the_workspace_with_the_readouts(operator) -> None:
    """The sticks stay beside flight settings within the shared workspace."""
    notebook = operator.controls_notebook
    target = [t for t in notebook.tabs() if notebook.tab(t, "text") == "操作工作區"]
    assert target, "the shared workspace is missing"
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


def _video_overlay_text(operator) -> str:
    canvas = operator.video_label
    return "\n".join(
        canvas.itemcget(item, "text")
        for item in canvas.find_withtag("video-overlay") if canvas.type(item) == "text"
    )


def test_held_video_updates_hud_without_reuploading_pixels(operator, monkeypatch) -> None:
    operator.video_frame = np.full((720, 1280, 3), (210, 50, 30), dtype=np.uint8)
    operator._video_frame_stamp = 1.0
    state = operator.backend.poll()
    state.frame_age_ms = 10.0
    operator.render_video(640, 360, state)
    photo = operator.video_photo
    def unexpected_upload(_image):
        raise AssertionError("HUD-only update must not upload the held video again")
    monkeypatch.setattr(photo, "paste", unexpected_upload)
    state.frame_age_ms = 1234.0
    operator.render_video(640, 360, state)
    assert operator.video_photo is photo
    assert "1234 ms" in _video_overlay_text(operator)
    assert operator.video_label.tk.call(str(photo), "get", 320, 180) == (210, 50, 30)


def test_video_resize_centres_pixels_and_scales_detection_labels(operator, monkeypatch) -> None:
    operator.video_frame = np.full((720, 1280, 3), (25, 70, 150), dtype=np.uint8)
    operator._video_frame_stamp = 1.0
    operator.detection_result = {
        "success": True, "boxes": [
            {"xyxy": [200, 100, 400, 200], "class_name": "target", "confidence": 0.9},
        ],
    }
    state = operator.backend.poll()
    operator.render_video(640, 480, state)
    canvas = operator.video_label
    frame_item, = canvas.find_withtag("video-frame")
    assert (operator.video_photo.width(), operator.video_photo.height()) == (640, 360)
    assert canvas.coords(frame_item) == [0.0, 60.0]
    labels = [item for item in canvas.find_withtag("video-overlay")
              if canvas.type(item) == "text" and canvas.itemcget(item, "text").startswith("target ")]
    assert len(labels) == 1
    assert canvas.coords(labels[0]) == [104.0, 111.0]
    def unexpected_upload(_image):
        raise AssertionError("letterbox-only resize must reuse the fitted pixels")
    monkeypatch.setattr(operator.video_photo, "paste", unexpected_upload)
    operator.render_video(800, 360, state)
    assert (operator.video_photo.width(), operator.video_photo.height()) == (640, 360)
    assert canvas.coords(frame_item) == [80.0, 0.0]
    assert canvas.coords(labels[0]) == [184.0, 51.0]


def test_video_banner_priority_and_removed_detections_do_not_leave_stale_items(operator, monkeypatch) -> None:
    operator.video_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    operator._video_frame_stamp = 1.0
    state = operator.backend.poll()
    operator.inspecting = True
    operator.loc_health = "FAIL"
    operator.detection_result = {
        "success": True, "boxes": [
            {"xyxy": [20, 20, 120, 100], "class_name": "target", "confidence": 0.8},
        ],
    }
    monkeypatch.setattr(operator, "_is_live_backend", lambda: True)
    state.link_ok = False
    operator.render_video(640, 360, state)
    assert "LINK LOST" in _video_overlay_text(operator)
    assert "LOCALIZATION LOST" not in _video_overlay_text(operator)
    operator.detection_result = None
    state.link_ok = True
    operator.render_video(640, 360, state)
    assert "LOCALIZATION LOST" in _video_overlay_text(operator)
    assert "LINK LOST" not in _video_overlay_text(operator)
    assert "target " not in _video_overlay_text(operator)
    count = len(operator.video_label.find_withtag("video-overlay"))
    for age in range(20):
        state.frame_age_ms = float(age)
        operator.render_video(640, 360, state)
    assert len(operator.video_label.find_withtag("video-overlay")) == count


def test_scheduled_held_video_refreshes_age_and_telemetry_without_a_new_frame(operator, monkeypatch) -> None:
    import operator_tick

    operator.video_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    operator._video_frame_stamp = 1.0
    state = operator.backend.poll()
    state.frame_age_ms = 20.0
    operator._map_dirty_key = operator_tick._map_dirty_key(
        operator, state,
        width=max(300, operator.map_label.winfo_width()),
        height=max(1, operator.map_label.winfo_height()),
    )
    operator._video_dirty_key = None
    operator_tick._render_if_dirty(operator, state)
    state.frame_age_ms = 2000.0
    operator._device_telemetry_lines = ("fresh-device-reading",)
    def unexpected_upload(_image):
        raise AssertionError("scheduled HUD refresh must retain held pixels")
    monkeypatch.setattr(operator.video_photo, "paste", unexpected_upload)
    operator_tick._render_if_dirty(operator, state)
    assert "2000 ms" in _video_overlay_text(operator)
    assert "fresh-device-reading" in _video_overlay_text(operator)


def test_video_hud_fits_short_viewport_without_obscuring_video(operator, monkeypatch) -> None:
    lines = (
        "AUTO PCMD ±10% | 定位 FPS 0.0 | inliers 0",
        "wall_ms - | core - | e2e -",
        "RTH UNKNOWN/UNKNOWN | GPS ? | 衛星 ?",
        "飛控高度 [相對起飛點] ? m | AGL ? m | 連接品質 ?",
        "飛控融合姿態 roll +0.0° | pitch +0.0° | yaw +0.0°",
        "三軸速度 N ? | E ? | D ? m/s",
        "全場尺度：等待有效飛行軌跡 (m/u)",
        "電池 100% | 未知 V | 健康 未知% | 循環 未知",
        "電池警告 未知 | 遙控器電池 未知",
        "相機 未知 | 曝光 未知 ISO 未知 | 白平衡 未知",
        "機載錄影 未知 | 未知 fps 未知 Mb/s | 無損變焦上限 未知x",
        "雲台絕對 R/P/Y 未知/-20.0/未知 | 相對 R/P/Y 未知/未知/未知",
        "雲台校正 未知 | 飽和 未知 | 鏡軸方向 未知",
        "XYZ 未知 | 速度 未知 | 飛行模式 未知",
    )
    monkeypatch.setattr(operator, "_video_diagnostic_lines", lambda: lines)
    # Exercise the HUD over actual video, not the separate no-stream placeholder.
    operator.video_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    state = operator.backend.poll()
    operator.render_video(924, 154, state)
    canvas = operator.video_label
    assert all(
        not canvas.itemcget(item, "fill")
        for item in canvas.find_withtag("video-overlay") if canvas.type(item) == "rectangle"
    )
    texts = [
        item for item in canvas.find_withtag("video-overlay")
        if canvas.type(item) == "text" and canvas.itemcget(item, "fill") == "#f4f4f4"
    ]
    assert len(texts) == len(lines) + 1
    for item in texts:
        x1, y1, x2, y2 = canvas.bbox(item)
        assert 18 <= x1 <= x2 <= 924 - 18
        assert 6 <= y1 <= y2 <= 154 - 6


def test_minimum_video_with_link_banner_keeps_source_age_and_scale_readable(operator, monkeypatch) -> None:
    lines = ("AUTO PCMD ±10%",) * 6 + ("全場尺度候選 0.42 m/u（尚不可用）",) + ("次要設備資訊",) * 8
    monkeypatch.setattr(operator, "_video_diagnostic_lines", lambda: lines)
    monkeypatch.setattr(operator, "_is_live_backend", lambda: True)
    state = operator.backend.poll()
    state.link_ok = False
    state.frame_age_ms = 24500.0
    operator.render_video(760, 96, state)
    canvas = operator.video_label
    texts = [item for item in canvas.find_withtag("video-overlay") if canvas.type(item) == "text"]
    hud = [item for item in texts if canvas.itemcget(item, "fill") == "#f4f4f4" and (
        "24500 ms" in canvas.itemcget(item, "text")
        or "全場尺度候選" in canvas.itemcget(item, "text")
    )]
    assert len(hud) == 2
    assert "LINK LOST" in _video_overlay_text(operator)
    for item in hud:
        _, top, _, bottom = canvas.bbox(item)
        assert 72 <= top <= bottom <= 96
    source, scale = (canvas.bbox(item) for item in hud)
    assert (
        source[2] <= scale[0] or scale[2] <= source[0]
        or source[3] <= scale[1] or scale[3] <= source[1]
    ), "freshness and scale must not overlap in the remaining banner-free band"


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


def test_video_hud_contains_only_requested_engineering_metrics(operator) -> None:
    assert "診斷／紀錄" not in {
        operator.controls_notebook.tab(tab_id, "text")
        for tab_id in operator.controls_notebook.tabs()
    }
    joined = "\n".join(operator._video_diagnostic_lines())
    for token in (
        "AUTO PCMD", "wall_ms", "core", "e2e", "RTH", "GPS",
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
    for token in (
        "目前設定", "高度", "距離", "圍欄", "AUTO PCMD",
    ):
        assert token in joined
    for removed in (
        "硬體", "版本", "失聯策略", "起飛資料", "控制器",
        "ground speed", "map Y",
    ):
        assert removed not in joined


def test_video_hud_updates_dynamic_metrics_without_remeasuring_static_rows(operator, monkeypatch) -> None:
    """Result ingestion must not redo the expensive unchanged CJK text widths."""
    operator.video_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    operator._video_frame_stamp = 1.0
    state = operator.backend.poll()
    static = "全場尺度：等待有效飛行軌跡 (m/u)"
    lines = [static, "latency 10.0ms"]
    monkeypatch.setattr(operator, "_video_diagnostic_lines", lambda: tuple(lines))
    operator.render_video(640, 360, state)
    draw = operator._video_presenter.draw
    original = draw.textlength
    static_measures = []

    def measured(text, *, font):
        if text == static:
            static_measures.append(text)
        return original(text, font=font)

    monkeypatch.setattr(draw, "textlength", measured)
    for value in range(11, 20):
        lines[1] = f"latency {value}.0ms"
        state.frame_age_ms = float(value)
        operator.render_video(640, 360, state)
    assert static_measures == []
    assert "latency 19.0ms" in _video_overlay_text(operator)
    assert "latency 10.0ms" not in _video_overlay_text(operator)
    assert static in _video_overlay_text(operator)


def test_video_hud_reflows_changed_row_instead_of_using_its_old_width(operator, monkeypatch) -> None:
    operator.video_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    operator._video_frame_stamp = 1.0
    state = operator.backend.poll()
    lines = ["short diagnostic"]
    monkeypatch.setattr(operator, "_video_diagnostic_lines", lambda: tuple(lines))
    operator.render_video(640, 360, state)
    lines[0] = "Updated diagnostic " + "W" * 45
    operator.render_video(640, 360, state)
    canvas = operator.video_label
    updated = [item for item in canvas.find_withtag("video-overlay")
               if canvas.type(item) == "text" and canvas.itemcget(item, "text") == lines[0]]
    assert lines[0] in _video_overlay_text(operator)
    assert "short diagnostic" not in _video_overlay_text(operator)
    for item in updated:
        left, top, right, bottom = canvas.bbox(item)
        assert 0 <= left < right <= 640
        assert 0 <= top < bottom <= 360


def test_map_updates_trails_without_uploading_cloud_and_removes_old_markers(operator, monkeypatch) -> None:
    state = operator.backend.poll()
    operator.route_pts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    operator.history = [np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0, 0.0])]
    operator.history_health = ["OK", "LOW"]
    operator.present_map(640, 360, state)
    photo = operator.map_photo

    def unexpected_upload(_image):
        raise AssertionError("a trail/pose change must not upload the cloud")

    monkeypatch.setattr(photo, "paste", unexpected_upload)
    before = [operator.map_label.coords(item) for item in operator.map_label.find_withtag("map-overlay")]
    operator.history[-1][0] = 1.5
    state.pose[0] += 0.5
    operator.present_map(640, 360, state)
    after = [operator.map_label.coords(item) for item in operator.map_label.find_withtag("map-overlay")]
    assert after != before
    assert operator.map_photo is photo
    count = len(operator.map_label.find_withtag("map-overlay"))
    operator.history_health[-1] = "OK"
    operator.present_map(640, 360, state)
    assert len(operator.map_label.find_withtag("map-overlay")) == count - 1
    operator.present_map(420, 280, state)
    assert operator.map_photo is not photo
    assert (operator.map_photo.width(), operator.map_photo.height()) == (420, 280)


def test_pending_map_paint_does_not_delay_video_alarm_or_recovery(operator, monkeypatch) -> None:
    import operator_tick

    operator.video_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    operator._video_frame_stamp = 1.0
    operator.inspecting = True
    state = operator.backend.poll()
    operator.loc_health = "OK"
    operator_tick._render_if_dirty(operator, state)
    operator._map_render_next = float("inf")
    state.pose[0] += 0.5
    operator.loc_health = "FAIL"
    operator_tick._render_if_dirty(operator, state)
    assert "LOCALIZATION LOST" in _video_overlay_text(operator)
    operator.loc_health = "OK"
    operator_tick._render_if_dirty(operator, state)
    assert "LOCALIZATION LOST" not in _video_overlay_text(operator)
    # A resize is not a paced pose update: both surfaces must fit immediately.
    monkeypatch.setattr(operator.map_label, "winfo_width", lambda: 620)
    monkeypatch.setattr(operator.map_label, "winfo_height", lambda: 340)
    operator_tick._render_if_dirty(operator, state)
    assert (operator.map_photo.width(), operator.map_photo.height()) == (620, 340)
