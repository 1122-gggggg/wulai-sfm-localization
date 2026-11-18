from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# The editor resolves the site's measured basis from the canonical flight-control
# owner, matching the application after runtime mirrors were removed.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[3] / "定位演算法" / "flight_control"),
)

import real_path_follow_controller as rpf
import local_site_assets as lsa
import route_editor_controller as rec
import route_editor_model as rem
from route_editor_controller import OrthoView, RouteEditorController, pointer_is_drag
from route_editor_objects import ObjectBoxController
from route_editor_wires import WireController
from route_editor_model import (
    PREVIEW_ROUTE_SCHEMA,
    ROUTE_SCHEMA,
    RouteDocument,
    editor_to_glomap,
    glomap_to_editor,
)
from route_editor_rendering import BACKGROUND, PointCloudRenderer, PointCloudRenderWorker


def test_editor_frame_round_trip_keeps_z_as_physical_up() -> None:
    glomap = np.array([[1.0, -3.0, 2.0], [-4.0, 0.5, 7.0]])

    editor = glomap_to_editor(glomap)

    assert editor.tolist() == [[1.0, 2.0, 3.0], [-4.0, 7.0, -0.5]]
    np.testing.assert_allclose(editor_to_glomap(editor), glomap)


def test_route_document_writes_the_validated_site_bound_contract(tmp_path) -> None:
    document = RouteDocument(
        site_id="yard",
        coordinate_frame_id="yard-map-v1",
        points=[[1.0, 2.0, 3.0], [2.0, 2.5, 4.0]],
    )

    path = document.save(tmp_path / "draft.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == ROUTE_SCHEMA
    assert payload["site_id"] == "yard"
    assert payload["coordinate_frame_id"] == "yard-map-v1"
    assert payload["frame"] == "aligned"
    assert payload["purpose"] == "flight"
    assert payload["closed"] is False
    assert payload["waypoints"] == document.points
    assert "site profile" in payload["note"]


def test_ply_only_draft_is_explicitly_unbound_and_not_a_flight_route(tmp_path) -> None:
    document = RouteDocument(
        site_id="must-not-leak",
        coordinate_frame_id="must-not-leak",
        points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )

    path = document.save(tmp_path / "preview.json", preview_only=True)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == PREVIEW_ROUTE_SCHEMA
    assert payload["site_id"] is None
    assert payload["coordinate_frame_id"] is None
    assert payload["purpose"] == "preview_only"
    assert "cannot be imported" in payload["note"]


def test_successful_route_save_approves_and_closes_the_finished_editor(tmp_path) -> None:
    import route_editor_window as rew

    saved = tmp_path / "flight_route.json"
    status = []
    closed = []
    imported_payloads = []

    def import_route(path):
        imported_payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return SimpleNamespace(
            message="航線已核准供自動巡航使用",
            asset_path=saved,
            approved_for_auto=True,
        )

    editor = SimpleNamespace(
        objects=ObjectBoxController(),
        wires=WireController(),
        _preview_only=False,
        controller=RouteEditorController([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        document=RouteDocument(site_id="yard", coordinate_frame_id="yard-map-v1"),
        arrive_radius_var=SimpleNamespace(get=lambda: 0.025),
        import_route=import_route,
        profile=SimpleNamespace(
            route_json=saved, asset_sha256=SimpleNamespace(map_ply="a" * 64),
        ),
        _signature=lambda: ("saved",),
        status_var=SimpleNamespace(set=status.append),
        close_editor=lambda *, force=False: closed.append(force),
    )
    editor._update_document = lambda: rew.RouteEditorWindow._update_document(editor)
    editor._refresh_distance_labels = lambda: None

    result = rew.RouteEditorWindow.save_route(editor)

    assert result == saved
    assert imported_payloads[0]["arrive_radius_map_units"] == pytest.approx(0.025)
    assert imported_payloads[0]["map_ply_sha256"] == "a" * 64
    assert "max_route_deviation_map_units" not in imported_payloads[0]
    assert "SHA 已同步" in status[-1]
    assert closed == [True]


def test_save_and_test_starts_only_after_a_successful_bound_save(tmp_path) -> None:
    import route_editor_window as rew

    saved = tmp_path / "flight_route.json"
    tested = []
    editor = SimpleNamespace(
        save_route=lambda: saved,
        test_route_callback=tested.append,
    )

    result = rew.RouteEditorWindow.save_and_test_route(editor)

    assert result == saved
    assert tested == [saved]

    editor.save_route = lambda: None
    assert rew.RouteEditorWindow.save_and_test_route(editor) is None
    assert tested == [saved]


def test_route_document_loads_raw_glomap_into_editor_z_up(tmp_path) -> None:
    path = tmp_path / "route.json"
    path.write_text(
        json.dumps(
            {
                "site_id": "yard",
                "coordinate_frame_id": "yard-map-v1",
                "frame": "glomap",
                "closed": False,
                "waypoints": [[1.0, -3.0, 2.0], [4.0, -6.0, 5.0]],
            }
        ),
        encoding="utf-8",
    )

    document = RouteDocument.load(
        path, site_id="yard", coordinate_frame_id="yard-map-v1"
    )

    assert document.points == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_route_document_rejects_another_site_frame(tmp_path) -> None:
    path = tmp_path / "route.json"
    path.write_text(
        json.dumps(
            {
                "site_id": "yard",
                "coordinate_frame_id": "another-map",
                "frame": "aligned",
                "waypoints": [[0, 0, 0], [1, 0, 0]],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="coordinate_frame_id"):
        RouteDocument.load(path, site_id="yard", coordinate_frame_id="yard-map-v1")


def test_two_stage_controller_supports_move_cancel_undo_and_redo() -> None:
    controller = RouteEditorController()
    controller.set_placement_enabled(True)
    controller.add([0, 0, 0])
    controller.add([1, 0, 0])
    controller.finish_layout()
    controller.selected = 1

    assert controller.begin_move() is True
    controller.constrain(2)
    controller.preview_move([0, 0, 3])
    assert controller.points[1] == [1.0, 0.0, 3.0]
    assert controller.cancel_move() is True
    assert controller.points[1] == [1.0, 0.0, 0.0]

    controller.begin_move()
    controller.preview_move([0, 0, 3])
    controller.confirm_move()
    assert controller.undo() is True
    assert controller.points[1] == [1.0, 0.0, 0.0]
    assert controller.redo() is True
    assert controller.points[1] == [1.0, 0.0, 3.0]


def test_waypoints_require_explicit_placement_mode_and_stop_after_stage_one() -> None:
    controller = RouteEditorController()

    with pytest.raises(ValueError, match="標路徑點模式"):
        controller.add([0, 0, 0])

    assert controller.set_placement_enabled(True) is True
    controller.add([0, 0, 0])
    controller.add([1, 0, 0])
    controller.finish_layout()

    assert controller.placement_enabled is False
    assert controller.set_placement_enabled(True) is False
    with pytest.raises(ValueError, match="第一階段"):
        controller.add([2, 0, 0])


def test_g_z_screen_motion_changes_only_editor_height() -> None:
    view = OrthoView(center=np.zeros(3), radius=10.0)
    view.front()

    delta = view.screen_delta_to_world(0, -100, 1000, 800, axis=2)

    assert delta[0] == pytest.approx(0.0)
    assert delta[1] == pytest.approx(0.0)
    assert delta[2] > 0.0


def test_left_view_mirrors_right_view_with_height_up() -> None:
    point = [[2.0, 3.0, 5.0]]
    right = OrthoView(center=np.zeros(3), radius=10.0)
    left = OrthoView(center=np.zeros(3), radius=10.0)
    right.right()
    left.left()

    rx, ry, rd = right.project(point, 1000, 800)
    lx, ly, ld = left.project(point, 1000, 800)

    assert lx[0] - 500.0 == pytest.approx(-(rx[0] - 500.0))
    assert ly[0] == pytest.approx(ry[0])
    assert ly[0] < 400.0
    assert ld[0] == pytest.approx(-rd[0])


def test_five_pixel_threshold_separates_click_from_left_drag_rotation() -> None:
    assert pointer_is_drag((100, 100), (103, 104)) is False
    assert pointer_is_drag((100, 100), (106, 100)) is True


# --- 2026-08-06: the editor frame is a MEASUREMENT, and the radius is confirmed ---

def test_editor_transform_round_trips_through_the_measured_basis():
    """A mismatched pair silently rotates every exported waypoint."""
    frame = rpf.MapFrame.from_gravity([0.009067509372034937,
                                       0.9237964066045453,
                                       0.38277667042064856])
    glomap = np.array([[0.3, -1.2, 0.8], [2.0, 0.5, -0.4]])

    editor = rem.glomap_to_editor(glomap, frame)
    back = rem.editor_to_glomap(editor, frame)

    assert np.allclose(back, glomap, atol=1e-12)
    # Up in the editor is +Z, and it must be the MEASURED up.
    up_only = rem.editor_to_glomap(np.array([[0.0, 0.0, 1.0]]), frame)[0]
    assert np.allclose(up_only, frame.up, atol=1e-12)
    # Reading the same editor points back with the legacy basis is wrong, and by
    # how much: this is the defect the frame parameter exists to prevent.
    wrong = rem.editor_to_glomap(editor, None)
    assert float(np.linalg.norm(wrong - glomap)) > 0.4


def test_editor_defaults_to_the_legacy_swap_without_a_frame():
    """Sites with no T_align_gravity.json must behave exactly as before."""
    glomap = np.array([[1.0, 2.0, 3.0]])
    editor = rem.glomap_to_editor(glomap, None)
    assert np.allclose(editor, [[1.0, 3.0, -2.0]])
    assert np.allclose(rem.editor_to_glomap(editor, None), glomap)


def test_saved_route_declares_which_alignment_produced_it(tmp_path):
    """'aligned' alone is ambiguous; the flight loader refuses a route without it."""
    document = rem.RouteDocument(site_id="s", coordinate_frame_id="f")
    document.points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    document.align_source = "measured"
    document.arrive_radius_map_units = 0.3

    saved = document.save(tmp_path / "flight_path.json")
    payload = json.loads(saved.read_text(encoding="utf-8"))

    assert payload["frame"] == "aligned"
    assert payload["align_source"] == "measured"
    assert payload["arrive_radius_map_units"] == 0.3
    assert "max_route_deviation_map_units" not in payload


def test_editor_arrive_limit_round_trips(tmp_path):
    source = tmp_path / "route.json"
    source.write_text(
        json.dumps(
            {
                "site_id": "yard",
                "coordinate_frame_id": "yard-map-v1",
                "frame": "aligned",
                "closed": False,
                "waypoints": [[0, 0, 0], [1, 0, 0]],
                "arrive_radius_map_units": 0.02,
            }
        ),
        encoding="utf-8",
    )

    document = RouteDocument.load(
        source,
        site_id="yard",
        coordinate_frame_id="yard-map-v1",
    )

    assert document.arrive_radius_map_units == pytest.approx(0.02)


def test_save_preview_persists_the_dragged_arrive_limit(tmp_path):
    import route_editor_window as rew

    target = tmp_path / "route_preview.json"
    document = RouteDocument(site_id="yard", coordinate_frame_id="yard-map-v1")
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    editor = SimpleNamespace(
        objects=ObjectBoxController(),
        wires=WireController(),
        controller=RouteEditorController(points),
        _choose_preview_path=lambda: target,
        document=document,
        arrive_radius_var=SimpleNamespace(get=lambda: 0.025),
        _preview_only=True,
        _signature=lambda: ("saved",),
        status_var=SimpleNamespace(set=lambda _message: None),
    )
    editor._update_document = lambda: rew.RouteEditorWindow._update_document(editor)
    editor._refresh_distance_labels = lambda: None

    rew.RouteEditorWindow.save_route(editor)
    payload = json.loads(target.read_text(encoding="utf-8"))

    assert payload["purpose"] == "preview_only"
    assert payload["arrive_radius_map_units"] == pytest.approx(0.025)
    assert "max_route_deviation_map_units" not in payload


def test_bound_save_stages_before_atomic_route_import(tmp_path):
    import route_editor_window as rew

    target = tmp_path / "current_route.json"
    RouteDocument(
        site_id="yard",
        coordinate_frame_id="yard-map-v1",
        points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ).save(target)
    document = RouteDocument.load(
        target,
        site_id="yard",
        coordinate_frame_id="yard-map-v1",
    )
    replacement = [[0.0, 0.0, 0.0], [2.0, 1.0, 0.5]]
    imported = []

    def import_route(path):
        imported.append(json.loads(path.read_text(encoding="utf-8")))
        current = json.loads(target.read_text(encoding="utf-8"))
        assert current["waypoints"] != replacement
        return SimpleNamespace(message="saved", asset_path=target, approved_for_auto=True)

    editor = SimpleNamespace(
        objects=ObjectBoxController(),
        wires=WireController(),
        controller=RouteEditorController(replacement),
        document=document,
        arrive_radius_var=SimpleNamespace(get=lambda: 0.025),
        _preview_only=False,
        import_route=import_route,
        profile=SimpleNamespace(
            route_json=target, asset_sha256=SimpleNamespace(map_ply="a" * 64),
        ),
        _signature=lambda: ("saved",),
        status_var=SimpleNamespace(set=lambda _message: None),
        close_editor=lambda *, force=False: None,
    )
    editor._update_document = lambda: rew.RouteEditorWindow._update_document(editor)
    editor._refresh_distance_labels = lambda: None

    saved = rew.RouteEditorWindow.save_route(editor)

    assert saved == target.resolve()
    assert imported[0]["waypoints"] == replacement
    assert not (tmp_path / "route_drafts").exists()


def test_bound_save_map_mismatch_points_to_reselecting_the_correct_map():
    import route_editor_window as rew

    assert rew._save_failure_hint(ValueError("something else")) is None
    hint = rew._save_failure_hint(
        ValueError("managed site adoption map/coordinate identity mismatch")
    )
    assert hint is not None and "重新選擇正確地圖" in hint

    shown = []

    def import_route(_path):
        raise ValueError("managed site adoption map/coordinate identity mismatch")

    document = RouteDocument(
        site_id="yard",
        coordinate_frame_id="yard-map-v1",
        points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )
    editor = SimpleNamespace(
        objects=ObjectBoxController(),
        wires=WireController(),
        controller=RouteEditorController(document.points),
        document=document,
        arrive_radius_var=SimpleNamespace(get=lambda: 0.025),
        _preview_only=False,
        import_route=import_route,
        profile=SimpleNamespace(asset_sha256=SimpleNamespace(map_ply="a" * 64)),
        _signature=lambda: ("saved",),
        status_var=SimpleNamespace(set=shown.append),
        close_editor=lambda *, force=False: None,
    )
    editor._update_document = lambda: rew.RouteEditorWindow._update_document(editor)
    editor._refresh_distance_labels = lambda: None

    assert rew.RouteEditorWindow.save_route(editor) is None
    assert len(shown) == 1
    assert shown[0].startswith("航線儲存失敗：")
    assert "重新選擇正確地圖" in shown[0]


@pytest.mark.parametrize("approved,has_asset", [(False, True), (True, False)])
def test_bound_save_without_committed_approval_keeps_editor_unsaved(
    tmp_path, approved, has_asset,
):
    import route_editor_window as rew

    document = RouteDocument(
        site_id="yard", coordinate_frame_id="yard-map-v1",
        points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )
    shown, closed = [], []
    editor = SimpleNamespace(
        objects=ObjectBoxController(),
        wires=WireController(),
        document=document,
        profile=SimpleNamespace(asset_sha256=SimpleNamespace(map_ply="a" * 64)),
        _preview_only=False,
        _update_document=lambda: None,
        _refresh_distance_labels=lambda: None,
        import_route=lambda _path: SimpleNamespace(
            asset_path=tmp_path / "saved.json" if has_asset else None,
            approved_for_auto=approved,
        ),
        status_var=SimpleNamespace(set=shown.append),
        close_editor=lambda **_kw: closed.append(True),
        _saved_signature=("old",),
        _signature=lambda: ("edited",),
    )
    assert rew.RouteEditorWindow.save_route(editor) is None
    assert editor._saved_signature == ("old",)
    assert not closed
    assert len(shown) == 1 and shown[0].startswith("航線儲存失敗：")


def test_bound_save_emits_drawn_per_waypoint_radii(tmp_path):
    import route_editor_window as rew

    target = tmp_path / "current_route.json"
    document = rem.RouteDocument(
        site_id="yard",
        coordinate_frame_id="yard-map-v1",
        points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        arrive_radius_map_units=0.02,
        waypoint_arrive_radii=[0.05, 0.02, 0.03],
    )
    imported = []

    def import_route(path):
        imported.append(json.loads(path.read_text(encoding="utf-8")))
        return SimpleNamespace(message="saved", asset_path=target, approved_for_auto=True)

    editor = SimpleNamespace(
        objects=ObjectBoxController(),
        wires=WireController(),
        controller=RouteEditorController(document.points),
        document=document,
        arrive_radius_var=SimpleNamespace(get=lambda: 0.02),
        _preview_only=False,
        import_route=import_route,
        profile=SimpleNamespace(
            route_json=target, asset_sha256=SimpleNamespace(map_ply="a" * 64),
        ),
        _signature=lambda: ("saved",),
        status_var=SimpleNamespace(set=lambda _message: None),
        close_editor=lambda *, force=False: None,
    )
    editor._update_document = lambda: rew.RouteEditorWindow._update_document(editor)
    editor._refresh_distance_labels = lambda: None

    rew.RouteEditorWindow.save_route(editor)
    assert imported[0]["waypoint_arrive_radii"] == [0.05, 0.02, 0.03]


def test_update_document_realigns_radii_after_undo(tmp_path):
    import route_editor_window as rew

    document = rem.RouteDocument(
        site_id="yard",
        coordinate_frame_id="yard-map-v1",
        arrive_radius_map_units=0.02,
        waypoint_arrive_radii=[0.05, 0.02, 0.03, 0.04],
    )
    editor = SimpleNamespace(
        objects=ObjectBoxController(),
        wires=WireController(),
        controller=RouteEditorController([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        document=document,
        arrive_radius_var=SimpleNamespace(get=lambda: 0.02),
    )

    rew.RouteEditorWindow._update_document(editor)
    assert document.waypoint_arrive_radii == [0.05, 0.02]

    document.waypoint_arrive_radii = [0.05]
    editor.controller.points.append([2.0, 0.0, 0.0])
    rew.RouteEditorWindow._update_document(editor)
    assert document.waypoint_arrive_radii == [0.05, 0.02, 0.02]


def test_per_waypoint_spheres_draw_at_own_sizes():
    import route_editor_window as rew

    class View:
        @staticmethod
        def project(_points, _width, _height):
            return np.array([10.0, 110.0]), np.array([20.0, 20.0]), np.zeros(2)

        @staticmethod
        def scale(_width, _height):
            return 100.0

    class Draw:
        def __init__(self):
            self.ellipses = []

        def line(self, *_args, **_kwargs):
            return None

        def ellipse(self, bounds, **kwargs):
            self.ellipses.append((bounds, kwargs))

        def text(self, *_args, **_kwargs):
            return None

    draw = Draw()
    editor = SimpleNamespace(
        controller=SimpleNamespace(
            points=[[0, 0, 0], [1, 0, 0]], selected=None, segment_look_at=[None],
        ),
        document=SimpleNamespace(waypoint_arrive_radii=[0.02, 0.05]),
        view=View(),
        show_arrive_var=SimpleNamespace(get=lambda: True),
        arrive_radius_var=SimpleNamespace(get=lambda: 0.03),
    )
    editor._point_radius_for = lambda index: rew.RouteEditorWindow._point_radius_for(
        editor, index
    )

    rew.RouteEditorWindow._draw_route(editor, draw, 800, 600)

    spheres = [
        item for item in draw.ellipses if item[1].get("fill") == rew._ARRIVE_FILL
    ]
    assert len(spheres) == 2
    assert spheres[0][0] == pytest.approx((8.0, 18.0, 12.0, 22.0))
    assert spheres[1][0] == pytest.approx((105.0, 15.0, 115.0, 25.0))


def test_dragging_the_arrive_limit_turns_on_its_live_visualization():
    import route_editor_window as rew

    class Variable:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    redraws = []
    arrive = SimpleNamespace(
        arrive_radius_var=Variable(0.02),
        arrive_label_var=Variable(""),
        show_arrive_var=Variable(False),
        document=SimpleNamespace(waypoint_arrive_radii=None),
        controller=SimpleNamespace(
            points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            shortest_leg=lambda: 1.0,
            spheres_overlap=lambda _radius: False,
        ),
        request_redraw=lambda: redraws.append("arrive"),
    )
    arrive._set_all_arrive_radii = lambda r: rew.RouteEditorWindow._set_all_arrive_radii(arrive, r)
    rew.RouteEditorWindow._on_arrive_radius(arrive, "0.02")

    assert arrive.show_arrive_var.get() is True
    assert redraws == ["arrive"]


def _radius_editor(radii, points=3):
    import route_editor_window as rew

    class Variable:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    editor = SimpleNamespace(
        arrive_radius_var=Variable(0.02),
        arrive_label_var=Variable(""),
        point_radius_var=Variable("0.020"),
        point_radius_note_var=Variable(""),
        show_arrive_var=Variable(False),
        document=SimpleNamespace(waypoint_arrive_radii=radii),
        controller=SimpleNamespace(
            points=[[float(i), 0.0, 0.0] for i in range(points)],
            selected=None,
            shortest_leg=lambda: 1.0,
            spheres_overlap=lambda _radius: False,
        ),
        request_redraw=lambda: None,
    )
    editor._set_all_arrive_radii = lambda r: rew.RouteEditorWindow._set_all_arrive_radii(editor, r)
    editor._on_arrive_radius = lambda v: rew.RouteEditorWindow._on_arrive_radius(editor, v)
    editor._point_radius_for = lambda i: rew.RouteEditorWindow._point_radius_for(editor, i)
    editor._update_point_radius_note = lambda: None
    editor._sync_point_radius_box = lambda: rew.RouteEditorWindow._sync_point_radius_box(editor)
    return rew, editor


def test_dragging_the_slider_resizes_every_sphere_of_a_route_with_per_point_radii():
    rew, editor = _radius_editor([0.05, 0.04, 0.023])
    editor.arrive_radius_var.set(0.07)

    rew.RouteEditorWindow._on_arrive_radius(editor, "0.07")

    assert editor.document.waypoint_arrive_radii == [0.07, 0.07, 0.07]
    assert [editor._point_radius_for(i) for i in range(3)] == [0.07, 0.07, 0.07]


def test_typing_a_radius_resizes_every_sphere_clamped_to_the_operator_range():
    rew, editor = _radius_editor([0.05, 0.04, 0.023])
    for typed, expected in (("0.03", 0.03), ("0.5", rew.ARRIVE_RADIUS_MAX_U), ("0", rew.ARRIVE_RADIUS_MIN_U)):
        editor.point_radius_var.set(typed)
        rew.RouteEditorWindow._on_point_radius(editor)  # no waypoint needs to be selected
        assert editor.document.waypoint_arrive_radii == [expected] * 3
        assert editor.arrive_radius_var.get() == pytest.approx(expected)

    editor.point_radius_var.set("abc")
    rew.RouteEditorWindow._on_point_radius(editor)
    assert editor.document.waypoint_arrive_radii == [rew.ARRIVE_RADIUS_MIN_U] * 3


def test_a_route_without_per_point_radii_keeps_only_the_global_radius():
    rew, editor = _radius_editor(None)
    editor.arrive_radius_var.set(0.08)
    rew.RouteEditorWindow._on_arrive_radius(editor, "0.08")
    assert editor.document.waypoint_arrive_radii is None
    assert editor.arrive_radius_var.get() == pytest.approx(0.08)


def test_arrival_sphere_draws_a_scaled_translucent_fill():
    import route_editor_window as rew

    class View:
        @staticmethod
        def project(_points, _width, _height):
            return np.array([10.0, 110.0]), np.array([20.0, 20.0]), np.zeros(2)

        @staticmethod
        def scale(_width, _height):
            return 100.0

    class Draw:
        def __init__(self):
            self.ellipses = []

        def line(self, *_args, **_kwargs):
            return None

        def ellipse(self, bounds, **kwargs):
            self.ellipses.append((bounds, kwargs))

        def text(self, *_args, **_kwargs):
            return None

    draw = Draw()
    editor = SimpleNamespace(
        controller=SimpleNamespace(
            points=[[0, 0, 0], [1, 0, 0]], selected=None, segment_look_at=[None],
        ),
        document=SimpleNamespace(waypoint_arrive_radii=None),
        view=View(),
        show_arrive_var=SimpleNamespace(get=lambda: True),
        arrive_radius_var=SimpleNamespace(get=lambda: 0.03),
    )
    editor._point_radius_for = lambda index: rew.RouteEditorWindow._point_radius_for(
        editor, index
    )

    rew.RouteEditorWindow._draw_route(editor, draw, 800, 600)

    bounds, style = next(
        item for item in draw.ellipses if item[1].get("fill") == rew._ARRIVE_FILL
    )
    assert bounds == pytest.approx((7.0, 17.0, 13.0, 23.0))


def test_overlap_predicate_fires_exactly_at_half_the_shortest_leg():
    """The previous version asserted two literals and never called the predicate."""
    import route_editor_window as rew

    controller = rec.RouteEditorController()
    for point in ([0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [3.0, 0.0, 0.0]):
        controller.points.append(list(point))

    assert controller.shortest_leg() == pytest.approx(0.4)
    assert not controller.spheres_overlap(0.199)
    assert controller.spheres_overlap(0.2)      # exactly half the shortest leg
    assert controller.spheres_overlap(0.5)

    # The shipped default must be usable on a route with legs this short.
    assert not controller.spheres_overlap(rew.DEFAULT_ARRIVE_RADIUS_U)


def test_overlap_predicate_is_false_below_two_waypoints():
    controller = rec.RouteEditorController()
    assert controller.spheres_overlap(1.0) is False
    controller.points.append([0.0, 0.0, 0.0])
    assert controller.spheres_overlap(1.0) is False

def test_shortest_leg_is_none_below_two_waypoints():
    controller = rec.RouteEditorController()
    assert controller.shortest_leg() is None
    controller.points.append([0.0, 0.0, 0.0])
    assert controller.shortest_leg() is None


# --- folder discovery -------------------------------------------------------

def _touch(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_discovery_fills_every_asset_slot_from_a_folder(tmp_path):
    _touch(tmp_path, "maps/edm_v1/site_dense.ply")
    _touch(tmp_path, "maps/edm_v1/site_ref_poses.json")
    _touch(tmp_path, "maps/edm_v1/T_align_gravity.json")
    _touch(tmp_path, "bundles/site_reloc_map_edm.pt")
    _touch(tmp_path, "routes/authored/flight_path.json")

    package = lsa.discover_site_package(tmp_path)

    assert package.missing == ()
    assert package.ambiguous == ()
    assert package.has_route
    assert package.asset("map_ply").path.name == "site_dense.ply"
    assert package.asset("map_align").path.name == "T_align_gravity.json"


def test_discovery_reports_a_missing_route_rather_than_guessing(tmp_path):
    _touch(tmp_path, "site.ply")
    _touch(tmp_path, "site_ref_poses.json")

    package = lsa.discover_site_package(tmp_path)

    assert not package.has_route
    assert "route_json" in package.missing
    assert "localization_bundle" in package.missing
    assert any("未找到" in line for line in package.summary_lines())


def test_discovery_never_proposes_a_route_preview_as_the_map(tmp_path):
    """The editor writes flight_path.ply beside the map; it is not a map."""
    _touch(tmp_path, "routes/authored/flight_path.ply")
    real_map = _touch(tmp_path, "maps/site_dense.ply")

    package = lsa.discover_site_package(tmp_path)

    assert package.asset("map_ply").candidates == (real_map,)


def test_discovery_surfaces_ambiguity_instead_of_picking_one(tmp_path):
    _touch(tmp_path, "maps/a_dense.ply")
    _touch(tmp_path, "maps/b_dense.ply")

    package = lsa.discover_site_package(tmp_path)
    slot = package.asset("map_ply")

    assert slot.ambiguous and slot.path is None
    assert len(slot.candidates) == 2
    assert "map_ply" in package.ambiguous


def test_discovery_gives_each_file_to_only_one_slot(tmp_path):
    """`*route*.json` must not re-claim the reference poses or the alignment."""
    _touch(tmp_path, "site.ply")
    # Matches BOTH *ref_poses*.json and the looser *route*.json. The earlier,
    # more specific slot must keep it.
    poses = _touch(tmp_path, "site_route_ref_poses.json")
    _touch(tmp_path, "T_align_gravity.json")
    _touch(tmp_path, "site_reloc_map_edm.pt")

    package = lsa.discover_site_package(tmp_path)
    claimed = [item.path for item in package.assets if item.path is not None]

    assert len(claimed) == len(set(claimed)), "a file filled more than one slot"
    assert package.asset("map_reference_poses").path == poses
    assert not package.has_route, "the reference poses were re-claimed as a route"


def test_discovery_rejects_a_non_folder(tmp_path):
    target = _touch(tmp_path, "not_a_folder.ply")
    with pytest.raises(ValueError, match="not a directory"):
        lsa.discover_site_package(target)


# --- site package import gates ---------------------------------------------

_LENS = ("PINHOLE", 2688, 1512, (1955.5340190972222, 1955.5340190972222, 1344.0, 756.0))
_LENS_720P = ("PINHOLE", 1280, 720, (931.2057783503648, 931.2057783503648, 640.0, 360.0))


def test_same_lens_accepts_the_build_resolution_of_the_same_camera():
    """Maps are built from 2688x1512 stills and flown from a 720p stream."""
    assert lsa._camera_is_same_lens(_LENS, _LENS_720P)
    assert lsa._camera_is_same_lens(_LENS_720P, _LENS_720P)


def test_same_lens_rejects_a_different_camera():
    other = ("PINHOLE", 1280, 720, (1400.0, 1400.0, 640.0, 360.0))
    assert not lsa._camera_is_same_lens(_LENS, other)


def test_same_lens_rejects_a_crop_rather_than_a_scale():
    """A different aspect ratio means cropping; the principal point no longer maps."""
    cropped = ("PINHOLE", 1280, 960, (931.2057783503648, 931.2057783503648, 640.0, 360.0))
    assert not lsa._camera_is_same_lens(_LENS, cropped)


def test_same_lens_rejects_a_different_model():
    assert not lsa._camera_is_same_lens(_LENS, ("RADIAL", 1280, 720, _LENS_720P[3]))


def test_reference_poses_may_be_a_superset_but_never_a_subset(tmp_path):
    """A bundle reference with no pose is a pose computed from nothing."""
    camera = {"model": "PINHOLE", "width": 1280, "height": 720,
              "params": list(_LENS_720P[3])}
    pose = {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0, 0, 0], "camera_id": 1}
    path = tmp_path / "refs.json"

    # Superset: the bundle dropped a frame on quality. Accepted.
    path.write_text(json.dumps({"camera": camera,
                                "poses": {"a.jpg": pose, "b.jpg": pose}}),
                    encoding="utf-8")
    query = lsa.QueryCamera(model="PINHOLE", width=1280, height=720,
                            params=_LENS_720P[3])
    assert lsa._validate_reference_poses(path, query, ["a.jpg"]) == 2

    # Subset: a bundle reference the localizer can match but cannot place.
    with pytest.raises(ValueError, match="no pose"):
        lsa._validate_reference_poses(path, query, ["a.jpg", "ghost.jpg"])


# --- automated route follow-up after a map import ---------------------------

class _StubPanel:
    """The follow-up flow with Tk and the real actions replaced by seams."""

    def __init__(self, answer=True, import_route=None):
        import site_assets_panel as sap

        self.follow_up_route_for_site = sap.SiteAssetsPanel.follow_up_route_for_site.__get__(self)
        self._answers = answer
        self.asked = []
        self.editor_calls = []
        self.status = ""
        self.imported_results: list = []
        self.route_imported = self.imported_results.append
        self.actions = type("A", (), {"import_route": staticmethod(
            import_route or (lambda source: type("R", (), {"message": "ok", "kind": "route"})())
        )})()
        self.status_var = type("V", (), {
            "set": lambda _self, value: setattr(self, "status", value),
            "get": lambda _self: self.status,
        })()

    def ask_yes_no(self, title, message):
        self.asked.append(title)
        return self._answers

    def _open_route_editor(self, edit_current):
        self.editor_calls.append(bool(edit_current))


def _site_folder(tmp_path, *, with_route: bool):
    (tmp_path / "maps").mkdir(parents=True, exist_ok=True)
    (tmp_path / "maps" / "site.ply").write_bytes(b"x")
    if with_route:
        (tmp_path / "routes").mkdir(parents=True, exist_ok=True)
        (tmp_path / "routes" / "flight_path.json").write_bytes(b"x")
    return tmp_path


def test_import_with_an_existing_route_imports_it_and_offers_to_edit(tmp_path):
    panel = _StubPanel(answer=True)

    outcome = panel.follow_up_route_for_site(_site_folder(tmp_path, with_route=True))

    assert outcome == "imported"
    assert panel.editor_calls == [True], "editing an existing route, not drawing a new one"
    assert "已自動匯入航線" in panel.status
    assert len(panel.imported_results) == 1


def test_import_without_a_route_opens_the_drawing_editor(tmp_path):
    panel = _StubPanel(answer=True)

    outcome = panel.follow_up_route_for_site(_site_folder(tmp_path, with_route=False))

    assert outcome == "drawing"
    assert panel.editor_calls == [False], "must open in draw mode, not edit mode"


def test_declining_leaves_the_site_imported_and_says_so(tmp_path):
    panel = _StubPanel(answer=False)

    outcome = panel.follow_up_route_for_site(_site_folder(tmp_path, with_route=False))

    assert outcome == "declined"
    assert panel.editor_calls == []
    assert "尚無航線" in panel.status


def test_two_candidate_routes_are_never_guessed(tmp_path):
    folder = _site_folder(tmp_path, with_route=True)
    (folder / "routes" / "old_route.json").write_bytes(b"x")

    panel = _StubPanel(answer=True)
    outcome = panel.follow_up_route_for_site(folder)

    assert outcome == "ambiguous"
    assert panel.editor_calls == [], "which route the aircraft flies is not a guess"


def test_a_failed_route_import_offers_the_editor_instead(tmp_path):
    def explode(_source):
        raise ValueError("route schema rejected")

    panel = _StubPanel(answer=True, import_route=explode)

    outcome = panel.follow_up_route_for_site(_site_folder(tmp_path, with_route=True))

    assert outcome == "drawing"
    assert panel.editor_calls == [False]


# --- picking which map to import -------------------------------------------

def test_site_package_listing_shows_every_ready_made_package(tmp_path):
    for name, site_id, display in (("alpha", "a_edm", "Alpha"),
                                   ("beta", "b_edm", "Beta")):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "site_profile.json").write_text(
            json.dumps({"site_id": site_id, "display_name": display}), encoding="utf-8")
    (tmp_path / "not_a_package").mkdir()

    packages = lsa.list_site_packages(tmp_path)

    assert [p.site_id for p in packages] == ["a_edm", "b_edm"]
    assert "Alpha (a_edm)" == packages[0].label
    assert all(p.error is None for p in packages)


def test_an_unreadable_package_is_listed_with_its_reason(tmp_path):
    """A site the operator expects to see must say why, not vanish."""
    folder = tmp_path / "broken"
    folder.mkdir()
    (folder / "site_profile.json").write_text("{ not json", encoding="utf-8")

    packages = lsa.list_site_packages(tmp_path)

    assert len(packages) == 1
    assert packages[0].error is not None
    assert "讀取失敗" in packages[0].label


def test_missing_sites_root_lists_nothing_rather_than_raising(tmp_path):
    assert lsa.list_site_packages(tmp_path / "absent") == []


def test_switching_into_a_routeless_site_goes_straight_to_marking(tmp_path):
    """The panel is only a site switch now; the route step must be automatic."""
    panel = _StubPanel(answer=True)
    panel.applied = []
    panel.offer_apply_after_import = lambda: panel.applied.append(True) or False

    outcome = panel.follow_up_route_for_site(_site_folder(tmp_path, with_route=False))

    assert outcome == "drawing"
    assert panel.editor_calls == [False]


def _write_route(path: Path, **overrides) -> Path:
    payload = {
        "schema": "sfm-flight-route/v1",
        "purpose": "flight",
        "align_source": "measured",
        "waypoints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_describe_site_routes_skips_files_that_are_not_routes(tmp_path):
    """routes/ also holds T_align.json and wires.json.

    list_site_routes matches on path shape alone, so offering its output verbatim
    puts a button in front of the operator that can only fail validation.
    """
    routes = tmp_path / "routes"
    _write_route(routes / "good_route.json")
    (routes / "T_align.json").write_text(
        json.dumps({"frame_from": "a", "frame_to": "b", "R": [], "t": []}),
        encoding="utf-8")
    (routes / "wires.json").write_text(json.dumps([[0, 0, 0]]), encoding="utf-8")
    _write_route(routes / "empty_route.json", waypoints=[])
    _write_route(routes / "one_point_route.json", waypoints=[[0.0, 0.0, 0.0]])
    (routes / "broken_route.json").write_text("{not json", encoding="utf-8")

    described = lsa.describe_site_routes(tmp_path)

    assert [item.path.name for item in described] == ["good_route.json"]


def test_describe_site_routes_marks_display_only_and_legacy_routes(tmp_path):
    routes = tmp_path / "routes"
    _write_route(routes / "flight.json")
    _write_route(routes / "preview.json",
                 schema="sfm-route-preview/v1", purpose="preview_only")
    _write_route(routes / "old.json", align_source="legacy")

    by_name = {item.path.name: item for item in lsa.describe_site_routes(tmp_path)}

    assert by_name["flight.json"].flight_ready is True
    assert "僅供顯示" not in by_name["flight.json"].label
    assert by_name["preview.json"].flight_ready is False
    assert "僅供顯示" in by_name["preview.json"].label
    assert "舊對齊" in by_name["old.json"].label
    assert "2 點" in by_name["flight.json"].label


def test_route_choice_candidates_keep_distinct_formal_route_files(tmp_path):
    canonical = _write_route(tmp_path / "routes" / "flight_route.json")
    duplicate = _write_route(
        tmp_path / "routes" / "authored" / "route_20260807_013944.json"
    )
    other = _write_route(
        tmp_path / "routes" / "authored" / "route_20260807_013811.json",
        waypoints=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )

    choices = lsa.describe_site_routes(tmp_path)

    assert duplicate.read_bytes() == canonical.read_bytes()
    assert {item.path for item in choices} == {canonical, duplicate, other}


def test_existing_route_selection_is_refused_until_landed(tmp_path):
    import site_assets_panel as sap

    route = _write_route(tmp_path / "routes" / "flight.json")
    statuses = []
    selected = []
    panel = SimpleNamespace(
        request_route_preview=lambda path: selected.append(path) or "selected",
        flight_state_check=lambda: (False, "飛機尚未確認 landed"),
        status_var=SimpleNamespace(set=statuses.append),
    )

    sap.SiteAssetsPanel._preview_existing_route(panel, route)

    assert selected == []
    assert statuses == ["航線選擇已拒絕：飛機尚未確認 landed"]

    panel.flight_state_check = lambda: (True, "")
    sap.SiteAssetsPanel._preview_existing_route(panel, route)

    assert selected == [route]
    assert statuses[-1] == "selected"


def test_asset_panel_drops_a_second_import_while_the_first_is_running():
    import site_assets_panel as sap

    statuses = []
    actions = []
    panel = SimpleNamespace(
        _busy=True,
        status_var=SimpleNamespace(set=statuses.append),
        _set_busy=lambda _busy: None,
    )

    sap.SiteAssetsPanel._run(
        panel,
        "航線匯入",
        lambda: actions.append("ran"),
    )

    assert actions == []
    assert statuses == ["航線匯入：前一個資產作業尚未完成"]


def test_site_pack_root_comes_from_the_map_not_the_profile_location(tmp_path):
    """A system profile's parent is site_profiles/, not the site it describes.

    Deriving the pack from the profile's own path is what put editor drafts and
    the route picker in a directory holding no routes.
    """
    packages = tmp_path / "場域"
    site = packages / "river_site"
    (site / "maps").mkdir(parents=True)
    ply = site / "maps" / "map.ply"
    ply.write_bytes(b"x")

    elsewhere = tmp_path / "site_profiles"
    elsewhere.mkdir()
    profile_path = elsewhere / "river_site_edm.json"

    import local_site_assets as mod

    class _Profile:
        source = profile_path
        site_id = "river-site"
        map_ply = ply
        coordinate_frame = SimpleNamespace(id="river-frame")
        asset_sha256 = SimpleNamespace(map_ply=mod._sha256(ply))

    original = mod.load_site_profile
    mod.load_site_profile = lambda _path: _Profile()
    try:
        assert mod.site_pack_root_for_profile(profile_path, packages) == site
    finally:
        mod.load_site_profile = original

    assert lsa.site_pack_root_for_profile(None, packages) is None


def test_route_buttons_draw_the_route_and_never_import_it():
    """The operator asked to flip between routes and see each drawn.

    import_route additionally needs a managed profile under the site packages root
    and resets flight approval, so a view change must not go through it.
    """
    import site_assets_panel as sap

    previewed = []
    status = []
    panel = SimpleNamespace(
        actions=SimpleNamespace(
            import_route=lambda src: pytest.fail("preview must not import")),
        request_route_preview=lambda path: previewed.append(path) or "已顯示",
        flight_state_check=lambda: (True, ""),
        status_var=SimpleNamespace(set=status.append),
        _run=lambda *a, **k: pytest.fail("preview must not run an import action"),
    )
    sap.SiteAssetsPanel._preview_existing_route(panel, Path("/tmp/route.json"))

    assert previewed == [Path("/tmp/route.json")]
    assert status == ["已顯示"]


def test_route_button_failure_reports_instead_of_killing_the_callback():
    import site_assets_panel as sap

    status = []

    def explode(_path):
        raise ValueError("align_source 不符")

    panel = SimpleNamespace(
        request_route_preview=explode,
        flight_state_check=lambda: (True, ""),
        status_var=SimpleNamespace(set=status.append),
    )
    sap.SiteAssetsPanel._preview_existing_route(panel, Path("/tmp/bad.json"))

    assert status and "align_source 不符" in status[0]


def test_map_axis_gizmo_uses_the_measured_basis_and_is_resolved_once():
    """The gizmo hard-coded -Y as "UP", which is the assumption, not a measurement.

    river_site's measured up is 5.22 deg off that arrow and urai's is 22.51 deg,
    while the route editor already drew the same cloud through the measured basis.
    """
    import flight_operator_app as app
    from real_path_follow_controller import MapFrame

    basis = app.OperatorApp._map_axis_basis

    # No measured alignment: reproduce the previous arrows exactly.
    legacy = SimpleNamespace(_map_axis_basis_cache=None,
                             _active_site_map_frame=lambda: None,
                             write_log=lambda _m: None)
    east, north, up, measured = basis(legacy)
    assert np.allclose(east, [1, 0, 0])
    assert np.allclose(north, [0, 0, 1])
    assert np.allclose(up, [0, -1, 0])
    assert measured is False

    # Measured alignment: the arrows follow it.
    frame = MapFrame.from_gravity(np.array([0.0, 0.9958562, 0.0908950]))
    site = SimpleNamespace(_map_axis_basis_cache=None,
                           _active_site_map_frame=lambda: frame,
                           write_log=lambda _m: None)
    _, _, measured_up, is_measured = basis(site)
    assert is_measured is True
    assert np.allclose(measured_up, frame.up)
    tilt = np.degrees(np.arccos(np.clip(np.dot(measured_up, [0, -1, 0]), -1, 1)))
    assert tilt > 1.0, "a measured basis that matches the legacy arrow proves nothing"

    # Resolved once: this runs in the redraw loop and reloads the profile on miss.
    calls = []
    cached = SimpleNamespace(_map_axis_basis_cache=None,
                             _active_site_map_frame=lambda: calls.append(1) or None,
                             write_log=lambda _m: None)
    for _ in range(5):
        basis(cached)
    assert len(calls) == 1

    # An unreadable alignment degrades to legacy instead of killing the redraw.
    logged = []
    broken = SimpleNamespace(
        _map_axis_basis_cache=None,
        _active_site_map_frame=lambda: (_ for _ in ()).throw(ValueError("讀不到")),
        write_log=logged.append,
    )
    _, _, fallback_up, fallback_measured = basis(broken)
    assert np.allclose(fallback_up, [0, -1, 0])
    assert fallback_measured is False
    assert logged


def test_arrival_radius_slider_matches_the_operator_range():
    """0.005 to 0.1 map units (operator decisions 2026-08-06, max raised 2026-10-09)."""
    import route_editor_window as rew
    import real_path_follow_controller as rpf_mod

    assert rew.ARRIVE_RADIUS_MIN_U == pytest.approx(0.005)
    assert rew.ARRIVE_RADIUS_MAX_U == pytest.approx(0.1)
    assert rew.ARRIVE_RADIUS_MIN_U <= rew.DEFAULT_ARRIVE_RADIUS_U <= rew.ARRIVE_RADIUS_MAX_U

    # The controller default must sit in the same range, or the editor confirms a
    # radius the flight code will not use.
    flown = rpf_mod.ControlConfig().waypoint_arrive_radius
    assert rew.ARRIVE_RADIUS_MIN_U <= flown <= rew.ARRIVE_RADIUS_MAX_U
    assert flown == pytest.approx(rew.DEFAULT_ARRIVE_RADIUS_U)


def test_a_site_and_its_imported_copy_are_listed_once(tmp_path):
    """Importing copies a package to managed_root/<site_id>, and managed_root is
    the sites directory -- so both live here and describe the same field."""
    for name in ("test_field", "test_field_edm"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "site_profile.json").write_text(
            json.dumps({"site_id": "test_field_edm", "display_name": "測試場域 EDM"}),
            encoding="utf-8")

    packages = lsa.list_site_packages(tmp_path)

    assert len(packages) == 1, [p.folder.name for p in packages]
    # The SOURCE wins, not the copy at managed_root/<site_id>: the copy re-points
    # its bundle at localization/localization_bundle.pt, which carries no trusted
    # digest, so validate_folder refuses it. Listing it gave the operator a button
    # that could not import.
    assert packages[0].folder.name == "test_field"


def test_two_genuinely_different_sites_are_both_listed(tmp_path):
    for name, site_id in (("alpha", "a_edm"), ("beta", "b_edm")):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "site_profile.json").write_text(
            json.dumps({"site_id": site_id, "display_name": name}), encoding="utf-8")

    assert len(lsa.list_site_packages(tmp_path)) == 2


def test_dedup_keeps_the_source_whichever_order_the_folders_scan_in(tmp_path):
    """The copy sorts before the source for some names and after for others."""
    for name in ("zzz_source", "b_edm"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "site_profile.json").write_text(
            json.dumps({"site_id": "b_edm", "display_name": "Beta"}), encoding="utf-8")

    packages = lsa.list_site_packages(tmp_path)

    assert len(packages) == 1
    assert packages[0].folder.name == "zzz_source", "the un-importable copy was kept"




def test_site_routes_are_listed_newest_first(tmp_path):
    """A field accumulates drafts; which one flies is the operator's choice."""
    import os
    import time as _time

    (tmp_path / "routes" / "authored").mkdir(parents=True)
    (tmp_path / "route_drafts").mkdir()
    old = tmp_path / "routes" / "authored" / "flight_path.json"
    new = tmp_path / "route_drafts" / "route_20260806_210000.json"
    old.write_text("{}", encoding="utf-8")
    new.write_text("{}", encoding="utf-8")
    now = _time.time()
    os.utime(old, (now - 3600, now - 3600))
    os.utime(new, (now, now))
    # Must not be mistaken for a route.
    (tmp_path / "site_profile.json").write_text("{}", encoding="utf-8")

    routes = lsa.list_site_routes(tmp_path)

    assert [path.name for path in routes] == [old.name]
    assert all(path.name != "site_profile.json" for path in routes)


def test_route_listing_of_a_missing_folder_is_empty(tmp_path):
    assert lsa.list_site_routes(tmp_path / "absent") == []


RIGHT_30 = {"type": "yaw", "degrees": 30.0}
HOVER_3 = {"type": "hover", "seconds": 3.0}


def test_waypoint_actions_stay_on_their_waypoint_through_delete_and_undo() -> None:
    controller = RouteEditorController()
    controller.set_placement_enabled(True)
    for x in range(3):
        controller.add((float(x), 0.0, 0.0))
    controller.selected = 1
    assert controller.add_action(RIGHT_30)
    controller.selected = 2
    assert controller.add_action(HOVER_3)
    assert controller.actions == [[], [RIGHT_30], [HOVER_3]]

    controller.selected = 1
    assert controller.delete_selected()
    assert controller.actions == [[], [HOVER_3]]
    assert controller.undo()
    assert controller.points[1] == [1.0, 0.0, 0.0]
    assert controller.actions == [[], [RIGHT_30], [HOVER_3]]
    assert controller.redo()
    assert controller.actions == [[], [HOVER_3]]

    controller.selected = 1
    assert controller.remove_last_action() and controller.actions == [[], []]
    assert controller.undo() and controller.actions == [[], [HOVER_3]]


def test_route_file_round_trips_waypoint_actions(tmp_path) -> None:
    document = RouteDocument(
        site_id="yard", coordinate_frame_id="yard-map-v1",
        points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        waypoint_actions=[[RIGHT_30, HOVER_3], []],
    )
    saved = document.save(tmp_path / "route.json")
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["waypoint_actions"] == [[RIGHT_30, HOVER_3], []]
    # The flight loader accepts exactly what the editor wrote.
    route = rpf.RouteDocument.from_path(saved)
    assert route.waypoint_actions[0][0].to_json() == RIGHT_30

    loaded = RouteDocument.load(saved, site_id="yard", coordinate_frame_id="yard-map-v1")
    assert loaded.waypoint_actions == [[RIGHT_30, HOVER_3], []]

    document.waypoint_actions = [[], []]
    assert document.payload()["waypoint_actions"] is None
    payload["waypoint_actions"] = [[{"type": "yaw", "degrees": 0}], []]
    saved.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="degrees"):
        RouteDocument.load(saved, site_id="yard", coordinate_frame_id="yard-map-v1")


def test_bound_save_emits_the_controller_actions(tmp_path):
    import route_editor_window as rew

    document = rem.RouteDocument(
        site_id="yard", coordinate_frame_id="yard-map-v1", arrive_radius_map_units=0.02,
    )
    imported = []

    def import_route(path):
        imported.append(json.loads(path.read_text(encoding="utf-8")))
        return SimpleNamespace(
            message="saved", asset_path=tmp_path / "route.json", approved_for_auto=True,
        )

    editor = SimpleNamespace(
        objects=ObjectBoxController(),
        wires=WireController(),
        controller=RouteEditorController(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[], [RIGHT_30]]
        ),
        document=document,
        arrive_radius_var=SimpleNamespace(get=lambda: 0.02),
        _preview_only=False,
        import_route=import_route,
        profile=SimpleNamespace(
            route_json=tmp_path / "route.json", asset_sha256=SimpleNamespace(map_ply="a" * 64),
        ),
        _signature=lambda: ("saved",),
        status_var=SimpleNamespace(set=lambda _message: None),
        close_editor=lambda *, force=False: None,
    )
    editor._update_document = lambda: rew.RouteEditorWindow._update_document(editor)
    editor._refresh_distance_labels = lambda: None

    rew.RouteEditorWindow.save_route(editor)
    assert imported[0]["waypoint_actions"] == [[], [RIGHT_30]]


def test_full_cloud_visibility_preserves_front_point_and_stable_depth_ties():
    cloud = np.array([
        [0, 0, 1, 255, 0, 0],
        [0, 0, -1, 0, 255, 0],
        [0, 0, 1, 0, 0, 255],
    ], dtype=np.float32)
    renderer = PointCloudRenderer(cloud, use_cuda=False)
    view = OrthoView(np.zeros(3), 1)
    image = renderer.render(view, 100, 100)
    assert image[50, 50].tolist() == [0, 0, 255]
    view.front()
    image = renderer.render(view, 100, 100)
    assert image[4, 50].tolist() == [0, 0, 255]
    assert image[96, 50].tolist() == [0, 255, 0]


def test_full_cloud_draws_a_point_the_old_interaction_stride_would_omit():
    cloud = np.zeros((500_002, 6), dtype=np.float32)
    cloud[:, 0] = 1_000  # Outside the viewport, not a decimated input cloud.
    cloud[-1] = [0, 0, 0, 10, 20, 30]
    renderer = PointCloudRenderer(cloud, use_cuda=False)
    image = renderer.render(OrthoView(np.zeros(3), 1), 100, 100)
    assert image[50, 50].tolist() == [10, 20, 30]
    assert image[0, 0].tolist() == list(BACKGROUND)


def test_full_cloud_clips_points_and_handles_an_empty_cloud():
    renderer = PointCloudRenderer(np.array([
        [100, 0, 0, 255, 0, 0],
        [np.nan, 0, 0, 0, 255, 0],
        [0, -100, 0, 0, 0, 255],
    ], dtype=np.float32), use_cuda=False)
    view = OrthoView(np.zeros(3), 1)
    expected = np.empty((80, 100, 3), dtype=np.uint8)
    expected[:] = BACKGROUND
    np.testing.assert_array_equal(renderer.render(view, 100, 80), expected)
    empty = PointCloudRenderer(np.empty((0, 6)), use_cuda=False)
    np.testing.assert_array_equal(empty.render(view, 100, 80), expected)


def test_renderer_picks_the_frontmost_point_within_the_pixel_radius():
    cloud = np.array([
        [0, 0, -1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [3, 0, 0, 0, 0, 0],
        [np.nan, 0, 0, 0, 0, 0],
    ], dtype=np.float32)
    renderer = PointCloudRenderer(cloud, use_cuda=False)
    view = OrthoView(np.zeros(3), 1)  # 100 px canvas: 46 px per unit, origin at (50, 50)
    assert renderer.nearest(view, 100, 100, 50, 50, 5) == 1
    assert renderer.nearest(view, 100, 100, 50 + 3 * 46, 50, 5) == 2
    assert renderer.nearest(view, 100, 100, 10, 10, 5) is None
    assert PointCloudRenderer(np.empty((0, 6)), use_cuda=False).nearest(
        view, 100, 100, 50, 50, 5
    ) is None


def test_cuda_picking_matches_the_cpu_result_on_a_rotated_view():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(7)
    cloud = np.concatenate(
        [rng.normal(size=(20_000, 3)) * 5 + 100, np.zeros((20_000, 3))], axis=1
    ).astype(np.float32)
    view = OrthoView(np.array([100.0, 100.0, 100.0]), 12.0, yaw=0.7, pitch=1.1, pan_x=9.0)
    gpu = PointCloudRenderer(cloud)
    cpu = PointCloudRenderer(cloud, use_cuda=False)
    assert gpu.device == "cuda"
    hits = 0
    for x, y in rng.uniform(100, 700, size=(60, 2)):
        expected = cpu.nearest(view, 800, 800, x, y, 20)
        assert gpu.nearest(view, 800, 800, x, y, 20) == expected
        hits += expected is not None
    assert hits > 10


def test_point_picking_uses_displayed_view_while_latest_rotation_is_pending():
    import route_editor_window as rew

    displayed = OrthoView(np.zeros(3), 1)
    latest = OrthoView(np.zeros(3), 1, yaw=np.pi / 2)
    cloud = np.array([[0.5, 0, 0, 255, 0, 0]], dtype=np.float32)
    editor = SimpleNamespace(
        map_points=cloud, view=latest, _displayed_view=displayed,
        _displayed_size=(100, 100), _displayed_cloud=cloud,
        controller=RouteEditorController([[0.5, 0, 0]]),
    )
    point = rew.RouteEditorWindow._nearest_cloud(editor, 73, 50, threshold=1)
    np.testing.assert_array_equal(point, [0.5, 0, 0])
    assert rew.RouteEditorWindow._nearest_waypoint(editor, 73, 50, threshold=1) == 0
    editor._displayed_cloud = None  # Incoming replacement map is not on-screen yet.
    assert rew.RouteEditorWindow._nearest_cloud(editor, 73, 50) is None


def test_camera_pitch_editor_save_reload_undo_redo_and_canvas_marker(tmp_path):
    import route_editor_window as rew
    from operator_autonomy_status import waypoint_action_text

    points = [[0., 0., 0.], [1., 0., 0.]]
    controller = RouteEditorController(points)
    controller.selected = 0
    for pitch in (-90, 0, 90):
        action = {"type": "gimbal_pitch", "pitch_deg": float(pitch)}
        controller.add_action(action)
    assert controller.undo() and len(controller.actions[0]) == 2
    assert controller.redo() and len(controller.actions[0]) == 3
    document = RouteDocument(
        site_id="yard", coordinate_frame_id="yard-map-v1",
        points=controller.points, waypoint_actions=controller.actions,
    )
    saved = document.save(tmp_path / "camera_route.json")
    loaded = RouteDocument.load(saved, site_id="yard", coordinate_frame_id="yard-map-v1")
    assert loaded.waypoint_actions == controller.actions
    flight = rpf.RouteDocument.from_path(saved)
    assert [action.value for action in flight.waypoint_actions[0]] == [-90, 0, 90]
    assert rew._action_marker(controller.actions[0][0]) == "C-90"
    assert rew._action_marker(controller.actions[0][1]) == "C+0"
    assert "鏡頭俯仰+90°" in waypoint_action_text(controller.actions[0][2])


@pytest.mark.parametrize("value,accepted", [("0", True), ("-90", True), ("90", True),
                                           ("91", False), ("nan", False), ("bad", False)])
def test_camera_pitch_widget_uses_the_validated_action_contract(value, accepted):
    import route_editor_window as rew

    controller = RouteEditorController([[0., 0., 0.], [1., 0., 0.]])
    controller.selected = 0
    messages = []
    editor = SimpleNamespace(
        controller=controller,
        action_pitch_var=SimpleNamespace(get=lambda: value),
        action_direction_var=SimpleNamespace(get=lambda: "out"),
        status_var=SimpleNamespace(set=messages.append),
        _refresh_labels=lambda: None, request_redraw=lambda: None,
    )
    editor._add_action = lambda raw: rew.RouteEditorWindow._add_action(editor, raw)
    rew.RouteEditorWindow._add_gimbal_pitch_action(editor)
    assert bool(controller.actions[0]) is accepted
    assert messages


def test_outbound_and_return_actions_are_independent_and_follow_their_waypoint() -> None:
    controller = RouteEditorController()
    controller.set_placement_enabled(True)
    for x in range(4):
        controller.add([float(x), 0.0, 0.0])
    controller.selected = 1
    assert controller.add_action(RIGHT_30, "out")
    assert controller.add_action(HOVER_3, "back")
    assert controller.actions[1] == [RIGHT_30]
    assert controller.return_actions[1] == [HOVER_3]

    # Undo/redo restore both lists together; deleting an earlier point shifts both.
    controller.selected = 0
    controller.delete_selected()
    assert controller.actions[0] == [RIGHT_30] and controller.return_actions[0] == [HOVER_3]
    assert controller.undo()
    assert controller.actions[1] == [RIGHT_30] and controller.return_actions[1] == [HOVER_3]
    assert controller.redo()
    assert controller.return_actions[0] == [HOVER_3]

    controller.selected = 0
    assert controller.remove_last_action("back")
    assert controller.return_actions[0] == [] and controller.actions[0] == [RIGHT_30]


def test_return_actions_are_refused_on_the_landing_and_turnaround_points() -> None:
    controller = RouteEditorController([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    for index, allowed in ((0, False), (1, True), (2, False)):
        controller.selected = index
        assert controller.return_actions_allowed(index) is allowed
        assert controller.add_action(HOVER_3, "back") is allowed
        # Outbound actions are unaffected by that rule.
        assert controller.add_action(HOVER_3, "out")
    with pytest.raises(ValueError):
        controller.add_action(HOVER_3, "sideways")


def test_route_file_round_trips_return_actions_and_rejects_illegal_ones(tmp_path) -> None:
    document = RouteDocument(
        site_id="yard", coordinate_frame_id="yard-map-v1",
        points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        waypoint_actions=[[], [RIGHT_30], []],
        return_waypoint_actions=[[], [HOVER_3], []],
    )
    saved = document.save(tmp_path / "route.json")
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["waypoint_actions"] == [[], [RIGHT_30], []]
    assert payload["return_waypoint_actions"] == [[], [HOVER_3], []]
    flight = rpf.RouteDocument.from_path(saved)
    assert flight.return_waypoint_actions[1][0].to_json() == HOVER_3
    loaded = RouteDocument.load(saved, site_id="yard", coordinate_frame_id="yard-map-v1")
    assert loaded.return_waypoint_actions == [[], [HOVER_3], []]

    document.return_waypoint_actions = [[], [], []]
    assert document.payload()["return_waypoint_actions"] is None
    for bad_index in (0, 2):
        entries = [[], [], []]
        entries[bad_index] = [HOVER_3]
        document.return_waypoint_actions = entries
        with pytest.raises(ValueError, match="回程動作不能設在"):
            document.payload()
        payload["return_waypoint_actions"] = entries
        saved.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            rpf.RouteDocument.from_path(saved)


def test_editor_adds_to_the_selected_trip_only() -> None:
    import route_editor_window as rew

    controller = RouteEditorController([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    messages = []
    direction = {"value": "back"}
    editor = SimpleNamespace(
        controller=controller,
        action_direction_var=SimpleNamespace(get=lambda: direction["value"]),
        status_var=SimpleNamespace(set=messages.append),
        _refresh_labels=lambda: None, request_redraw=lambda: None,
    )
    controller.selected = 0
    rew.RouteEditorWindow._add_action(editor, HOVER_3)
    assert controller.return_actions[0] == [] and "不能設定回程動作" in messages[-1]
    controller.selected = 1
    rew.RouteEditorWindow._add_action(editor, HOVER_3)
    assert controller.return_actions[1] == [HOVER_3] and controller.actions[1] == []
    assert "回程動作" in messages[-1]
    direction["value"] = "out"
    rew.RouteEditorWindow._add_action(editor, RIGHT_30)
    assert controller.actions[1] == [RIGHT_30] and controller.return_actions[1] == [HOVER_3]


class _DeferredCloudExecutor:
    def __init__(self):
        self.calls = []
        self.shutdown_calls = []

    def submit(self, function, *args):
        from concurrent.futures import Future

        future = Future()
        self.calls.append((function, args, future))
        return future

    def shutdown(self, *, wait, cancel_futures):
        self.shutdown_calls.append((wait, cancel_futures))
        if cancel_futures:
            for _, _, future in self.calls:
                future.cancel()

    def complete(self):
        function, args, future = self.calls[-1]
        future.set_result(function(*args))


class _RenderEditor(SimpleNamespace):
    pass


def _render_editor(monkeypatch):
    from types import MethodType
    import route_editor_window as rew
    import route_editor_rendering as rendering

    monkeypatch.setattr(rendering, "PointCloudRenderer",
                        lambda points: PointCloudRenderer(points, use_cuda=False))
    monkeypatch.setattr(rew.ImageTk, "PhotoImage", lambda image: image.copy())
    editor = _RenderEditor(
        map_points=np.array([
            [-0.8, 0.4, 0.1, 230, 20, 30],
            [0.5, -0.7, 0.2, 20, 230, 30],
            [0, 0, 0.6, 20, 30, 230],
        ], dtype=np.float32),
        view=OrthoView(np.zeros(3), 1),
        controller=RouteEditorController([
            [-0.4, -0.3, 0], [0.2, 0.1, 0], [0.6, 0.3, 0],
        ]),
        objects=ObjectBoxController(),
        wires=WireController(),
        _wiring=False,
        measurement_points=[],
        _measurement_selected=None,
        _boxing=False,
        _show_arrive=False,
        _cloud_revision=0,
        _cloud_worker=PointCloudRenderWorker(),
        _raster_cache_key=None,
        _raster_cache=None,
        _displayed_view=None,
        _displayed_cloud=None,
        _displayed_size=None,
        _distance_labels_dirty=False,
        _closed=False,
        _redraw_pending=False,
        _render_future=None,
        _render_after=None,
        _render_executor=_DeferredCloudExecutor(),
        _canvas_item=None,
        _photo=None,
        size=(320, 240),
        callbacks=[],
        cancelled_callbacks=[],
        destroyed=False,
        on_close_callback=None,
        status_var=SimpleNamespace(set=lambda value: pytest.fail(value)),
        canvas=SimpleNamespace(
            create_image=lambda *args, **kwargs: 1,
            itemconfigure=lambda *args, **kwargs: None,
        ),
    )
    editor._canvas_size = lambda: editor.size
    editor.show_arrive_var = SimpleNamespace(get=lambda: editor._show_arrive)
    editor._point_radius_for = lambda index: 0.05
    editor.after = lambda delay, callback: editor.callbacks.append(callback) or "timer"
    editor.after_idle = lambda callback: editor.after(0, callback)
    editor.after_cancel = editor.cancelled_callbacks.append
    editor.attributes = lambda *args: None
    editor.destroy = lambda: setattr(editor, "destroyed", True)
    for name in (
        "request_redraw", "_redraw", "_poll_render", "close_editor",
        "_current_raster_key", "_paint_render", "_draw_axes", "_draw_route",
        "_draw_measurement", "_draw_object_boxes", "_draw_wires",
    ):
        setattr(editor, name, MethodType(getattr(rew.RouteEditorWindow, name), editor))
    return editor


def _finish_cloud_frame(editor):
    editor.callbacks.pop(0)()
    editor._render_executor.complete()
    editor.callbacks.pop(0)()


def _expected_editor_image(editor):
    from PIL import Image, ImageDraw

    width, height = editor.size
    raster = PointCloudRenderer(editor.map_points, use_cuda=False).render(
        editor.view, width, height
    )
    image = Image.fromarray(raster)
    draw = ImageDraw.Draw(image, "RGBA")
    for method in (
        editor._draw_axes, editor._draw_route,
        editor._draw_measurement, editor._draw_object_boxes,
    ):
        method(draw, width, height, editor.view)
    return np.asarray(image)


def test_fixed_camera_overlay_edits_reuse_pristine_cloud_pixels(monkeypatch):
    editor = _render_editor(monkeypatch)
    editor.request_redraw()
    _finish_cloud_frame(editor)
    pristine = np.asarray(editor._raster_cache).copy()
    initial = np.asarray(editor._photo).copy()
    editor.controller.selected = 1
    editor.controller.add_action({"type": "hover", "seconds": 2})
    editor._show_arrive = True
    editor.measurement_points = [[-0.6, 0.6, 0], [0.7, -0.6, 0]]
    editor.objects = ObjectBoxController([
        {"center": [0.2, 0.2, 0], "size": [0.4, 0.5, 0.3]},
    ])
    editor.request_redraw()
    editor.callbacks.pop(0)()
    assert len(editor._render_executor.calls) == 1
    assert not np.array_equal(initial, np.asarray(editor._photo))
    np.testing.assert_array_equal(np.asarray(editor._photo), _expected_editor_image(editor))
    np.testing.assert_array_equal(np.asarray(editor._raster_cache), pristine)
    editor.controller = RouteEditorController([
        [-0.4, -0.3, 0], [0.2, 0.1, 0], [0.6, 0.3, 0],
    ])
    editor._show_arrive = False
    editor.measurement_points.clear()
    editor.objects = ObjectBoxController()
    editor.request_redraw()
    editor.callbacks.pop(0)()
    assert len(editor._render_executor.calls) == 1
    np.testing.assert_array_equal(np.asarray(editor._photo), initial)


@pytest.mark.parametrize("mutation", [
    "radius", "center", "yaw", "pitch", "zoom", "pan_x", "pan_y",
    "width", "height", "cloud_identity", "cloud_revision",
])
def test_cloud_cache_invalidation_matches_fresh_raster(monkeypatch, mutation):
    editor = _render_editor(monkeypatch)
    editor.request_redraw()
    _finish_cloud_frame(editor)
    if mutation == "center":
        editor.view.center[0] += 0.15
    elif mutation in {"width", "height"}:
        width, height = editor.size
        editor.size = (width + 13, height) if mutation == "width" else (width, height + 11)
    elif mutation == "cloud_identity":
        editor.map_points = editor.map_points.copy()
        editor.map_points[:, 3:6] = [70, 80, 90]
    elif mutation == "cloud_revision":
        editor.map_points[:, 3:6] = [90, 80, 70]
        editor._cloud_revision += 1
    else:
        setattr(editor.view, mutation, getattr(editor.view, mutation) + 0.2)
    editor.request_redraw()
    _finish_cloud_frame(editor)
    assert len(editor._render_executor.calls) == 2
    width, height = editor.size
    np.testing.assert_array_equal(
        np.asarray(editor._raster_cache),
        PointCloudRenderer(editor.map_points, use_cuda=False).render(editor.view, width, height),
    )
    np.testing.assert_array_equal(np.asarray(editor._photo), _expected_editor_image(editor))


@pytest.mark.parametrize("reload_cloud", [False, True])
def test_pending_cloud_completion_never_displays_an_obsolete_camera(monkeypatch, reload_cloud):
    editor = _render_editor(monkeypatch)
    editor.request_redraw()
    editor.callbacks.pop(0)()
    for offset in range(1, 40):
        editor.view.center[0] = offset / 100
        editor.view.radius = 1 + offset / 100
        editor.view.yaw = offset / 200
        editor.size = (320 + offset, 240)
        if reload_cloud and offset == 20:
            editor.map_points = editor.map_points.copy()
            editor.map_points[:, 3:6] = [220, 150, 70]
            editor._cloud_revision += 1
        editor.request_redraw()
    assert len(editor._render_executor.calls) == 1
    editor._render_executor.complete()
    editor.callbacks.pop(0)()
    assert editor._photo is None
    assert editor._raster_cache is None
    assert len(editor.callbacks) == 1
    _finish_cloud_frame(editor)
    assert len(editor._render_executor.calls) == 2
    assert not editor.callbacks
    np.testing.assert_array_equal(np.asarray(editor._photo), _expected_editor_image(editor))
    np.testing.assert_array_equal(editor._displayed_view.center, editor.view.center)
    assert editor._displayed_cloud is editor.map_points


def test_overlay_request_during_cloud_render_does_not_submit_second_cloud(monkeypatch):
    editor = _render_editor(monkeypatch)
    editor.request_redraw()
    editor.callbacks.pop(0)()
    editor.controller.selected = 1
    editor.measurement_points = [[0, 0.5, 0]]
    editor.request_redraw()
    editor._render_executor.complete()
    editor.callbacks.pop(0)()
    editor.callbacks.pop(0)()
    assert len(editor._render_executor.calls) == 1
    np.testing.assert_array_equal(np.asarray(editor._photo), _expected_editor_image(editor))


@pytest.mark.parametrize("work_state", ["scheduled", "queued", "running", "completed"])
def test_close_rejects_pending_render_callbacks_without_changing_pixels(monkeypatch, work_state):
    editor = _render_editor(monkeypatch)
    editor.request_redraw()
    _finish_cloud_frame(editor)
    pristine = np.asarray(editor._raster_cache).copy()
    original_raster = editor._raster_cache
    original_photo = editor._photo
    editor.view.pan_x += 12
    editor.request_redraw()
    future = None
    if work_state != "scheduled":
        editor.callbacks.pop(0)()
        future = editor._render_future
        if work_state in {"running", "completed"}:
            assert future.set_running_or_notify_cancel()
        if work_state == "completed":
            editor._render_executor.complete()
    stale_callbacks = editor.callbacks[:]
    editor.callbacks.clear()

    editor.close_editor(force=True)

    assert editor.destroyed
    assert editor.cancelled_callbacks == ["timer"]
    assert editor._render_executor.shutdown_calls == [(False, True)]
    if work_state == "running":
        editor._render_executor.complete()
    elif work_state == "queued":
        assert future.cancelled()
    for callback in stale_callbacks:
        callback()
    editor.request_redraw()
    editor._redraw()
    assert not editor.callbacks
    assert editor._render_after is None
    assert editor._raster_cache is None
    assert editor._raster_cache_key is None
    assert editor._photo is original_photo
    np.testing.assert_array_equal(np.asarray(original_raster), pristine)


def test_running_cloud_job_does_not_retain_closed_editor(monkeypatch):
    import gc
    import threading
    import weakref
    from concurrent.futures import ThreadPoolExecutor

    started = threading.Event()
    release = threading.Event()
    original_render = PointCloudRenderWorker.render

    def blocked_render(worker, *args):
        started.set()
        if not release.wait(5):
            raise AssertionError("test did not release the cloud worker")
        return original_render(worker, *args)

    monkeypatch.setattr(PointCloudRenderWorker, "render", blocked_render)
    editor = _render_editor(monkeypatch)
    executor = ThreadPoolExecutor(max_workers=1)
    editor._render_executor = executor
    try:
        editor.request_redraw()
        editor.callbacks.pop(0)()
        future = editor._render_future
        assert started.wait(5)
        owner = weakref.ref(editor)
        editor.close_editor(force=True)
        assert not release.is_set()
        assert not future.done()
        del editor
        # Collect the fake UI's main-thread callback cycles while the real job
        # remains blocked. Its WorkItem may own only cloud data, not the UI.
        gc.collect()
        assert owner() is None
        release.set()
        assert future.result(timeout=5).shape == (240, 320, 3)
    finally:
        release.set()
        executor.shutdown(wait=True)


@pytest.fixture
def route_editor_gui(monkeypatch):
    """Real Tk layout, with only asynchronous raster work removed from geometry checks."""
    import tkinter as tk
    import route_editor_window as rew

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display required: {exc}")
    root.withdraw()
    callback_failures = []
    root.report_callback_exception = lambda _kind, value, _trace: callback_failures.append(value)
    monkeypatch.setattr(rew.RouteEditorWindow, "request_redraw", lambda self: None)
    editor = rew.RouteEditorWindow(
        root, profile=None,
        map_points=np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]]),
        map_source=None, route_path=None,
        import_route=lambda _path: pytest.fail("preview must not import a flight route"),
        discover_map_ply=lambda _path: [], load_map_ply=lambda _path: (None, [], ""),
        safety_check=lambda: (True, ""), distance_scale_loader=lambda _profile: None,
        test_route=lambda _path: pytest.fail("preview must not start simulation"),
    )
    try:
        yield root, editor
    finally:
        if editor.winfo_exists():
            editor.close_editor(force=True)
        root.destroy()
    assert not callback_failures


@pytest.mark.parametrize("size", [(1040, 640), (1280, 720), (1440, 900)])
def test_editor_reserves_map_and_save_exit_at_supported_sizes(route_editor_gui, size):
    from tkinter import ttk

    root, editor = route_editor_gui
    editor.geometry(f"{size[0]}x{size[1]}+0+0")
    root.update_idletasks()
    assert not editor.attributes("-fullscreen")
    assert editor.canvas.winfo_width() >= 600
    assert editor.canvas.winfo_height() >= 300
    assert "不能供 AUTO" in editor.map_status_var.get()
    assert str(editor.test_button["state"]) == "disabled"
    for button in (editor.save_button, editor.exit_button, editor.finish_button, editor.ply_button):
        assert button.winfo_ismapped()
        assert button.winfo_width() >= button.winfo_reqwidth()
        assert button.winfo_rootx() >= editor.winfo_rootx()
        assert button.winfo_rootx() + button.winfo_width() <= editor.winfo_rootx() + editor.winfo_width()
        assert button.winfo_rooty() + button.winfo_height() <= editor.winfo_rooty() + editor.winfo_height()

    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)

    for index in range(4):
        editor.side_tabs.select(index)
        root.update_idletasks()
        canvas = editor._tab_canvases[index]
        assert canvas.winfo_height() >= 100
        for widget in descendants(canvas):
            if isinstance(widget, (ttk.Button, ttk.Spinbox, ttk.Combobox)):
                assert widget.winfo_width() >= widget.winfo_reqwidth()
                assert widget.winfo_rootx() >= canvas.winfo_rootx()
                assert widget.winfo_rootx() + widget.winfo_width() <= canvas.winfo_rootx() + canvas.winfo_width()
        canvas.yview_moveto(1.0)
        root.update_idletasks()
        assert canvas.yview()[1] == pytest.approx(1.0)
        assert editor.save_button.winfo_ismapped() and editor.exit_button.winfo_ismapped()

    editor.status_var.set("儲存失敗：" + "很長的錯誤路徑／" * 150)
    root.update_idletasks()
    # Tk computes wrapped Text line heights asynchronously, outside idle layout.
    editor.status_text.count("1.0", "end", "update", "ypixels")
    assert editor.status_text.yview()[1] < 1.0
    editor.status_text.yview_moveto(1.0)
    root.update_idletasks()
    assert editor.status_text.yview()[0] > 0
    assert editor.status_text.yview()[1] == pytest.approx(1.0)
    assert editor.canvas.winfo_height() >= 300
    assert editor.exit_button.winfo_rooty() + editor.exit_button.winfo_height() <= editor.winfo_rooty() + editor.winfo_height()


def test_editor_sidebar_focus_scroll_and_keys_do_not_operate_map_or_parent(route_editor_gui):
    root, editor = route_editor_gui
    editor.geometry("1040x640+0+0")
    editor.marking_mode_button.invoke()
    for point in ([0., 0., 0.], [1., 0., 0.], [2., 0., 0.]):
        editor.controller.add(point)
    editor.finish_button.invoke()
    editor.controller.selected = 1
    editor._refresh_labels()
    editor.action_direction_var.set("back")
    editor._add_hover_action()
    assert editor.controller.return_actions[1] == [{"type": "hover", "seconds": 3.0}]
    assert editor.controller.actions[1] == []
    root.update_idletasks()

    global_bindings = {seq: root.bind_all(seq) for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>")}
    parent_keys = []
    root.bind("<KeyPress-g>", lambda _event: parent_keys.append("g"))
    editor.side_tabs.select(1)
    root.update_idletasks()
    editor.look_at_clear_button.focus_force()
    root.update()
    canvas = editor._tab_canvases[1]
    top = editor.look_at_clear_button.winfo_rooty() - canvas.winfo_rooty()
    assert top >= 0
    assert top + editor.look_at_clear_button.winfo_height() <= canvas.winfo_height()
    before = (editor.view.zoom, editor.view.pan_x, editor.view.pan_y)
    editor.look_at_clear_button.event_generate("<Button-4>")
    editor.look_at_clear_button.event_generate("<KeyPress-g>")
    editor.look_at_clear_button.event_generate("<KeyPress-Delete>")
    root.update()
    assert (editor.view.zoom, editor.view.pan_x, editor.view.pan_y) == before
    assert len(editor.controller.points) == 3
    assert not editor.controller.moving and not parent_keys
    assert {seq: root.bind_all(seq) for seq in global_bindings} == global_bindings

    editor.canvas.focus_force()
    root.update()
    editor.canvas.event_generate("<KeyPress-g>")
    root.update()
    assert editor.controller.moving
    editor.canvas.event_generate("<KeyPress-Escape>")
    root.update()
    assert not editor.controller.moving
    assert not editor.attributes("-fullscreen")
