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
from route_editor_model import (
    PREVIEW_ROUTE_SCHEMA,
    ROUTE_SCHEMA,
    RouteDocument,
    editor_to_glomap,
    glomap_to_editor,
)


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
    assert "unapproved" in payload["note"]


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


def test_site_shortcut_labels_are_the_short_field_name():
    """The button says 烏來, not the full audit-trail display name."""
    import site_assets_panel as sap

    short = sap.SiteAssetsPanel._short_site_name
    assert short("烏來（目標場域）EDM v1 — 2026-07-19 驗證") == "烏來"
    assert short("河濱場域 EDM") == "河濱場域"
    assert short("足球場 EDM") == "足球場"
    # No separator: keep it whole rather than truncating arbitrarily.
    assert short("Alpha") == "Alpha"


def test_switching_into_a_routeless_site_goes_straight_to_marking(tmp_path):
    """The panel is only a site switch now; the route step must be automatic."""
    panel = _StubPanel(answer=True)
    panel.applied = []
    panel.offer_apply_after_import = lambda: panel.applied.append(True) or False

    outcome = panel.follow_up_route_for_site(_site_folder(tmp_path, with_route=False))

    assert outcome == "drawing"
    assert panel.editor_calls == [False]


def test_applying_never_restarts_out_from_under_the_route_editor():
    """Applying restarts the interface; doing that mid-drawing loses the work.

    Driven through the real _finish_ok rather than asserting on its source text,
    which would still pass if the guard were reworded into something inert.
    """
    import site_assets_panel as sap

    for outcome, expect_apply in (
        ("drawing", False), ("imported", True),
        ("declined", True), ("ambiguous", True),
    ):
        panel = _StubPanel(answer=True)
        panel._set_busy = lambda _busy: None
        panel._pending_site_folder = Path("/tmp/whatever")
        panel.follow_up_route_for_site = lambda _folder, o=outcome: o
        applied = []
        panel.offer_apply_after_import = lambda: applied.append(True) or False
        result = type("R", (), {"kind": "site", "message": "ok"})()

        sap.SiteAssetsPanel._finish_ok(panel, result)

        assert bool(applied) is expect_apply, (
            f"outcome={outcome}: apply {'ran' if applied else 'did not run'}"
        )

def test_choose_route_opens_the_site_folder_of_the_current_profile(tmp_path, monkeypatch):
    """current_profile is a Path to site_profile.json, never a SiteProfile.

    Reading .source off it raised AttributeError inside the Tk callback, so
    「選擇航線」 did nothing at all and the dialog never opened.
    """
    import site_assets_panel as sap

    site = tmp_path / "river_site"
    (site / "routes" / "river_site_safezone").mkdir(parents=True)
    route = site / "routes" / "river_site_safezone" / "flight_path.json"
    route.write_text("{}", encoding="utf-8")
    profile_path = site / "site_profile.json"
    profile_path.write_text("{}", encoding="utf-8")

    opened: dict[str, str | None] = {}

    def fake_askopenfilename(**kwargs):
        opened["initialdir"] = kwargs.get("initialdir")
        return ""

    monkeypatch.setattr(sap.filedialog, "askopenfilename", fake_askopenfilename)

    panel = SimpleNamespace(
        actions=SimpleNamespace(current_profile=profile_path),
        # No pack root known: this is the fallback that must still work.
        site_pack_root=None,
        _run=lambda *a, **k: None,
    )
    sap.SiteAssetsPanel._choose_route(panel)

    assert opened["initialdir"] == str(route.parent)


def test_choose_route_never_imports_while_airborne(monkeypatch):
    import site_assets_panel as sap

    statuses = []
    opened = []
    imported = []
    monkeypatch.setattr(
        sap.filedialog,
        "askopenfilename",
        lambda **_kwargs: opened.append(True) or "/tmp/route.json",
    )

    panel = SimpleNamespace(
        flight_state_check=lambda: (False, "飛機尚未確認 landed"),
        status_var=SimpleNamespace(set=statuses.append),
        actions=SimpleNamespace(
            current_profile=None,
            import_route=lambda source: imported.append(source),
        ),
    )

    sap.SiteAssetsPanel._choose_route(panel)

    assert opened == []
    assert imported == []
    assert statuses == ["航線匯入已拒絕：飛機尚未確認 landed"]


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


def test_route_choice_candidates_dedupe_content_and_keep_canonical_name(tmp_path):
    import site_assets_panel as sap

    canonical = _write_route(tmp_path / "routes" / "flight_route.json")
    duplicate = _write_route(
        tmp_path / "routes" / "authored" / "route_20260807_013944.json"
    )
    other = _write_route(
        tmp_path / "routes" / "authored" / "route_20260807_013811.json",
        waypoints=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )

    choices = sap.SiteAssetsPanel._dedupe_route_choices(
        list(reversed(lsa.describe_site_routes(tmp_path)))
    )

    assert duplicate.read_bytes() == canonical.read_bytes()
    assert [item.path for item in choices] == [canonical, other]
    assert choices[0].label.startswith("flight_route.json")


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


def test_route_picker_opens_the_site_pack_when_the_profile_lives_elsewhere(
    tmp_path, monkeypatch
):
    """site_pack_root wins over current_profile.parent for the dialog's start dir."""
    import site_assets_panel as sap

    site = tmp_path / "river_site"
    route = site / "routes" / "authored" / "route.json"
    _write_route(route)
    elsewhere = tmp_path / "site_profiles"
    elsewhere.mkdir()
    (elsewhere / "river_site_edm.json").write_text("{}", encoding="utf-8")

    opened: dict[str, str | None] = {}
    monkeypatch.setattr(
        sap.filedialog, "askopenfilename",
        lambda **kw: opened.setdefault("initialdir", kw.get("initialdir")) and "")

    panel = SimpleNamespace(
        actions=SimpleNamespace(current_profile=elsewhere / "river_site_edm.json"),
        site_pack_root=site,
        _run=lambda *a, **k: None,
    )
    sap.SiteAssetsPanel._choose_route(panel)

    assert opened["initialdir"] == str(route.parent)


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


def test_site_switch_hides_the_site_that_is_already_loaded(tmp_path):
    """Switching to the active site re-imports and re-execs back to where it was."""
    import site_assets_panel as sap

    active = tmp_path / "river_site"
    other = tmp_path / "football_field"
    active.mkdir()
    other.mkdir()
    listed = [
        SimpleNamespace(folder=active, display_name="河濱場域 EDM"),
        SimpleNamespace(folder=other, display_name="足球場 EDM"),
    ]
    panel = SimpleNamespace(
        site_pack_root=active,
        _short_site_name=sap.SiteAssetsPanel._short_site_name,
    )
    panel._is_active_site = sap.SiteAssetsPanel._is_active_site.__get__(panel)
    panel._active_site_label = sap.SiteAssetsPanel._active_site_label.__get__(panel)

    assert panel._is_active_site(listed[0]) is True
    assert panel._is_active_site(listed[1]) is False
    assert panel._active_site_label(listed) == "河濱場域"

    # Without a known active site nothing is hidden.
    panel.site_pack_root = None
    assert [item for item in listed if panel._is_active_site(item)] == []
    assert panel._active_site_label(listed) == ""


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
    """0.005 to 0.05 map units (operator decision 2026-08-06)."""
    import route_editor_window as rew
    import real_path_follow_controller as rpf_mod

    assert rew.ARRIVE_RADIUS_MIN_U == pytest.approx(0.005)
    assert rew.ARRIVE_RADIUS_MAX_U == pytest.approx(0.05)
    assert rew.ARRIVE_RADIUS_MIN_U <= rew.DEFAULT_ARRIVE_RADIUS_U <= rew.ARRIVE_RADIUS_MAX_U

    # The controller default must sit in the same range, or the editor confirms a
    # radius the flight code will not use.
    flown = rpf_mod.ControlConfig().waypoint_arrive_radius
    assert rew.ARRIVE_RADIUS_MIN_U <= flown <= rew.ARRIVE_RADIUS_MAX_U
    assert flown == pytest.approx(rew.DEFAULT_ARRIVE_RADIUS_U)


def test_a_site_and_its_imported_copy_are_listed_once(tmp_path):
    """Importing copies a package to managed_root/<site_id>, and managed_root is
    the sites directory -- so both live here and describe the same field."""
    for name in ("football_field", "football_field_edm"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "site_profile.json").write_text(
            json.dumps({"site_id": "football_field_edm", "display_name": "足球場 EDM"}),
            encoding="utf-8")

    packages = lsa.list_site_packages(tmp_path)

    assert len(packages) == 1, [p.folder.name for p in packages]
    # The SOURCE wins, not the copy at managed_root/<site_id>: the copy re-points
    # its bundle at localization/localization_bundle.pt, which carries no trusted
    # digest, so validate_folder refuses it. Listing it gave the operator a button
    # that could not import.
    assert packages[0].folder.name == "football_field"


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


def test_the_panel_keeps_a_standing_route_entry_point():
    """The switch flow offers the editor once; declining must not strand the route."""
    import inspect as _inspect
    import site_assets_panel as sap

    source = _inspect.getsource(sap.SiteAssetsPanel._build_site_shortcuts)
    assert "編輯目前航線" in source, "no way back to an existing route"
    assert "_open_route_editor(True)" in source, "the entry must open in EDIT mode"
    assert "畫新航線" in source


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

    assert [path.name for path in routes] == [new.name, old.name]
    assert all(path.name != "site_profile.json" for path in routes)


def test_route_listing_of_a_missing_folder_is_empty(tmp_path):
    assert lsa.list_site_routes(tmp_path / "absent") == []
