from __future__ import annotations

import gc
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import tkinter as tk
from tkinter import ttk

from operator_actions import ActionResult, SiteAssetActions
from site_assets_panel import SiteAssetsPanel
from operator_ui_style import apply_operator_styles


def _write_route(path, index=1, *, purpose="flight"):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps({
        "schema": "sfm-flight-route/v1", "purpose": purpose,
        "waypoints": [[0, 0, 0], [index, 0, 0]],
    }), encoding="utf-8")
    return path


@pytest.fixture
def site_panel(tmp_path):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    apply_operator_styles(root)
    folders = {}
    for name in ("A", "B", "Empty", "Invalid"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "site_profile.json").write_text(
            "not JSON" if name == "Invalid" else
            json.dumps({"site_id": name, "display_name": name}), encoding="utf-8",
        )
        if name in ("A", "B"):
            for index in (1, 2):
                _write_route(folder / "routes" / f"{name}_route_{index}.json", index)
        folders[name] = folder

    imports, selected, edited, deleted, applied, created, saved = ([] for _ in range(7))

    def import_folder(folder):
        folder = Path(folder)
        imports.append(folder)
        if not (folder / "site_profile.json").is_file():
            raise ValueError("site validation rejected")
        return SimpleNamespace(
            profile_path=folder / "site_profile.json", site_id=folder.name,
            already_present=True,
        )

    actions = SiteAssetActions(
        SimpleNamespace(import_folder=import_folder), SimpleNamespace(),
        SimpleNamespace(), current_profile=folders["A"] / "site_profile.json",
    )
    panel = SiteAssetsPanel(
        root, actions=actions, request_apply=applied.append,
        request_route_preview=lambda path: selected.append(path) or "selected",
        request_route_edit=edited.append,
        request_route_delete=lambda path: deleted.append(path) or ActionResult(
            "deleted", actions.current_profile, path, "route_deleted",
        ),
        request_route_editor=created.append, route_imported=saved.append,
        site_packages_root=tmp_path, site_pack_root=folders["A"],
    )
    panel.pack(fill="both", expand=True)
    panel.ask_yes_no = lambda *_args: False
    state = SimpleNamespace(
        root=root, panel=panel, actions=actions, folders=folders, imports=imports,
        selected=selected, edited=edited, deleted=deleted, applied=applied,
        created=created, saved=saved,
    )

    def apply(profile):
        applied.append(profile)
        state.panel.activate_site(profile, profile.parent)

    panel.request_apply = apply
    root.update()
    try:
        yield state
    finally:
        # Release Tcl handles on the UI thread before import workers can run GC.
        del panel.status_var
        root.destroy()
        state.panel = state.root = None
        del panel, root
        gc.collect()


def _widgets(widget):
    for child in widget.winfo_children():
        yield child
        yield from _widgets(child)


def _texts(panel):
    return [str(widget.cget("text")) for widget in _widgets(panel)
            if isinstance(widget, (ttk.Label, ttk.Button))]


def _routes(panel):
    return {item.path for item in panel._route_choices}


def _wait_for_import(state):
    deadline = time.monotonic() + 3.0
    while state.panel._busy and time.monotonic() < deadline:
        state.root.update()
        time.sleep(0.005)
    assert not state.panel._busy, "offline import did not finish"
    state.root.update()


def _choose(state, name):
    selector = state.panel._map_selector
    index = next(index for index, item in enumerate(state.panel._map_choices)
                 if item.folder == state.folders[name])
    selector.current(index)
    selector.event_generate("<<ComboboxSelected>>")
    state.root.update()


def _choose_route(state, path):
    tree = state.panel._route_list
    index = next(index for index, item in enumerate(state.panel._route_choices)
                 if item.path == path)
    tree.selection_set(str(index))
    tree.see(str(index))
    state.root.update()


def test_registered_map_selection_applies_immediately_without_import_or_confirmation(site_panel):
    state = site_panel

    def unexpected_dialog(*_args):
        pytest.fail("registered selection must not ask for separate apply confirmation")

    state.panel.ask_yes_no = unexpected_dialog
    _choose(state, "B")
    assert state.applied == [state.folders["B"] / "site_profile.json"]
    assert not state.imports and not state.saved
    assert _routes(state.panel) == {
        state.folders["B"] / "routes/B_route_1.json",
        state.folders["B"] / "routes/B_route_2.json",
    }
    assert state.actions.current_profile == state.applied[-1]
    _choose_route(state, state.folders["B"] / "routes/B_route_1.json")
    assert state.selected == [state.folders["B"] / "routes/B_route_1.json"]

    _choose(state, "A")
    assert all(path.parent == state.folders["A"] / "routes" for path in _routes(state.panel))
    assert state.panel._selected_route_path is None
    assert not state.panel._route_list.selection()


def test_active_map_remains_selectable_without_restart_or_losing_route(site_panel):
    state = site_panel
    path = state.folders["A"] / "routes/A_route_1.json"
    _choose_route(state, path)
    _choose(state, "A")
    assert not state.applied and not state.imports
    assert state.panel._selected_route_path == path
    assert state.selected == [path]
    assert state.panel._map_choices[state.panel._map_selector.current()].folder == state.folders["A"]


def test_old_routes_remain_until_activation_and_pending_switch_blocks_handlers(site_panel):
    state = site_panel
    original = _routes(state.panel)
    state.panel.request_apply = state.applied.append
    _choose(state, "B")
    assert _routes(state.panel) == original
    assert state.panel.site_pack_root == state.folders["A"]
    assert state.panel._busy
    assert state.panel._map_selector.instate(["disabled"])
    _choose_route(state, next(iter(original)))
    _choose(state, "Empty")
    state.panel._import_site_folder(state.folders["Empty"])
    assert not state.selected and not state.imports
    assert state.applied == [state.folders["B"] / "site_profile.json"]
    state.panel.activate_site(state.applied[0], state.folders["B"])
    assert not state.panel._busy
    assert all(path.parent == state.folders["B"] / "routes" for path in _routes(state.panel))
    assert not state.panel._route_list.selection()


@pytest.mark.parametrize("reason", [
    "AUTO mission is active", "aircraft flying", "landing not confirmed",
])
def test_existing_flight_guard_blocks_map_selection_import_and_route_changes(site_panel, reason):
    state = site_panel
    state.panel.flight_state_check = lambda: (False, reason)
    _choose(state, "B")
    state.panel._import_site_folder(state.folders["B"])
    _choose_route(state, next(iter(_routes(state.panel))))
    state.panel._open_route_editor(False)
    assert not state.applied and not state.imports and not state.selected and not state.created
    assert state.panel.site_pack_root == state.folders["A"]
    assert state.panel._map_choices[state.panel._map_selector.current()].folder == state.folders["A"]
    assert reason in state.panel.status_var.get()


def test_root_editor_guard_rejection_retains_active_map_and_selected_route(site_panel):
    state = site_panel
    path = state.folders["A"] / "routes/A_route_1.json"
    _choose_route(state, path)

    def reject(_profile):
        raise ValueError("close the route editor first")

    state.panel.request_apply = reject
    _choose(state, "B")
    assert state.panel.site_pack_root == state.folders["A"]
    assert state.actions.current_profile == state.folders["A"] / "site_profile.json"
    assert state.panel._selected_route_path == path
    assert _routes(state.panel) == {state.folders["A"] / f"routes/A_route_{index}.json"
                                    for index in (1, 2)}
    assert "close the route editor first" in state.panel.status_var.get()
    assert not state.panel._busy


@pytest.mark.parametrize("result_message", ["switch cancelled", "switch failed, rolled back"])
def test_async_cancel_and_rollback_restore_previous_complete_selection(site_panel, result_message):
    state = site_panel
    path = state.folders["A"] / "routes/A_route_1.json"
    _choose_route(state, path)
    state.panel.request_apply = state.applied.append
    _choose(state, "B")
    # Canonical runtime result handlers call activate_site with the retained runtime.
    state.panel.activate_site(state.folders["A"] / "site_profile.json", state.folders["A"])
    state.panel.set_status(result_message)
    state.panel._set_busy(False)
    assert state.panel._selected_route_path == path
    assert state.panel._map_choices[state.panel._map_selector.current()].folder == state.folders["A"]
    assert all(item.parent == path.parent for item in _routes(state.panel))
    assert not state.panel._busy
    _choose(state, "B")
    assert len(state.applied) == 2
    state.panel.activate_site(state.folders["B"] / "site_profile.json", state.folders["B"])


def test_invalid_map_is_truthfully_listed_but_cannot_replace_active_map(site_panel):
    state = site_panel
    item = next(item for item in state.panel._map_choices if item.folder == state.folders["Invalid"])
    assert item.error
    _choose(state, "Invalid")
    assert not state.applied and not state.imports
    assert item.error in state.panel.status_var.get()
    assert state.panel.site_pack_root == state.folders["A"]
    assert all(path.parent == state.folders["A"] / "routes" for path in _routes(state.panel))


def test_empty_map_activation_clears_old_route_selection_and_offers_creation(site_panel):
    state = site_panel
    _choose_route(state, next(iter(_routes(state.panel))))
    _choose(state, "Empty")
    assert not _routes(state.panel)
    assert state.panel._selected_route_path is None
    assert "此地圖尚無可用航線" in _texts(state.panel)
    assert state.panel.site_pack_root == state.folders["Empty"]
    state.panel._open_route_editor(False)
    assert state.created == [False]


def test_empty_registry_keeps_active_map_and_new_folder_import_available(site_panel, tmp_path):
    state = site_panel
    state.panel.site_packages_root = tmp_path / "no-packages"
    state.panel._refresh()
    assert [item.folder for item in state.panel._map_choices] == [state.folders["A"]]
    assert state.panel.add_map_button.winfo_manager() == "grid"
    _choose(state, "A")
    assert not state.applied


def test_route_list_read_failure_does_not_mix_old_routes(site_panel, monkeypatch):
    import site_assets_panel as sap

    state = site_panel
    describe = sap.describe_site_routes

    def reject_b(folder):
        if Path(folder) == state.folders["B"]:
            raise OSError("route folder unavailable")
        return describe(folder)

    monkeypatch.setattr(sap, "describe_site_routes", reject_b)
    _choose(state, "B")
    assert not _routes(state.panel)
    assert any("route folder unavailable" in text for text in _texts(state.panel))
    assert state.panel.site_pack_root == state.folders["B"]
    assert state.panel._selected_route_path is None


def test_new_folder_import_applies_then_renders_only_activated_map(site_panel):
    state = site_panel
    state.panel._import_site_folder(state.folders["B"])
    _wait_for_import(state)
    assert state.imports == [state.folders["B"]]
    assert state.applied == [state.folders["B"] / "site_profile.json"]
    assert state.panel.site_pack_root == state.folders["B"]
    assert all(path.parent == state.folders["B"] / "routes" for path in _routes(state.panel))
    assert not state.saved and not state.selected


def test_cancelled_add_map_dialog_retains_selection_without_import(site_panel):
    state = site_panel
    path = next(iter(_routes(state.panel)))
    _choose_route(state, path)
    state.panel._choose_site()
    dialog = state.panel._add_map_dialog
    assert dialog is not None and dialog.winfo_exists()
    # Non-modal: the flight bar behind it must stay clickable.
    assert dialog.grab_current() is None
    dialog.cancel()
    state.root.update()
    assert not state.imports and not state.applied
    assert state.panel._selected_route_path == path


def test_add_map_dialog_only_imports_a_folder_that_passes_preview(site_panel, monkeypatch):
    import add_map_dialog
    from local_site_assets import SiteFolderFile, SiteFolderPreview

    state = site_panel
    folder = state.folders["B"]
    state.panel._choose_site()
    dialog = state.panel._add_map_dialog
    dialog.navigate(folder.parent, select=folder)
    state.root.update()
    assert not dialog.preview.ready
    assert dialog.import_button.instate(["disabled"])
    dialog.confirm()
    assert not state.imports and dialog.winfo_exists()

    ready = SiteFolderPreview(
        folder, display_name="B", site_id="B",
        files=(SiteFolderFile("場域 manifest", folder / "site_profile.json", 1),),
    )
    monkeypatch.setattr(add_map_dialog, "preview_site_folder", lambda _folder: ready)
    dialog.show_preview(folder)
    assert dialog.import_button.instate(["!disabled"])
    dialog.confirm()
    _wait_for_import(state)
    assert state.imports == [folder]
    assert state.applied == [folder / "site_profile.json"]


def test_import_validation_failure_preserves_active_map_and_profile(site_panel):
    state = site_panel
    previous = _routes(state.panel)
    state.panel._import_site_folder(state.folders["A"].parent / "missing")
    _wait_for_import(state)
    assert _routes(state.panel) == previous
    assert state.actions.current_profile == state.folders["A"] / "site_profile.json"
    assert "site validation rejected" in state.panel.status_var.get()
    assert not state.applied


def test_busy_import_cannot_replace_pending_folder(site_panel):
    state = site_panel
    started, release = threading.Event(), threading.Event()
    original = state.actions.import_site_folder

    def blocked_import(folder):
        started.set()
        if not release.wait(3.0):
            raise RuntimeError("offline worker release timed out")
        return original(folder)

    state.actions.import_site_folder = blocked_import
    try:
        state.panel._import_site_folder(state.folders["B"])
        assert started.wait(1.0)
        state.panel._import_site_folder(state.folders["Empty"])
        assert state.panel._pending_site_folder == state.folders["B"]
    finally:
        release.set()
    _wait_for_import(state)
    assert state.imports == [state.folders["B"]]
    assert state.panel.site_pack_root == state.folders["B"]



def test_landing_guard_changed_after_import_prevents_apply_and_restores_profile(site_panel):
    state = site_panel
    old_profile = state.actions.current_profile
    state.actions.current_profile = state.folders["B"] / "site_profile.json"
    state.panel.flight_state_check = lambda: (False, "landing not confirmed")
    state.panel._finish_ok(ActionResult(
        "imported", state.actions.current_profile, kind="site",
    ))
    assert not state.applied
    assert state.actions.current_profile == old_profile
    assert state.panel.site_pack_root == state.folders["A"]
    assert all(path.parent == state.folders["A"] / "routes" for path in _routes(state.panel))
    assert "landing not confirmed" in state.panel.status_var.get()

def test_successful_authored_save_refreshes_active_routes_without_other_maps(site_panel):
    state = site_panel
    route = state.folders["A"] / "routes/A_route_authored.json"
    calls = []

    def save(source, *, replace_route):
        calls.append((source, replace_route))
        _write_route(route)
        return ActionResult("saved and approved", state.actions.current_profile, route, "route")

    state.actions.import_authored_route = save
    result = state.panel.import_authored_route("editor-output.json", replace_route=None)
    assert calls == [("editor-output.json", None)]
    assert state.saved == [result]
    assert route in _routes(state.panel)
    assert all(path.parent == route.parent for path in _routes(state.panel))
    _choose_route(state, route)
    assert state.selected == [route]


def test_save_failure_does_not_publish_or_refresh_a_claimed_approval(site_panel):
    state = site_panel
    before = _routes(state.panel)

    def reject(*_args, **_kwargs):
        raise ValueError("invalid coordinate frame")

    state.actions.import_authored_route = reject
    with pytest.raises(ValueError, match="invalid coordinate frame"):
        state.panel.import_authored_route("editor-output.json")
    assert not state.saved
    assert _routes(state.panel) == before


def test_nonflight_route_can_be_edited_or_deleted_without_selecting_for_auto(site_panel):
    state = site_panel
    path = _write_route(state.folders["A"] / "routes/A_route_display.json", purpose="preview")
    state.panel._refresh()
    _choose_route(state, path)
    assert not state.selected
    state.panel._route_action(state.panel._edit_existing_route)
    assert state.edited == [path]
    state.panel.ask_yes_no = lambda *_args: True
    state.panel._route_action(state.panel._delete_existing_route)
    assert state.deleted == [path]


@pytest.mark.parametrize("route_count", [2, 5, 25])
def test_route_actions_remain_accessible_at_minimum_viewport(site_panel, route_count):
    from flight_operator_app import UI_MIN_SIZE

    state = site_panel
    for index in range(3, route_count + 1):
        _write_route(state.folders["A"] / f"routes/A_route_{index}.json", index)
    state.panel._refresh()
    state.root.geometry(f"{UI_MIN_SIZE[0]}x{UI_MIN_SIZE[1]}")
    state.root.deiconify()
    state.panel.pack_forget()
    state.panel.place(x=0, y=0, width=330, height=260)
    state.root.update()
    path = state.folders["A"] / f"routes/A_route_{route_count}.json"
    _choose_route(state, path)
    assert state.selected == [path]
    assert len(state.panel._route_list.get_children()) == route_count
    assert state.panel._route_list.bbox(state.panel._route_list.selection()[0])
    for widget in (state.panel._map_selector, state.panel._route_list, *state.panel._buttons):
        assert widget.winfo_ismapped()
        assert widget.winfo_rooty() >= state.panel.winfo_rooty()
        assert (widget.winfo_rooty() + widget.winfo_height()
                <= state.panel.winfo_rooty() + state.panel.winfo_height())
        assert widget.winfo_width() > 0
    state.panel._route_action(state.panel._edit_existing_route)
    state.panel._open_route_editor(False)
    assert state.edited == [path] and state.created == [False]


def test_long_map_names_and_errors_remain_readable_without_displacing_route_actions(site_panel):
    state = site_panel
    long_name = "北側河岸巡檢場域與高壓電塔定位地圖_" * 8
    profile = state.folders["A"] / "site_profile.json"
    profile.write_text(json.dumps({"site_id": "A", "display_name": long_name}), encoding="utf-8")
    state.panel._refresh()
    state.root.geometry("1180x720")
    state.root.deiconify()
    state.panel.pack_forget()
    state.panel.place(x=0, y=0, width=330, height=260)
    state.root.update()
    positions = [(button.winfo_x(), button.winfo_y()) for button in state.panel._buttons]
    message = "地圖切換失敗，保留目前地圖與航線：" + "來源尚未通過完整驗證；" * 30
    state.panel.set_status(message)
    state.root.update()
    text = state.panel._status_text
    assert message in text.get("1.0", "end")
    assert long_name in text.get("1.0", "end")
    assert text.yview()[1] < 1.0
    text.yview_moveto(1.0)
    state.root.update()
    assert text.yview()[1] == pytest.approx(1.0)
    assert [(button.winfo_x(), button.winfo_y()) for button in state.panel._buttons] == positions
    for button in state.panel._buttons:
        assert button.winfo_rooty() + button.winfo_height() <= state.panel.winfo_rooty() + state.panel.winfo_height()
    assert state.panel.site_pack_root == state.folders["A"]
