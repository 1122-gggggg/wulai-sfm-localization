from __future__ import annotations

import gc
import json
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import tkinter as tk
from tkinter import ttk

from operator_actions import ActionResult, SiteAssetActions
from site_assets_panel import SiteAssetsPanel


@pytest.fixture
def site_panel(tmp_path):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    folders = {}
    for name in ("A", "B", "Empty"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "site_profile.json").write_text(
            json.dumps({"site_id": name, "display_name": name}), encoding="utf-8"
        )
        if name != "Empty":
            (folder / "routes").mkdir()
            for index in (1, 2):
                (folder / "routes" / f"{name}_route_{index}.json").write_text(
                    json.dumps({
                        "schema": "sfm-flight-route/v1", "purpose": "flight",
                        "waypoints": [[0, 0, 0], [index, 0, 0]],
                    }), encoding="utf-8",
                )
        folders[name] = folder

    imports = []

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
    selected, edited, deleted, applied = [], [], [], []
    panel = SiteAssetsPanel(
        root, actions=actions, request_apply=applied.append,
        request_route_preview=lambda path: selected.append(path) or "selected",
        request_route_edit=edited.append, request_route_delete=deleted.append,
        site_packages_root=tmp_path, site_pack_root=folders["A"],
    )
    panel.pack()
    panel.ask_yes_no = lambda *_args: False
    root.update()
    state = SimpleNamespace(
        root=root, panel=panel, actions=actions, folders=folders, imports=imports,
        selected=selected, edited=edited, deleted=deleted, applied=applied,
    )
    try:
        yield state
    finally:
        # Tk variables must release their Tcl handle on this thread, before a
        # later import worker's cyclic GC can collect the destroyed panel.
        del panel.status_var
        root.destroy()
        state.panel = None
        state.root = None
        del panel, root
        gc.collect()


def _widgets(widget):
    for child in widget.winfo_children():
        yield child
        yield from _widgets(child)


def _texts(panel):
    return [str(widget.cget("text")) for widget in _widgets(panel)
            if isinstance(widget, (ttk.Label, ttk.Button))]


def _route_buttons(panel):
    return [widget for widget in _widgets(panel)
            if isinstance(widget, ttk.Button) and "_route_" in str(widget.cget("text"))]


def _wait_for_import(state):
    deadline = time.monotonic() + 3.0
    while state.panel._busy and time.monotonic() < deadline:
        state.root.update()
        time.sleep(0.005)
    assert not state.panel._busy, "offline import did not finish"
    state.root.update()


def _choose(state, name):
    button = next(widget for widget in _widgets(state.panel)
                  if isinstance(widget, ttk.Button) and widget.cget("text") == name)
    button.invoke()
    _wait_for_import(state)


def test_selected_site_replaces_all_routes_before_apply_and_can_return_to_active(site_panel):
    state = site_panel
    assert len(_route_buttons(state.panel)) == 2
    _choose(state, "B")
    assert {button.cget("text").split(" · ")[0] for button in _route_buttons(state.panel)} == {
        "B_route_1.json", "B_route_2.json",
    }
    assert not any("A_route_" in text for text in _texts(state.panel))
    assert any("待套用場域：B" in text for text in _texts(state.panel))
    assert state.actions.current_profile == state.folders["B"] / "site_profile.json"
    assert state.panel.site_pack_root == state.folders["A"]
    state.panel._set_busy(True)
    state.panel._set_busy(False)
    for button in _route_buttons(state.panel):
        assert button.instate(["disabled"])
        button.invoke()
    path = state.folders["B"] / "routes/B_route_1.json"
    state.panel._preview_existing_route(path)
    state.panel._edit_existing_route(path)
    state.panel._delete_existing_route(path)
    assert not state.selected and not state.edited and not state.deleted and not state.applied

    _choose(state, "A")
    assert all("A_route_" in button.cget("text") for button in _route_buttons(state.panel))
    assert all(not button.instate(["disabled"]) for button in _route_buttons(state.panel))
    assert not any("待套用場域" in text for text in _texts(state.panel))
    _route_buttons(state.panel)[0].invoke()
    assert len(state.selected) == 1
    assert state.selected[0].parent == state.folders["A"] / "routes"


def test_route_selection_is_enabled_only_after_existing_apply_confirmation(site_panel):
    state = site_panel
    state.panel.ask_yes_no = lambda title, _message: title == "切換到這個場域"

    def apply(profile):
        state.applied.append(profile)
        state.panel.activate_site(profile, profile.parent)

    state.panel.request_apply = apply
    _choose(state, "B")
    assert state.applied == [state.folders["B"] / "site_profile.json"]
    assert state.panel.site_pack_root == state.folders["B"]
    assert not any("待套用場域" in text for text in _texts(state.panel))
    assert len(_route_buttons(state.panel)) == 2
    _route_buttons(state.panel)[0].invoke()
    assert state.selected[0].parent == state.folders["B"] / "routes"


def test_no_route_site_clears_previous_choices_and_rollback_restores_active(site_panel):
    state = site_panel
    _choose(state, "Empty")
    assert _route_buttons(state.panel) == []
    assert "此場域尚無可用航線" in _texts(state.panel)
    assert any("待套用場域：Empty" in text for text in _texts(state.panel))
    state.panel.activate_site(state.folders["A"] / "site_profile.json", state.folders["A"])
    assert len(_route_buttons(state.panel)) == 2
    assert all("A_route_" in button.cget("text") for button in _route_buttons(state.panel))
    assert state.actions.current_profile == state.folders["A"] / "site_profile.json"


@pytest.mark.parametrize("pending_site", [False, True])
def test_validation_failure_preserves_previous_complete_selection(site_panel, pending_site):
    state = site_panel
    if pending_site:
        _choose(state, "B")
    previous = [button.cget("text") for button in _route_buttons(state.panel)]
    previous_profile = state.actions.current_profile
    state.panel._import_site_folder(state.folders["A"].parent / "Invalid")
    _wait_for_import(state)
    assert [button.cget("text") for button in _route_buttons(state.panel)] == previous
    assert state.actions.current_profile == previous_profile
    assert state.panel._pending_site_folder is None
    assert "site validation rejected" in state.panel.status_var.get()
    assert all(button.instate(["disabled"]) == pending_site
               for button in _route_buttons(state.panel))


def test_busy_site_import_cannot_replace_first_pending_folder(site_panel):
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
        assert "前一個資產作業尚未完成" in state.panel.status_var.get()
    finally:
        release.set()
    _wait_for_import(state)
    assert state.imports == [state.folders["B"]]
    assert all("B_route_" in button.cget("text") for button in _route_buttons(state.panel))


def test_synchronous_apply_rejection_restores_active_list_and_import_profile(site_panel):
    state = site_panel
    _choose(state, "B")

    def reject(_profile):
        raise ValueError("AUTO mission is active")

    state.panel.request_apply = reject
    state.panel._apply()
    assert all("A_route_" in button.cget("text") for button in _route_buttons(state.panel))
    assert state.actions.current_profile == state.folders["A"] / "site_profile.json"
    assert "AUTO mission is active" in state.panel.status_var.get()
    assert not state.selected


@pytest.mark.parametrize("rolled_back", [False, True])
def test_switch_result_restores_active_routes_for_cancel_and_rollback(
    site_panel, monkeypatch, rolled_back,
):
    import flight_operator_app as app
    from operator_site_runtime import SiteRuntimeSwitchResult

    state = site_panel
    _choose(state, "B")
    old = SimpleNamespace(prepared=SimpleNamespace(profile=SimpleNamespace(
        source=state.folders["A"] / "site_profile.json",
    )))
    result = SiteRuntimeSwitchResult(
        runtime=old, applied=False, old_closed=rolled_back,
        rolled_back=rolled_back, error="offline connection rejected",
    )
    results = queue.Queue()
    results.put(result)
    monkeypatch.setattr(app, "site_pack_root_for_profile", lambda *_: state.folders["A"])
    operator = SimpleNamespace(
        _site_switch_results=results, _site_runtime=old,
        site_assets_panel=state.panel, write_log=lambda _message: None,
        _attach_localizer_file_handler=lambda: None,
        _get_command_coordinator=lambda: SimpleNamespace(resume=lambda: None),
        _install_site_runtime=lambda runtime: state.panel.activate_site(
            runtime.prepared.profile.source, state.folders["A"],
        ),
        _set_site_switch_controls=lambda **_: None,
    )
    app.OperatorApp._poll_site_switch_result(operator)
    assert all("A_route_" in button.cget("text") for button in _route_buttons(state.panel))
    assert state.actions.current_profile == state.folders["A"] / "site_profile.json"
    assert all(not button.instate(["disabled"]) for button in _route_buttons(state.panel))
    assert not state.selected


def test_selected_source_routes_do_not_use_old_active_or_managed_folder(
    site_panel, tmp_path,
):
    state = site_panel
    managed = tmp_path / "managed_B"
    managed.mkdir()
    original = state.actions.import_site_folder

    def import_to_managed(folder):
        result = original(folder)
        state.actions.current_profile = managed / "site_profile.json"
        return ActionResult(result.message, state.actions.current_profile, kind="site")

    state.actions.import_site_folder = import_to_managed
    _choose(state, "B")
    assert len(_route_buttons(state.panel)) == 2
    assert all("B_route_" in button.cget("text") for button in _route_buttons(state.panel))
    assert state.panel.site_pack_root == state.folders["A"]


def test_route_import_failure_keeps_only_selected_site_routes(site_panel):
    state = site_panel
    (state.folders["B"] / "routes/B_route_2.json").unlink()

    def reject_route(*_args):
        raise ValueError("route validation rejected")

    state.actions.routes.import_file = reject_route
    _choose(state, "B")
    assert len(_route_buttons(state.panel)) == 1
    assert _route_buttons(state.panel)[0].cget("text").startswith("B_route_1.json")
    assert _route_buttons(state.panel)[0].instate(["disabled"])
    assert not any("A_route_" in text for text in _texts(state.panel))
    assert "route validation rejected" in state.panel.status_var.get()
    assert not state.selected and not state.applied


def test_route_list_read_failure_clears_old_routes_and_shows_reason(site_panel, monkeypatch):
    import site_assets_panel as sap

    state = site_panel
    describe = sap.describe_site_routes

    def reject_b(folder):
        if Path(folder) == state.folders["B"]:
            raise OSError("route folder unavailable")
        return describe(folder)

    monkeypatch.setattr(sap, "describe_site_routes", reject_b)
    _choose(state, "B")
    assert _route_buttons(state.panel) == []
    assert any("route folder unavailable" in text for text in _texts(state.panel))
    assert any("待套用場域：B" in text for text in _texts(state.panel))


def test_pending_route_save_refresh_includes_new_managed_route(site_panel, tmp_path):
    state = site_panel
    _choose(state, "B")
    managed = tmp_path / "managed_B"
    (managed / "routes").mkdir(parents=True)
    route = managed / "routes/B_route_authored.json"
    route.write_text(json.dumps({
        "schema": "sfm-flight-route/v1", "purpose": "flight",
        "waypoints": [[0, 0, 0], [1, 0, 0]],
    }), encoding="utf-8")
    state.actions.current_profile = managed / "site_profile.json"
    state.panel._finish_ok(ActionResult(
        "已新增航線；待套用場域", state.actions.current_profile, route, "route",
    ))
    assert len(_route_buttons(state.panel)) == 3
    assert all(button.instate(["disabled"]) for button in _route_buttons(state.panel))
    assert not any("A_route_" in text for text in _texts(state.panel))
    assert not state.selected and not state.applied
