from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


AUTHORING_ROOT = Path(__file__).resolve().parents[2] / "控制介面程式" / "authoring"


class _FakeOperator:
    pass


class _FakeViewMenu:
    draw = SimpleNamespace(_draw_funcs=[])

    @staticmethod
    def append(_func):
        return None

    @staticmethod
    def remove(_func):
        return None


class _FakeObjects:
    @staticmethod
    def get(_name):
        return None

    @staticmethod
    def remove(_obj, do_unlink=False):
        return None


def _install_fake_blender(monkeypatch, tmp_path):
    bpy = ModuleType("bpy")
    bpy.types = SimpleNamespace(Operator=_FakeOperator, VIEW3D_MT_view=_FakeViewMenu)
    bpy.utils = SimpleNamespace(
        register_class=lambda _cls: None,
        unregister_class=lambda _cls: None,
    )
    bpy.context = SimpleNamespace(
        preferences=SimpleNamespace(
            view=SimpleNamespace(show_developer_ui=False),
            inputs=SimpleNamespace(use_mouse_emulate_3_button=False),
        ),
        window_manager=SimpleNamespace(
            keyconfigs=SimpleNamespace(addon=None, user=None, active=None),
        ),
    )
    bpy.data = SimpleNamespace(objects=_FakeObjects())
    monkeypatch.setitem(sys.modules, "bpy", bpy)

    bpy_extras = ModuleType("bpy_extras")
    bpy_extras.view3d_utils = ModuleType("bpy_extras.view3d_utils")
    monkeypatch.setitem(sys.modules, "bpy_extras", bpy_extras)
    monkeypatch.setitem(sys.modules, "bpy_extras.view3d_utils", bpy_extras.view3d_utils)
    monkeypatch.setenv("SFM_MAP_ROOT", str(tmp_path))
    monkeypatch.setenv("SFM_SAFEZONE_DIR", str(tmp_path / "safezone"))


@pytest.fixture
def authoring_modules(monkeypatch, tmp_path):
    _install_fake_blender(monkeypatch, tmp_path)
    modules = {}
    for filename, name in (
        ("draw_path.py", "authoring_draw_path_test"),
        ("draw_poles.py", "authoring_draw_poles_test"),
        ("draw_wires.py", "authoring_draw_wires_test"),
        ("edit_path_height.py", "authoring_edit_height_test"),
    ):
        spec = importlib.util.spec_from_file_location(name, AUTHORING_ROOT / filename)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def _context(area_type="VIEW_3D"):
    area = None
    if area_type is not None:
        area = SimpleNamespace(
            type=area_type,
            header_text_set=lambda _message: None,
            tag_redraw=lambda: None,
        )
    return SimpleNamespace(area=area)


def _event(event_type, value="PRESS", *, alt=False, shift=False):
    return SimpleNamespace(type=event_type, value=value, alt=alt, shift=shift)


def _attach_reporter(operator):
    reports = []
    operator.report = lambda level, message: reports.append((level, message))
    return reports


def test_draw_path_modal_preserves_edit_navigation_and_finish_dispatch(authoring_modules, monkeypatch):
    module = authoring_modules["authoring_draw_path_test"]
    operator = module.PATH_OT_draw()
    operator._wp = []
    operator._closed = False
    reports = _attach_reporter(operator)
    operator._pick_xy = lambda _context, _event: (1.0, 2.0)
    monkeypatch.setattr(module, "_draw", lambda _wp, _closed: None)
    saved = []
    monkeypatch.setattr(module, "_save", lambda wp, closed: saved.append((wp[:], closed)))

    assert operator.modal(_context(None), _event("MOUSEMOVE")) == {"PASS_THROUGH"}
    assert operator.modal(_context(), _event("MOUSEMOVE", alt=True)) == {"PASS_THROUGH"}
    assert operator.modal(_context(), _event("LEFTMOUSE", shift=True)) == {"PASS_THROUGH"}
    assert operator.modal(_context(), _event("LEFTMOUSE")) == {"RUNNING_MODAL"}
    operator._wp.append([3.0, 4.0, 9.0])
    assert operator.modal(_context(), _event("WHEELUPMOUSE")) == {"RUNNING_MODAL"}
    assert operator.modal(_context(), _event("A")) == {"RUNNING_MODAL"}
    assert operator._wp == [[1.0, 2.0, 9.25], [3.0, 4.0, 9.25]]
    assert operator.modal(_context(), _event("C")) == {"RUNNING_MODAL"}
    assert operator._closed is True
    assert operator.modal(_context(), _event("S")) == {"RUNNING_MODAL"}
    assert operator.modal(_context(), _event("MIDDLEMOUSE")) == {"PASS_THROUGH"}
    assert operator.modal(_context(), _event("RET")) == {"FINISHED"}
    assert saved[-1][1] is True
    assert reports[-1][1] == "done, 2 waypoints saved"


def test_draw_poles_modal_preserves_two_click_radius_undo_and_finish(authoring_modules, monkeypatch):
    module = authoring_modules["authoring_draw_poles_test"]
    operator = module.POLE_OT_draw()
    operator._poles = []
    operator._pending_base = None
    reports = _attach_reporter(operator)
    points = iter(([0.0, 0.0, 1.0], [0.0, 0.0, 3.0]))
    operator._snap = lambda _context, _event: next(points)
    monkeypatch.setattr(module, "_redraw", lambda _poles: None)
    saved = []
    monkeypatch.setattr(module, "_save", lambda poles: saved.append(poles[:]))

    assert operator.modal(_context(), _event("LEFTMOUSE")) == {"RUNNING_MODAL"}
    assert operator._pending_base == [0.0, 0.0, 1.0]
    assert operator.modal(_context(), _event("LEFTMOUSE")) == {"RUNNING_MODAL"}
    assert len(operator._poles) == 1
    assert operator.modal(_context(), _event("WHEELUPMOUSE")) == {"RUNNING_MODAL"}
    assert operator._poles[0]["radius"] == pytest.approx(0.10)
    assert operator.modal(_context(), _event("Z")) == {"RUNNING_MODAL"}
    assert operator._poles == []
    assert operator.modal(_context(), _event("S")) == {"RUNNING_MODAL"}
    assert operator.modal(_context(), _event("ESC")) == {"FINISHED"}
    assert len(saved) == 2
    assert reports[-1][1] == "done, 0 poles saved"


def test_draw_wires_modal_preserves_endpoint_sag_undo_and_finish(authoring_modules, monkeypatch):
    module = authoring_modules["authoring_draw_wires_test"]
    operator = module.WIRE_OT_draw()
    operator._wires = []
    operator._pending = None
    reports = _attach_reporter(operator)
    points = iter(([0.0, 0.0, 2.0], [2.0, 0.0, 2.0]))
    operator._snap = lambda _context, _event: next(points)
    monkeypatch.setattr(module, "_make_curve", lambda *_args: None)
    saved = []
    monkeypatch.setattr(module, "_save", lambda wires: saved.append(wires[:]))

    assert operator.modal(_context(), _event("LEFTMOUSE")) == {"RUNNING_MODAL"}
    assert operator.modal(_context(), _event("LEFTMOUSE")) == {"RUNNING_MODAL"}
    assert len(operator._wires) == 1
    initial_sag = operator._wires[0]["sag"]
    assert operator.modal(_context(), _event("WHEELUPMOUSE")) == {"RUNNING_MODAL"}
    assert operator._wires[0]["sag"] > initial_sag
    assert operator.modal(_context(), _event("Z")) == {"RUNNING_MODAL"}
    assert operator._wires == []
    assert operator.modal(_context(), _event("S")) == {"RUNNING_MODAL"}
    assert operator.modal(_context(), _event("NUMPAD_3")) == {"PASS_THROUGH"}
    assert operator.modal(_context(), _event("RET")) == {"FINISHED"}
    assert len(saved) == 2
    assert reports[-1][1] == "done, 0 wires saved"


def test_edit_height_modal_preserves_adjustment_ramp_navigation_and_finish(
    authoring_modules, monkeypatch
):
    module = authoring_modules["authoring_edit_height_test"]
    operator = module.PATH_OT_edit_height()
    operator._wp = [[0.0, 0.0, 1.0], [1.0, 0.0, 3.0], [2.0, 0.0, 5.0]]
    operator._closed = False
    operator._sel = 0
    reports = _attach_reporter(operator)
    operator._pick = lambda _context, _event: 1
    monkeypatch.setattr(module, "_draw", lambda *_args: None)
    saved = []
    monkeypatch.setattr(module, "_save", lambda wp, closed: saved.append((wp[:], closed)))

    assert operator.modal(_context(None), _event("MOUSEMOVE")) == {"PASS_THROUGH"}
    assert operator.modal(_context(), _event("LEFTMOUSE")) == {"RUNNING_MODAL"}
    assert operator._sel == 1
    assert operator.modal(_context(), _event("WHEELUPMOUSE")) == {"RUNNING_MODAL"}
    assert operator.modal(_context(), _event("PAGE_DOWN")) == {"RUNNING_MODAL"}
    assert operator.modal(_context(), _event("RIGHT_BRACKET")) == {"RUNNING_MODAL"}
    assert operator._sel == 2
    assert operator.modal(_context(), _event("L")) == {"RUNNING_MODAL"}
    assert [point[2] for point in operator._wp] == [1.0, 3.0, 5.0]
    assert operator.modal(_context(), _event("S")) == {"RUNNING_MODAL"}
    assert operator.modal(_context(), _event("NUMPAD_1")) == {"PASS_THROUGH"}
    assert operator.modal(_context(), _event("ESC")) == {"FINISHED"}
    assert len(saved) == 2
    assert reports[-1][1] == "done, 3 waypoints saved"
