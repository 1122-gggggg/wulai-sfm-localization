"""Offline checks of real route selection and operator readiness feedback."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from flight_operator_app import OperatorApp, resolve_site_map_frame
from local_site_assets import describe_site_routes
from operator_autonomy import DesktopRouteAutonomy
from route_domain import MissionRouteLock, RouteDocument
from site_profile import load_site_profile

ROOT = Path(__file__).resolve().parents[3]
SITE = ROOT / "地圖檔/場域/river_site"
ROUTES = describe_site_routes(SITE)


@pytest.mark.parametrize("route", ROUTES, ids=lambda route: route.path.stem)
def test_available_route_overlay_is_the_desktop_auto_input(route):
    """Select through the UI path; build AUTO, without starting any worker."""
    profile_path = SITE / "site_profile.json"
    frame = resolve_site_map_frame(load_site_profile(profile_path))
    operator = OperatorApp.__new__(OperatorApp)
    operator.site_profile_path = profile_path
    operator.mission_route_lock = MissionRouteLock()
    operator._active_site_map_frame = lambda: frame
    operator.redraw_map_only = Mock()
    assert route.flight_ready
    assert operator._show_route_overlay(route.path) == route.waypoints
    snapshot = operator.mission_route_lock.begin_auto(
        displayed_sha256=operator._displayed_route_sha256
    )
    backend = Mock()
    auto = DesktopRouteAutonomy(
        backend=backend, snapshot=snapshot, map_frame=frame,
        get_pose=lambda: None, pose_is_weak=lambda: False,
        pose_confidence=lambda: 100, force_relocalize=Mock(),
        stream_healthy=lambda: True, takeoff=Mock(), land=Mock(),
    )
    expected = RouteDocument.from_path(route.path, map_frame=frame).controller_waypoints()
    np.testing.assert_allclose(operator.route_pts, expected)
    np.testing.assert_allclose(auto.waypoints, expected)
    np.testing.assert_allclose(auto.controller.wp[:len(expected)], expected)
    assert snapshot.path == route.path
    auto.takeoff.assert_not_called()
    auto.land.assert_not_called()
    assert not backend.mock_calls


def _guidance(*, complete=True, paused=False, active=False, approved=True):
    operator = OperatorApp.__new__(OperatorApp)
    operator.preflight_guide = SimpleNamespace(complete=complete, current_step="system")
    operator._preflight_step_evidence = lambda *_: (None, "等待即時遙測讀回")
    operator._auto_paused = paused
    operator._integrated_auto_active = lambda: active
    operator._autonomy_gate_snapshot = lambda: {
        "mission_flight_ready": approved,
        "autonomous_locked": False,
        "autonomous_approval_valid": True,
        "profile_verified": True,
    }
    operator.mission_route_lock = SimpleNamespace(snapshot=SimpleNamespace(
        path=Path("selected-route.json"), waypoints=((0, 0, 0), (1, 0, 0)),
    ))
    operator.flight_action_hint_var = Mock()
    operator.flight_action_hint = Mock()
    return operator


@pytest.mark.parametrize("gps", [True, False, None])
def test_unfinished_preflight_never_advertises_auto_ready(gps):
    operator = _guidance(complete=False)
    operator._update_flight_action_guidance(SimpleNamespace(flight_state="landed", gps_fixed=gps))
    text = operator.flight_action_hint_var.set.call_args.args[0]
    assert "請先完成四項驗證" in text
    assert "等待即時遙測讀回" in text
    assert "可按" not in text


@pytest.mark.parametrize("gps", [True, False, None])
def test_ready_guidance_identifies_route_and_visual_localization_requirement(gps):
    operator = _guidance()
    operator._update_flight_action_guidance(SimpleNamespace(flight_state="landed", gps_fixed=gps))
    text = operator.flight_action_hint_var.set.call_args.args[0]
    assert "selected-route.json（2 點）" in text
    assert "定位穩定後從第 1 航點" in text
    assert "起飛並等待定位" in text


def test_complete_preflight_does_not_hide_mission_approval_blocker():
    operator = _guidance(approved=False)
    operator._update_flight_action_guidance(SimpleNamespace(flight_state="landed"))
    text = operator.flight_action_hint_var.set.call_args.args[0]
    assert "自動飛行：尚未就緒" in text
    assert "mission is not flight-ready" in text
    assert "起飛並等待定位" not in text


@pytest.mark.parametrize("paused,active,expected", [
    (True, True, "繼續自動飛行"),
    (False, True, "執行中"),
    (False, False, "接續巡航"),
])
def test_airborne_guidance_distinguishes_pause_active_and_restart(paused, active, expected):
    operator = _guidance(paused=paused, active=active)
    operator._update_flight_action_guidance(SimpleNamespace(flight_state="hovering"))
    text = operator.flight_action_hint_var.set.call_args.args[0]
    assert expected in text
    assert "起飛並等待定位" not in text
