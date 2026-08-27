import math

import numpy as np
import pytest

from route_geometry import (RouteModel, adaptive_lookahead, aligned_to_raw,
                            build_route, heading_of, horiz, raw_to_aligned,
                            sample_start_offset, up_of, wrap_angle)


def L_route():
    # straight line along +x at up=2 (y=-2), 3 segments of 4 m
    return RouteModel([np.array([i * 4.0, -2.0, 0.0]) for i in range(4)])


def test_wrap_angle():
    assert wrap_angle(0.0) == 0.0
    assert abs(wrap_angle(math.pi + 0.1) - (-math.pi + 0.1)) < 1e-12
    assert abs(wrap_angle(-math.pi - 0.1) - (math.pi - 0.1)) < 1e-12
    assert abs(wrap_angle(4 * math.pi)) < 1e-12


def test_aligned_raw_roundtrip():
    p = [1.0, 2.0, 3.0]
    raw = aligned_to_raw(p)                    # aligned z-up -> raw (x, -z, y)
    assert np.allclose(raw, [1.0, -3.0, 2.0])
    assert np.allclose(raw_to_aligned(raw), p)


def test_nearest_polyline_projection_not_waypoint():
    r = L_route()
    # point beside the MIDDLE of segment 1: nearest waypoint is 4 m away at
    # x=4 or x=8, but the nearest polyline point is the perpendicular foot.
    p = np.array([6.0, -2.0, 1.0])
    pr = r.project(p)
    assert pr.seg_index == 1
    assert pr.cross_track == pytest.approx(1.0)
    assert pr.point[0] == pytest.approx(6.0)
    assert pr.s == pytest.approx(6.0)
    d_wp = min(np.linalg.norm(horiz(p) - horiz(w)) for w in r.wp)
    assert pr.cross_track < d_wp               # projection beats nearest waypoint


def test_tube_projection_uses_full_3d_distance():
    r = L_route()
    p = np.array([6.0, 0.0, 0.0])              # directly above segment 1 by 2 units
    horiz_pr = r.project(p)
    tube_pr = r.project_tube(p)
    assert horiz_pr.cross_track == pytest.approx(0.0)
    assert tube_pr.distance == pytest.approx(2.0)
    assert tube_pr.seg_index == 1
    assert np.allclose(tube_pr.point, [6.0, -2.0, 0.0])


def test_tube_projection_clamps_to_route_ends():
    r = L_route()
    tube_pr = r.project_tube(np.array([-1.0, -2.0, 0.0]))
    assert tube_pr.seg_index == 0
    assert tube_pr.t == 0.0
    assert tube_pr.distance == pytest.approx(1.0)


def test_tube_projection_can_limit_to_segment_window():
    r = RouteModel([
        np.array([0.0, -2.0, 0.0]),
        np.array([6.0, -2.0, 0.0]),
        np.array([6.0, -2.0, 1.0]),
        np.array([0.0, -2.0, 1.0]),
    ])
    p = np.array([3.0, -2.0, 1.0])
    assert r.project_tube(p).distance == pytest.approx(0.0)
    active = r.project_tube(p, seg_index=0, segment_window=0)
    assert active.seg_index == 0
    assert active.distance == pytest.approx(1.0)


def test_projection_clamps_to_segment_ends():
    r = L_route()
    pr = r.project_active(np.array([-3.0, -2.0, 0.5]), 0)
    assert pr.t == 0.0 and pr.s == 0.0
    pr = r.project_active(np.array([99.0, -2.0, 0.0]), 2)
    assert pr.t == 1.0 and pr.s == pytest.approx(r.length)


def test_active_vs_full_projection():
    r = L_route()
    p = np.array([9.0, -2.0, 0.2])             # over segment 2
    full = r.project(p)
    act = r.project_active(p, 0)               # forced to segment 0 -> clamped
    assert full.seg_index == 2
    assert act.seg_index == 0 and act.t == 1.0
    assert act.cross_track > full.cross_track


def test_point_at_s_and_fixed_lookahead_target():
    r = L_route()
    pt, i = r.point_at_s(5.0)
    assert i == 1 and pt[0] == pytest.approx(5.0)
    # lookahead target from s=5 with L=2 -> s=7
    tgt, _ = r.point_at_s(5.0 + 2.0)
    assert tgt[0] == pytest.approx(7.0)
    # clamped at route end
    tgt, _ = r.point_at_s(1e9)
    assert tgt[0] == pytest.approx(12.0)


def test_adaptive_lookahead_clamping():
    assert adaptive_lookahead(0.0) == pytest.approx(0.8)
    assert adaptive_lookahead(1.0) == pytest.approx(0.8 + 0.5)
    assert adaptive_lookahead(100.0) == pytest.approx(3.0)   # clamps high
    assert adaptive_lookahead(0.0, base=0.1, lo=0.5) == pytest.approx(0.5)  # clamps low
    # farther from route -> larger lookahead (monotone)
    assert adaptive_lookahead(2.0) > adaptive_lookahead(0.5)


def test_vertical_height_interpolation():
    r = RouteModel([np.array([0.0, -2.0, 0.0]), np.array([4.0, -4.0, 0.0])])
    pr = r.project(np.array([2.0, -1.0, 0.3]))
    assert pr.t == pytest.approx(0.5)
    assert pr.target_y == pytest.approx(-3.0)  # halfway between -2 and -4
    # vertical error is separate from horizontal cross-track
    assert pr.cross_track == pytest.approx(0.3)


def test_slope_and_steep_segment_warning():
    r = RouteModel([np.array([0.0, -2.0, 0.0]),
                    np.array([0.5, -5.0, 0.0]),      # 3 m climb over 0.5 m horiz
                    np.array([4.5, -5.0, 0.0])])
    v = r.validate()
    assert v[0]["steep"] and "near-vertical" in v[0]["warning"]
    assert v[0]["height_change_up"] == pytest.approx(3.0)
    assert v[0]["slope_ratio"] == pytest.approx(6.0)
    assert not v[1]["steep"]
    assert v[1]["height_change_up"] == pytest.approx(0.0)


def test_up_of_frame_convention():
    assert up_of(np.array([0.0, -2.0, 0.0])) == pytest.approx(2.0)  # up = -y


def test_build_route_patterns_and_height():
    for pat in ("line", "square", "s_curve"):
        wps = build_route(pat, np.zeros(3), seg_len=4.0)
        assert len(wps) >= 2
    wps = build_route("line", np.zeros(3), seg_len=4.0, height_amp=2.0)
    ups = [up_of(w) for w in wps]
    assert max(ups) == pytest.approx(2.0) and ups[0] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        build_route("nope", np.zeros(3))


def test_random_start_sampling_within_radius_and_quadrants():
    rng = np.random.default_rng(0)
    head = math.radians(30.0)
    seen = set()
    for q in range(4):
        for _ in range(50):
            off, rel, r = sample_start_offset(rng, 5.0, head, quadrant=q)
            assert 0.75 <= r <= 5.0 + 1e-9
            assert abs(np.linalg.norm(horiz(off)) - r) < 1e-9
            assert off[1] == 0.0                       # horizontal offset only
            # relative bearing falls in the quadrant's 90-degree sector
            assert abs(wrap_angle(rel - q * math.pi / 2)) <= math.pi / 4 + 1e-9
            seen.add(q)
    assert seen == {0, 1, 2, 3}                        # front/side/back covered


def test_heading_of():
    assert heading_of(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert heading_of(np.array([0.0, 0.0, 1.0])) == pytest.approx(math.pi / 2)
