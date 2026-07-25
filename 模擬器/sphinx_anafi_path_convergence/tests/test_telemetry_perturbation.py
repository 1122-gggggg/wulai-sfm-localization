import math

import numpy as np
import pytest

from controllers import ExpPose
from route_geometry import wrap_angle
from telemetry_sources import (HeadingEstimator, KinematicAnafi, PerturbationConfig,
                               PerturbedPoseSource)


class StillSource:
    """Fixed pose source for wrapper tests."""

    def __init__(self, x=1.0, y=-2.0, z=3.0, yaw=0.5):
        self.x, self.y, self.z, self._yaw = x, y, z, yaw

    def get_pose(self, now):
        return ExpPose(self.x, self.y, self.z, stamp=now)

    def yaw(self, now):
        return self._yaw


def test_clean_config_passthrough():
    src = PerturbedPoseSource(StillSource(), PerturbationConfig())
    p = src.get_pose(1.0)
    assert (p.x, p.y, p.z) == (1.0, -2.0, 3.0)
    assert src.yaw(1.0) == pytest.approx(0.5)
    assert PerturbationConfig().label == "clean"
    assert not PerturbationConfig().any_active


def test_pose_quality_metadata_preserved_through_perturbation():
    class HlocSource(StillSource):
        def get_pose(self, now):
            return ExpPose(self.x, self.y, self.z, stamp=now, source_seq=7,
                           valid=True, match_count=42, inlier_ratio=0.7,
                           reprojection_error=1.2)

    src = PerturbedPoseSource(HlocSource(), PerturbationConfig(pose_noise_m=0.1, seed=9))
    p = src.get_pose(1.0)
    assert p.source_seq == 7
    assert p.match_count == 42
    assert p.inlier_ratio == pytest.approx(0.7)
    assert p.reprojection_error == pytest.approx(1.2)


def test_pose_noise_applied_on_all_axes():
    cfg = PerturbationConfig(pose_noise_m=0.2, seed=1)
    src = PerturbedPoseSource(StillSource(), cfg)
    xs, ys, zs = [], [], []
    for i in range(300):
        p = src.get_pose(float(i))
        xs.append(p.x); ys.append(p.y); zs.append(p.z)
    assert 0.1 < np.std(xs) < 0.35
    assert 0.1 < np.std(ys) < 0.35              # height noise too
    assert 0.1 < np.std(zs) < 0.35


def test_pose_noise_max_bounds_3d_resultant_error():
    cfg = PerturbationConfig(pose_noise_max_m=1.0, seed=12)
    src = PerturbedPoseSource(StillSource(), cfg)
    errs = []
    for i in range(400):
        p = src.get_pose(float(i))
        errs.append(math.sqrt((p.x - 1.0) ** 2 + (p.y + 2.0) ** 2 + (p.z - 3.0) ** 2))
    assert max(errs) <= 1.0 + 1e-9
    assert max(errs) > 0.5
    assert PerturbationConfig(pose_noise_max_m=1.0).label == "boundednoise1m"


def test_gaussian_pose_noise_is_clamped_by_max_error():
    cfg = PerturbationConfig(pose_noise_m=2.0, pose_noise_max_m=1.0, seed=13)
    src = PerturbedPoseSource(StillSource(), cfg)
    errs = []
    for i in range(400):
        p = src.get_pose(float(i))
        errs.append(math.sqrt((p.x - 1.0) ** 2 + (p.y + 2.0) ** 2 + (p.z - 3.0) ** 2))
    assert max(errs) <= 1.0 + 1e-9
    assert max(errs) == pytest.approx(1.0)


def test_yaw_bias_and_noise():
    src = PerturbedPoseSource(StillSource(yaw=0.0),
                              PerturbationConfig(yaw_bias_deg=10.0, seed=2))
    assert src.yaw(0.0) == pytest.approx(math.radians(10.0))
    noisy = PerturbedPoseSource(StillSource(yaw=0.0),
                                PerturbationConfig(yaw_noise_deg=5.0, seed=3))
    ys = [noisy.yaw(float(i)) for i in range(300)]
    assert 0.04 < np.std(ys) < 0.15


def test_yaw_noise_max_bounds_heading_error():
    src = PerturbedPoseSource(StillSource(yaw=0.2),
                              PerturbationConfig(yaw_noise_deg=60.0,
                                                 yaw_noise_max_deg=20.0,
                                                 seed=14))
    errs = [abs(wrap_angle(src.yaw(float(i)) - 0.2)) for i in range(400)]
    assert max(errs) <= math.radians(20.0) + 1e-12
    assert max(errs) == pytest.approx(math.radians(20.0))
    assert PerturbationConfig(yaw_noise_max_deg=20.0).label == "boundedyawnoise20deg"


def test_telemetry_delay():
    class MovingSource(StillSource):
        def get_pose(self, now):
            return ExpPose(now, -2.0, 0.0, stamp=now)   # x == time

        def yaw(self, now):
            return now

    src = PerturbedPoseSource(MovingSource(), PerturbationConfig(telemetry_delay_ms=200))
    src.get_pose(0.0)
    assert src.yaw(0.1) is None
    p = src.get_pose(0.1)                        # 0.0 sample not yet deliverable
    assert p is None or p.x <= 0.0
    p = src.get_pose(0.25)                       # now the 0.0 sample arrives
    assert p is not None and p.x <= 0.05 + 1e-9
    assert p.stamp <= 0.05 + 1e-9                # stamp keeps true age visible
    assert src.yaw(0.25) <= 0.05 + 1e-9


def test_periodic_hloc_outage_keeps_last_pose_until_stale():
    class MovingSource(StillSource):
        def get_pose(self, now):
            return ExpPose(now, -2.0, 0.0, stamp=now)

    src = PerturbedPoseSource(MovingSource(), PerturbationConfig(
        hloc_outage_interval_s=1.0, hloc_outage_duration_s=0.2))
    assert src.get_pose(0.05) is None             # first outage: no hloc result
    p = src.get_pose(0.25)
    assert p is not None and p.stamp == pytest.approx(0.25)
    p = src.get_pose(0.95)
    assert p.stamp == pytest.approx(0.95)
    p = src.get_pose(1.05)                        # next outage: no new result
    assert p.stamp == pytest.approx(0.95)


def test_dropout_loses_samples():
    class CountingSource(StillSource):
        def __init__(self):
            super().__init__()
            self.n = 0

        def get_pose(self, now):
            self.n += 1
            return ExpPose(float(self.n), -2.0, 0.0, stamp=now)

    src = PerturbedPoseSource(CountingSource(), PerturbationConfig(
        telemetry_drop_rate=0.5, seed=4))
    seen = {src.get_pose(float(i)).x for i in range(200) if src.get_pose(float(i))}
    # with 50% drop the delivered set must have gaps (repeats of last pose)
    assert len(seen) < 200


def test_deterministic_by_seed():
    a = PerturbedPoseSource(StillSource(), PerturbationConfig(pose_noise_m=0.1, seed=7))
    b = PerturbedPoseSource(StillSource(), PerturbationConfig(pose_noise_m=0.1, seed=7))
    pa = [a.get_pose(float(i)).x for i in range(20)]
    pb = [b.get_pose(float(i)).x for i in range(20)]
    assert pa == pb


def test_speed_noise():
    plant = KinematicAnafi(np.zeros(3), 0.0)
    src = PerturbedPoseSource(plant, PerturbationConfig(speed_noise_mps=0.5, seed=5))
    vs = [src.velocity_ned(float(i))[0] for i in range(200)]
    assert 0.3 < np.std(vs) < 0.8


# ---------------------------------------------------------------------------
# Kinematic plant sanity (signs match the PCMD conventions the controllers use)

def test_kinematic_pitch_moves_forward():
    k = KinematicAnafi(np.zeros(3), 0.0)         # facing +x
    k.send_pcmd(0, 20, 0, 0)
    for _ in range(100):
        k.step(0.05)
    assert k.pos[0] > 1.0 and abs(k.pos[2]) < 0.1


def test_kinematic_positive_yaw_turns_toward_east():
    k = KinematicAnafi(np.zeros(3), 0.0)
    k.send_pcmd(0, 0, 30, 0)
    for _ in range(40):
        k.step(0.05)
    assert k.yaw_v > 0.2                          # NED yaw increases


def test_kinematic_gaz_climbs():
    k = KinematicAnafi(np.array([0.0, -2.0, 0.0]), 0.0)
    k.send_pcmd(0, 0, 0, 30)
    for _ in range(60):
        k.step(0.05)
    assert k.pos[1] < -2.3                        # up = -y: climbing lowers y


def test_kinematic_roll_moves_right():
    k = KinematicAnafi(np.zeros(3), 0.0)          # facing +x; right = +z
    k.send_pcmd(20, 0, 0, 0)
    for _ in range(100):
        k.step(0.05)
    assert k.pos[2] > 1.0 and abs(k.pos[0]) < 0.1


def test_kinematic_pcmd_clamped():
    k = KinematicAnafi(np.zeros(3), 0.0)
    k.send_pcmd(500, -500, 101, -101)
    assert k._cmd == (100, -100, 100, -100)


# ---------------------------------------------------------------------------
# Heading estimator (fused yaw + offset) under bias

def test_heading_estimator_seed_and_motion_refinement():
    est = HeadingEstimator()
    bias = math.radians(20.0)                     # fused yaw reads 20 deg high
    true_heading = 0.0
    fused = true_heading + bias
    est.seed_from_path(np.zeros(3), np.array([4.0, 0.0, 0.0]), fused)
    # seeding assumes nose points at the goal -> offset == -bias
    assert est.offset == pytest.approx(-bias)
    assert est.heading(fused) == pytest.approx(true_heading, abs=1e-9)
    # drone actually moves along +x while fused yaw stays biased: offset holds
    for i in range(1, 20):
        est.update(np.array([0.2 * i, 0.0, 0.0]), fused)
    assert est.heading(fused) == pytest.approx(0.0, abs=0.05)


def test_heading_estimator_recovers_from_bad_seed():
    est = HeadingEstimator()
    fused = 0.0
    # bad seed: goal direction says +x but drone will actually move +z
    est.seed_from_path(np.zeros(3), np.array([4.0, 0.0, 0.0]), fused)
    for i in range(1, 60):
        est.update(np.array([0.0, 0.0, 0.1 * i]), fused)
    # offset converges toward pi/2 (motion heading) via EMA
    assert abs(wrap_angle(est.heading(fused) - math.pi / 2)) < 0.15


def test_heading_estimator_no_update_without_motion():
    est = HeadingEstimator()
    est.seed_from_path(np.zeros(3), np.array([1.0, 0.0, 0.0]), 0.0)
    off0 = est.offset
    for _ in range(10):
        est.update(np.array([0.001, 0.0, 0.001]), 0.3)   # sub-threshold jitter
    assert est.offset == off0
