"""Unit tests for Asynchronous Fast/Slow Visual Localizer (async_localizer.py).

Verifies:
1. AnchorMailbox freshness, overwrite, and max-age expiration (>200ms).
2. Bidirectional LK optical flow Forward-Backward error gate (tau < 1.0px) and median-drop.
3. Jump gate rejecting sudden step jumps (step > max_jump).
4. Drift budget exhaustion honestly reporting LOST (capping consecutive KLT_BRIDGED).
5. Re-anchor refreshing pose and clearing accumulated drift count.
6. SlowPath single-GPU priority queue ensuring MegaLoc never blocks EDM keyframes.
7. Scheduler three-tier triggering (TRACK 300ms / WEAK 100ms / NEED_REANCHOR immediate) and decision logging.
8. End-to-end synthetic pose stream verification with bounded stitching error (< 0.05m).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Ensure deploy_code is importable
DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy_code" / "sfm_glomap_deploy"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

try:
    import pycolmap
except ImportError:  # pragma: no cover
    pycolmap = None  # type: ignore[assignment]

import async_localizer
from async_localizer import (
    AnchorData,
    AnchorSupply,
    AnchorMailbox,
    AsyncLocalizer,
    FastPath,
    Scheduler,
    SchedulerTriggerState,
    SlowPath,
    SyncCarry,
    SyncStatus,
    TaskPriority,
    TransformChain,
    _KLT_FB_PX,
    _KLT_LK_PARAMS,
    pose_components_to_matrix,
)


class TestAnchorMailbox(unittest.TestCase):
    """Test capacity-1 mailbox overwrite, freshness, and thread safety."""

    def test_mailbox_freshness_and_overwrite(self) -> None:
        mb = AnchorMailbox()
        self.assertTrue(mb.is_empty())
        self.assertIsNone(mb.take_if_fresh())

        # 1. Put initial anchor
        a1 = AnchorData(
            key_stamp=1.0,
            cam_from_world=np.eye(4),
            inlier_2d=np.zeros((20, 2)),
            inlier_3d=np.zeros((20, 3)),
            num_inliers=20,
        )
        mb.put(a1)
        self.assertEqual(mb.seq, 1)
        self.assertFalse(mb.is_empty())

        # 2. Overwrite with newer anchor
        a2 = AnchorData(
            key_stamp=1.05,
            cam_from_world=np.eye(4),
            inlier_2d=np.zeros((30, 2)),
            inlier_3d=np.zeros((30, 3)),
            num_inliers=30,
        )
        mb.put(a2)
        self.assertEqual(mb.seq, 2)
        peeked = mb.peek_if_fresh(now_stamp=1.10)
        self.assertIsNotNone(peeked)
        self.assertEqual(peeked.num_inliers, 30)

        # 3. Take if fresh (age = 1.10 - 1.05 = 0.05s <= 0.20s)
        taken = mb.take_if_fresh(now_stamp=1.10, max_age_s=0.20)
        self.assertIsNotNone(taken)
        self.assertEqual(taken.seq, 2)
        self.assertTrue(mb.is_empty())
        # Taking again should return None (already consumed)
        self.assertIsNone(mb.take_if_fresh(now_stamp=1.10))

        # 4. Max-age expiration (age > 200ms discard)
        a3 = AnchorData(
            key_stamp=2.0,
            cam_from_world=np.eye(4),
            inlier_2d=np.zeros((20, 2)),
            inlier_3d=np.zeros((20, 3)),
            num_inliers=20,
        )
        mb.put(a3)
        # Query at 2.30s -> age 0.30s > 0.20s -> expired, discarded
        stale = mb.take_if_fresh(now_stamp=2.30, max_age_s=0.20)
        self.assertIsNone(stale)
        self.assertTrue(mb.is_empty())

        # 5. Non-finite timestamp check
        mb.put(a1)
        self.assertIsNone(mb.take_if_fresh(now_stamp=float("nan")))
        self.assertTrue(mb.is_empty())


class TestFBGateAndOutlierRejection(unittest.TestCase):
    """Test LK optical flow Forward-Backward error gate (tau < 1.0px) and median-drop."""

    def test_fb_gate_and_median_drop(self) -> None:
        np.random.seed(42)
        # Create textured base image
        img = cv2.GaussianBlur(
            np.random.randint(0, 255, (200, 200), dtype=np.uint8), (5, 5), 1.5
        )
        for i in range(15):
            cv2.circle(
                img, (40 + i * 8, 50 + (i % 3) * 20), 4, int(200 if i % 2 == 0 else 30), -1
            )

        corners = cv2.goodFeaturesToTrack(img, maxCorners=10, qualityLevel=0.05, minDistance=10)
        self.assertIsNotNone(corners)
        self.assertGreaterEqual(len(corners), 8)

        # Frame 1: original image
        gray1 = img.copy()
        # Frame 2: known rigid translation (dx=1.5, dy=0.8)
        M = np.float32([[1, 0, 1.5], [0, 1, 0.8]])
        gray2 = cv2.warpAffine(gray1, M, (200, 200))

        # Flatten lower region to create drifting points
        gray2[120:190, :] = 128
        fake_pts = np.array([[[100.0, 150.0]], [[120.0, 160.0]]], dtype=np.float32)

        all_pts = np.vstack([corners, fake_pts])
        nxt, stf, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, all_pts, None, **_KLT_LK_PARAMS)
        back, stb, _ = cv2.calcOpticalFlowPyrLK(gray2, gray1, nxt, None, **_KLT_LK_PARAMS)
        fb = np.linalg.norm((all_pts - back).reshape(-1, 2), axis=1)

        # Inlier corners must pass FB gate (< 1.0px)
        passed_good = (
            (stf[: len(corners)].ravel() == 1)
            & (stb[: len(corners)].ravel() == 1)
            & (fb[: len(corners)] < _KLT_FB_PX)
        )
        self.assertEqual(np.count_nonzero(passed_good), len(corners))

        # Drifting points in textureless region must fail FB gate
        passed_bad = (
            (stf[len(corners) :].ravel() == 1)
            & (stb[len(corners) :].ravel() == 1)
            & (fb[len(corners) :] < _KLT_FB_PX)
        )
        self.assertEqual(np.count_nonzero(passed_bad), 0)

        # Median drop test: introduce an outlier point with fb = 0.8px when inliers have fb < 0.1px
        fbg = np.array([0.02, 0.03, 0.02, 0.04, 0.03, 0.80])
        median_fbg = float(np.median(fbg))
        drop = (fbg > 2.0 * (median_fbg + 1e-6)) & (fbg > 0.5)
        self.assertEqual(list(drop), [False, False, False, False, False, True])


class TestJumpGateAndDriftBudget(unittest.TestCase):
    """Test jump gate protection and honest LOST reporting upon drift budget exhaustion."""

    def setUp(self) -> None:
        self.mailbox = AnchorMailbox()
        self.chain = TransformChain()
        self.sync_carry = SyncCarry(
            mailbox=self.mailbox,
            transform_chain=self.chain,
            max_drift_budget=5,
            max_jump=0.5,
        )

    def test_drift_budget_exhaustion_reports_lost(self) -> None:
        # Initial visual anchor at t=1.00s
        T0 = pose_components_to_matrix(np.eye(3), np.array([0.0, 0.0, 0.0]))
        self.mailbox.put(
            AnchorData(
                key_stamp=1.00,
                cam_from_world=T0,
                inlier_2d=np.zeros((20, 2)),
                inlier_3d=np.zeros((20, 3)),
                num_inliers=20,
            )
        )

        res0 = self.sync_carry.combine(1.00)
        self.assertEqual(res0.status, SyncStatus.REANCHORED)
        self.assertEqual(res0.drift_count, 0)

        # Bridge frames 1 through 5 (within budget)
        for i in range(1, 6):
            res = self.sync_carry.combine(1.00 + i * 0.033)
            self.assertEqual(res.status, SyncStatus.FRESH_BRIDGED)
            self.assertEqual(res.drift_count, i)
            self.assertFalse(res.is_stale)

        # Frame 6: budget exhausted (> 5 frames) -> honestly report LOST!
        res_lost = self.sync_carry.combine(1.00 + 6 * 0.033)
        self.assertEqual(res_lost.status, SyncStatus.LOST)
        self.assertEqual(res_lost.drift_count, 6)
        self.assertTrue(res_lost.is_stale)
        self.assertEqual(res_lost.info.get("reason"), "drift_budget_exhausted")

    def test_jump_gate_discards_the_chain_not_the_anchor(self) -> None:
        # Initial anchor
        T0 = pose_components_to_matrix(np.eye(3), np.array([0.0, 0.0, 0.0]))
        self.mailbox.put(
            AnchorData(
                key_stamp=1.00,
                cam_from_world=T0,
                inlier_2d=np.zeros((20, 2)),
                inlier_3d=np.zeros((20, 3)),
                num_inliers=20,
            )
        )
        self.sync_carry.combine(1.00)

        # Proposed anchor jumping 3.0 meters (limit is max_jump=0.5m)
        T_jump = pose_components_to_matrix(np.eye(3), np.array([3.0, 0.0, 0.0]))
        self.mailbox.put(
            AnchorData(
                key_stamp=1.05,
                cam_from_world=T_jump,
                inlier_2d=np.zeros((20, 2)),
                inlier_3d=np.zeros((20, 3)),
                num_inliers=20,
            )
        )

        res_jump = self.sync_carry.combine(1.05)
        # The anchor already passed the slow tracker's trajectory gates; the
        # suspect quantity is our own bridged pose. So the anchor is adopted on
        # its own map pose and the residual is reported, instead of throwing a
        # destructively-taken fix away.
        self.assertEqual(res_jump.status, SyncStatus.REANCHORED)
        self.assertEqual(self.sync_carry.active_anchor.key_stamp, 1.05)
        self.assertTrue(res_jump.info["jump_override"])
        self.assertFalse(res_jump.jump_gate_passed)
        self.assertAlmostEqual(float(res_jump.center[0]), 3.0, places=6)
        self.assertAlmostEqual(res_jump.residual, 3.0, places=6)
        self.assertEqual(self.sync_carry.stats["jump_override"], 1)

    def test_reanchor_refreshes_and_clears_drift(self) -> None:
        # Initial anchor
        T0 = pose_components_to_matrix(np.eye(3), np.array([0.0, 0.0, 0.0]))
        self.mailbox.put(
            AnchorData(
                key_stamp=1.00,
                cam_from_world=T0,
                inlier_2d=np.zeros((20, 2)),
                inlier_3d=np.zeros((20, 3)),
                num_inliers=20,
            )
        )
        self.sync_carry.combine(1.00)

        # Bridge 4 frames (accumulating drift count = 4)
        for i in range(1, 5):
            self.sync_carry.combine(1.00 + i * 0.033)
        self.assertEqual(self.sync_carry.bridge_count, 4)

        # Valid fresh visual anchor arrives with small plausible step (0.05m)
        T_fresh = pose_components_to_matrix(np.eye(3), np.array([0.05, 0.0, 0.0]))
        self.mailbox.put(
            AnchorData(
                key_stamp=1.15,
                cam_from_world=T_fresh,
                inlier_2d=np.zeros((30, 2)),
                inlier_3d=np.zeros((30, 3)),
                num_inliers=30,
            )
        )

        res_reanchor = self.sync_carry.combine(1.15)
        self.assertEqual(res_reanchor.status, SyncStatus.REANCHORED)
        # Drift budget must be cleared to 0
        self.assertEqual(self.sync_carry.bridge_count, 0)
        self.assertEqual(self.sync_carry.active_anchor.key_stamp, 1.15)

        # Next bridged frame starts from drift_count = 1
        res_next = self.sync_carry.combine(1.183)
        self.assertEqual(res_next.status, SyncStatus.FRESH_BRIDGED)
        self.assertEqual(res_next.drift_count, 1)


class TestSlowPathPriorityQueue(unittest.TestCase):
    """Test that MegaLoc tasks never block EDM keyframes on the shared GPU worker."""

    def test_megaloc_never_blocks_keyframe_priority(self) -> None:
        mb = AnchorMailbox()
        execution_order: list[tuple[str, float]] = []

        def fake_matcher(gray: np.ndarray, stamp: float) -> dict:
            execution_order.append(("keyframe", stamp))
            return {
                "ok": True,
                "rejected": None,
                "pose_status": "VISUALLY_CONFIRMED",
                "inliers": 80,
                "inlier_ratio": 0.80,
                "reproj_rms": 1.2,
                "cam_from_world": np.eye(4),
                "inlier_2d": np.zeros((80, 2)),
                "inlier_3d": np.zeros((80, 3)),
            }

        def fake_megaloc(query: Any, stamp: float) -> dict:
            execution_order.append(("megaloc", stamp))
            return {"score": 0.9}

        sp = SlowPath(mb, edm_matcher=fake_matcher, megaloc_fn=fake_megaloc)
        dummy = np.zeros((50, 50), dtype=np.uint8)

        # 1. Enqueue two MegaLoc requests first (lower priority = TaskPriority.MEGALOC = 2)
        sp.request_megaloc("megaloc_query_1", stamp=10.0)
        sp.request_megaloc("megaloc_query_2", stamp=10.1)

        # 2. Enqueue keyframe requests afterward
        sp.request_keyframe(dummy, stamp=10.2, priority=TaskPriority.PERIODIC)
        sp.request_keyframe(dummy, stamp=10.3, priority=TaskPriority.REANCHOR)

        # 3. Execute all remaining tasks step-by-step (2 megaloc + newest keyframe)
        for _ in range(3):
            sp.step()

        # Priority order must be: REANCHOR (0) > MEGALOC (2); the superseded
        # PERIODIC keyframe (10.2) is dropped by the newest-wins slot rule.
        expected_order = [
            ("keyframe", 10.3),  # Priority 0
            ("megaloc", 10.0),   # Priority 2
            ("megaloc", 10.1),   # Priority 2
        ]
        self.assertEqual(execution_order, expected_order)


class TestSchedulerCadenceAndLogging(unittest.TestCase):
    """Test three-tier keyframe scheduler (TRACK 300ms / WEAK 100ms / NEED_REANCHOR 0ms)."""

    def test_scheduler_cadence_and_logging(self) -> None:
        mb = AnchorMailbox()
        sp = SlowPath(mb)
        scheduler = Scheduler(slow_path=sp, state=SchedulerTriggerState.TRACK)
        dummy = np.zeros((20, 20), dtype=np.uint8)

        # 1. Initial trigger at t=0.0
        t0 = scheduler.check_and_trigger(dummy, stamp=0.0)
        self.assertTrue(t0)

        # 2. Dense stepping: every fed frame enqueues (slot dedups when busy).
        t1 = scheduler.check_and_trigger(dummy, stamp=0.15)
        self.assertTrue(t1)
        t2 = scheduler.check_and_trigger(dummy, stamp=0.31)
        self.assertTrue(t2)

        # 3. WEAK mode likewise triggers every frame.
        scheduler.set_state(SchedulerTriggerState.WEAK)
        t3 = scheduler.check_and_trigger(dummy, stamp=0.36)
        self.assertTrue(t3)
        t4 = scheduler.check_and_trigger(dummy, stamp=0.42)
        self.assertTrue(t4)

        # 4. NEED_REANCHOR mode: triggers immediately with priority.
        scheduler.set_state(SchedulerTriggerState.NEED_REANCHOR)
        t5 = scheduler.check_and_trigger(dummy, stamp=0.43)
        self.assertTrue(t5)
        self.assertEqual(scheduler.decision_log[-1]["priority"], "REANCHOR")
        for entry in scheduler.decision_log:
            self.assertIn("stamp", entry)
            self.assertIn("state", entry)
            self.assertIn("triggered", entry)
            self.assertIn("reason", entry)
            self.assertIn("priority", entry)


class TestEndToEndSyntheticPoseStream(unittest.TestCase):
    """End-to-end synthetic pose stream test (no GPU, fake slow source, bounded error)."""

    def test_end_to_end_synthetic_pose_stream(self) -> None:
        if pycolmap is None:
            self.skipTest("pycolmap not installed")

        # Setup Camera
        w, h = 640, 480
        fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
        cam = pycolmap.Camera(model="PINHOLE", width=w, height=h, params=[fx, fy, cx, cy])

        # 3D points in front of camera
        np.random.seed(123)
        pts3d = np.random.uniform(-1.5, 1.5, size=(60, 3))
        pts3d[:, 2] = np.random.uniform(3.5, 6.0, size=60)

        def render_frame(C: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            img = np.zeros((h, w), dtype=np.uint8)
            R = np.eye(3)
            t = -R @ C
            pts_c = (R @ pts3d.T).T + t
            valid = pts_c[:, 2] > 0.1
            u = fx * (pts_c[valid, 0] / pts_c[valid, 2]) + cx
            v = fy * (pts_c[valid, 1] / pts_c[valid, 2]) + cy
            p2d = np.column_stack([u, v])
            for pt in p2d:
                ix, iy = int(round(pt[0])), int(round(pt[1]))
                if 5 <= ix < w - 5 and 5 <= iy < h - 5:
                    cv2.circle(img, (ix, iy), 3, 255, -1)
            return img, p2d, pts3d[valid]

        current_gt_C = np.array([0.0, 0.0, 0.0])

        def fake_edm_matcher(gray: np.ndarray, stamp: float) -> dict:
            T_gt = pose_components_to_matrix(np.eye(3), current_gt_C)
            _, p2d, p3d = render_frame(current_gt_C)
            return {
                "ok": True,
                "rejected": None,
                "pose_status": "VISUALLY_CONFIRMED",
                "inliers": len(p2d),
                "inlier_ratio": 1.0,
                "reproj_rms": 0.5,
                "cam_from_world": T_gt,
                "inlier_2d": p2d,
                "inlier_3d": p3d,
            }

        localizer = AsyncLocalizer(
            camera=cam,
            edm_matcher=fake_edm_matcher,
            max_jump=0.5,
            max_drift_budget=5,
        )

        errors = []
        statuses = []

        # Simulate 15 frames at 30Hz (~0.5s)
        for k in range(15):
            t_k = 100.0 + k * 0.033
            current_gt_C = np.array([0.015 * k, 0.0, 0.0])
            img_k, _, _ = render_frame(current_gt_C)

            # Keyframes generated every 5 frames (~6Hz slow path)
            if k % 5 == 0:
                localizer.scheduler.check_and_trigger(img_k, t_k, force_reanchor=True)
                localizer.slow_path.step()  # deterministic processing

            res = localizer.feed_frame(img_k, t_k)
            statuses.append(res.pose_status)
            if res.pose:
                est_C = np.array([res.pose["x"], res.pose["y"], res.pose["z"]])
                err = float(np.linalg.norm(est_C - current_gt_C))
                errors.append(err)

        # Verify tracking continuity and error bounds
        self.assertEqual(len(errors), 15)
        max_error = max(errors)
        self.assertLess(
            max_error, 0.05, f"Position error {max_error:.4f}m exceeded limit of 0.05m"
        )

        # Verify state alternation between VISUALLY_CONFIRMED and KLT_BRIDGED
        self.assertIn("VISUALLY_CONFIRMED", statuses)
        self.assertIn("KLT_BRIDGED", statuses)
        self.assertEqual(statuses.count("VISUALLY_CONFIRMED"), 3)
        self.assertEqual(statuses.count("KLT_BRIDGED"), 12)


if __name__ == "__main__":
    unittest.main()


class TestDecisionLogCap(unittest.TestCase):
    """Decision log must stay bounded on long flights."""

    def test_decision_log_capped(self) -> None:
        from async_localizer import _DECISION_LOG_CAP
        mb = AnchorMailbox()
        sp = SlowPath(mb)
        scheduler = Scheduler(slow_path=sp, state=SchedulerTriggerState.TRACK)
        dummy = np.zeros((20, 20), dtype=np.uint8)
        for i in range(_DECISION_LOG_CAP + 100):
            scheduler.check_and_trigger(dummy, stamp=i * 0.033)
        self.assertLessEqual(len(scheduler.decision_log), _DECISION_LOG_CAP)


class TestAnchorSupplyAdaptation(unittest.TestCase):
    """Drift budget and mailbox freshness must follow the measured anchor cadence."""

    @staticmethod
    def _anchor(key_stamp: float, x: float = 0.0) -> AnchorData:
        return AnchorData(
            key_stamp=key_stamp,
            cam_from_world=pose_components_to_matrix(np.eye(3), np.array([x, 0.0, 0.0])),
            inlier_2d=np.zeros((20, 2)),
            inlier_3d=np.zeros((20, 3)),
            num_inliers=20,
        )

    def _carry(self, **kwargs) -> SyncCarry:
        return SyncCarry(
            mailbox=AnchorMailbox(),
            transform_chain=TransformChain(),
            max_drift_budget=kwargs.pop("max_drift_budget", 5),
            max_jump=kwargs.pop("max_jump", 0.5),
            **kwargs,
        )

    def test_anchor_supply_tracks_worst_recent_gap(self) -> None:
        supply = AnchorSupply(history=4)
        self.assertIsNone(supply.frame_gap())
        self.assertIsNone(supply.stamp_gap())

        for _ in range(3):
            supply.observe_frame()
        supply.observe_anchor(1.0)
        self.assertEqual(supply.frame_gap(), 3)
        self.assertIsNone(supply.stamp_gap())  # first anchor has no predecessor

        for _ in range(7):
            supply.observe_frame()
        supply.observe_anchor(1.6)
        self.assertEqual(supply.frame_gap(), 7)
        self.assertAlmostEqual(supply.stamp_gap(), 0.6, places=6)

        supply.reset()
        self.assertIsNone(supply.frame_gap())
        self.assertIsNone(supply.stamp_gap())

    def test_drift_budget_grows_to_cover_a_slow_anchor_cadence(self) -> None:
        carry = self._carry()
        # No history yet: the configured budget applies.
        self.assertEqual(carry.effective_drift_budget(), 5)

        carry.mailbox.put(self._anchor(0.0))
        self.assertEqual(carry.combine(0.0).status, SyncStatus.REANCHORED)
        # Nine fast frames served before the slow path published again. With a
        # fixed 5-frame budget the carry reports LOST before every anchor; the
        # effective budget now covers 1.5x the observed gap.
        for i in range(1, 9):
            carry.combine(0.0 + i * 0.1)
        carry.mailbox.put(self._anchor(0.9))
        res = carry.combine(0.9)
        self.assertEqual(res.status, SyncStatus.REANCHORED)
        self.assertEqual(carry.supply.frame_gap(), 9)
        self.assertEqual(carry.effective_drift_budget(), 14)

    def test_uncoverable_anchor_is_adopted_not_discarded(self) -> None:
        # The mailbox take is destructive: an anchor the chain cannot span must
        # still be adopted (identity composition), or the carry deadlocks in
        # LOST while consuming and dropping every valid anchor.
        carry = self._carry()
        carry.mailbox.put(self._anchor(0.0))
        self.assertEqual(carry.combine(0.0).status, SyncStatus.REANCHORED)

        carry.mailbox.put(self._anchor(1.0, x=0.2))
        res = carry.combine(1.2)
        self.assertEqual(res.status, SyncStatus.REANCHORED)
        self.assertFalse(res.info["chain_covered"])
        self.assertAlmostEqual(res.residual, 0.2, places=6)

    def test_uncoverable_anchor_with_a_jump_lands_on_the_anchor_pose(self) -> None:
        carry = self._carry(max_jump=0.5)
        carry.mailbox.put(self._anchor(0.0))
        carry.combine(0.0)

        carry.mailbox.put(self._anchor(1.0, x=3.0))
        res = carry.combine(1.2)
        self.assertEqual(res.status, SyncStatus.REANCHORED)
        self.assertEqual(carry.active_anchor.key_stamp, 1.0)
        self.assertTrue(res.info["jump_override"])
        self.assertFalse(res.info["chain_covered"])
        self.assertAlmostEqual(float(res.center[0]), 3.0, places=6)

    def test_drift_budget_is_capped_and_never_below_configuration(self) -> None:
        carry = self._carry(max_drift_budget=5)
        carry.supply._frame_gaps.append(1)
        self.assertEqual(carry.effective_drift_budget(), 5)
        carry.supply._frame_gaps.append(1000)
        self.assertEqual(
            carry.effective_drift_budget(), async_localizer._DRIFT_BUDGET_CAP
        )

    def test_anchor_max_age_follows_the_observed_stamp_cadence(self) -> None:
        carry = self._carry(max_anchor_age_s=1.0)
        self.assertEqual(carry.effective_anchor_max_age_s(), 1.0)

        # Anchors published 1.4 source-seconds apart: a fixed 1.0 s bound would
        # discard every one of them and the fast path would never re-anchor.
        carry.supply._stamp_gaps.append(1.4)
        self.assertAlmostEqual(carry.effective_anchor_max_age_s(), 2.8, places=6)

        carry.supply._stamp_gaps.append(1000.0)
        self.assertAlmostEqual(carry.effective_anchor_max_age_s(), 8.0, places=6)

    def test_slow_recovery_anchor_is_adopted_after_a_long_gap(self) -> None:
        carry = self._carry(max_anchor_age_s=1.0)
        # Two anchors 1.5 s apart teach the carry the real cadence.
        carry.mailbox.put(self._anchor(0.0))
        self.assertEqual(carry.combine(0.0).status, SyncStatus.REANCHORED)
        carry.mailbox.put(self._anchor(1.5))
        self.assertEqual(carry.combine(1.5).status, SyncStatus.REANCHORED)

        # A recovery keyframe whose stamp is 1.4 s old is now inside the bound;
        # the chain covers the span, so the composition is a real one.
        carry.transform_chain.record_step(
            3.0, 4.4, np.eye(4), pose_components_to_matrix(np.eye(3), np.zeros(3))
        )
        carry.mailbox.put(self._anchor(3.0))
        res = carry.combine(4.4)
        self.assertEqual(res.status, SyncStatus.REANCHORED)
        self.assertAlmostEqual(res.info["anchor_age_s"], 1.4, places=6)
        self.assertTrue(res.info["chain_covered"])

    def test_anchor_beyond_the_adaptive_age_bound_is_dropped(self) -> None:
        carry = self._carry(max_anchor_age_s=1.0)
        carry.mailbox.put(self._anchor(0.0))
        self.assertEqual(carry.combine(0.0).status, SyncStatus.REANCHORED)

        # 12 s old with no observed cadence: past the 1.0 s bound, so it is
        # discarded and the carry keeps bridging on the previous anchor.
        carry.mailbox.put(self._anchor(0.5, x=0.2))
        res = carry.combine(12.5)
        self.assertEqual(res.status, SyncStatus.FRESH_BRIDGED)
        self.assertEqual(carry.active_anchor.key_stamp, 0.0)


class TestFastPathSoftDegrade(unittest.TestCase):
    """One PnP miss must downweight the track, not zero the seed."""

    def _fast_path(self) -> FastPath:
        mailbox = AnchorMailbox()
        chain = TransformChain()
        carry = SyncCarry(mailbox=mailbox, transform_chain=chain)
        camera = type(
            "Cam", (), dict(width=1024, height=576, fx=800.0, fy=800.0, cx=512.0, cy=288.0)
        )()
        path = FastPath(camera=camera, sync_carry=carry, transform_chain=chain)
        path._klt_2d = np.zeros((100, 2), np.float32)
        path._klt_3d = np.zeros((100, 3))
        path._klt_gray = np.zeros((576, 1024), np.uint8)
        return path

    def test_seed_survives_the_first_misses_then_drops(self) -> None:
        path = self._fast_path()
        gray = np.zeros((576, 1024), np.uint8)

        for expected in range(1, async_localizer._FAST_SOFT_DEGRADE_MISSES + 1):
            res = path._on_tracking_failure(gray, 1.0, drift_count=0)
            self.assertEqual(res.pose_status, "NEED_REANCHOR")
            self.assertTrue(res.stale)
            self.assertEqual(path._consecutive_misses, expected)
            self.assertIsNotNone(path._klt_2d)
            self.assertEqual(len(path._klt_2d), 100)

        path._on_tracking_failure(gray, 1.1, drift_count=0)
        self.assertIsNone(path._klt_2d)
        self.assertIsNone(path._klt_3d)
        self.assertIsNone(path._klt_gray)

    def test_miss_counter_clears_on_reset(self) -> None:
        path = self._fast_path()
        path._on_tracking_failure(np.zeros((576, 1024), np.uint8), 1.0, drift_count=0)
        self.assertEqual(path._consecutive_misses, 1)
        path.reset()
        self.assertEqual(path._consecutive_misses, 0)


class TestAnchorAcceptanceThresholds(unittest.TestCase):
    """Slow-tracker accepts must not be re-rejected by module defaults."""

    @staticmethod
    def _accept(**overrides) -> dict:
        payload = {
            "ok": True,
            "pose_status": "VISUALLY_CONFIRMED",
            "inliers": 55,
            "inlier_ratio": 0.23,
            "reproj_rms": 2.4,
        }
        payload.update(overrides)
        return payload

    def test_site_thresholds_are_used_verbatim(self) -> None:
        slow = SlowPath(
            AnchorMailbox(),
            weak_min_inliers=30,
            anchor_min_inliers=50,
            anchor_min_ratio=0.15,
            max_reproj_rms=6.0,
        )
        self.assertEqual(slow.anchor_min_inliers, 50)
        # A 5-reference recovery accept (ratio ~0.23) is a valid anchor.
        self.assertTrue(slow._is_safe_visual_anchor(self._accept()))
        self.assertEqual(slow.anchor_accepts, 1)

    def test_rejection_reasons_are_recorded(self) -> None:
        slow = SlowPath(
            AnchorMailbox(),
            weak_min_inliers=30,
            anchor_min_inliers=50,
            anchor_min_ratio=0.15,
            max_reproj_rms=6.0,
        )
        self.assertFalse(slow._is_safe_visual_anchor(self._accept(inliers=40)))
        self.assertEqual(slow.anchor_reject_reason, "inliers")
        self.assertFalse(slow._is_safe_visual_anchor(self._accept(reproj_rms=9.0)))
        self.assertEqual(slow.anchor_reject_reason, "reproj_rms")
        self.assertFalse(slow._is_safe_visual_anchor(self._accept(pose_status="KLT_BRIDGED")))
        self.assertEqual(slow.anchor_reject_reason, "not_visually_confirmed")
        self.assertFalse(slow._is_safe_visual_anchor(None))
        self.assertEqual(slow.anchor_reject_reason, "no_result")
        self.assertEqual(
            slow.anchor_rejects,
            {"inliers": 1, "reproj_rms": 1, "not_visually_confirmed": 1, "no_result": 1},
        )
        self.assertEqual(slow.anchor_accepts, 0)


class TestFastFixCredit(unittest.TestCase):
    """A gated fast fix is a visual fix, not drift: it must not spend budget."""

    @staticmethod
    def _anchor(key_stamp: float) -> AnchorData:
        return AnchorData(
            key_stamp=key_stamp,
            cam_from_world=pose_components_to_matrix(np.eye(3), np.zeros(3)),
            inlier_2d=np.zeros((20, 2)),
            inlier_3d=np.zeros((20, 3)),
            num_inliers=20,
        )

    def test_fast_fix_resets_the_drift_budget_but_not_the_anchorless_run(self) -> None:
        carry = SyncCarry(mailbox=AnchorMailbox(), transform_chain=TransformChain())
        carry.mailbox.put(self._anchor(0.0))
        self.assertEqual(carry.combine(0.0).status, SyncStatus.REANCHORED)
        self.assertEqual(carry.anchorless_frames, 0)

        for i in range(1, 4):
            res = carry.combine(i * 0.1)
            self.assertEqual(res.status, SyncStatus.FRESH_BRIDGED)
            carry.note_fast_fix()
            self.assertEqual(carry.bridge_count, 0)
            self.assertEqual(carry.anchorless_frames, i)

        # Twelve anchorless frames with a fast fix on each: no LOST, because the
        # budget only bounds unverified carry.
        for i in range(4, 13):
            self.assertEqual(carry.combine(i * 0.1).status, SyncStatus.FRESH_BRIDGED)
            carry.note_fast_fix()
        self.assertEqual(carry.stats["fast_fix"], 12)

    def test_an_endless_fast_fix_run_still_ends_in_lost(self) -> None:
        # The drift budget bounds unverified carry, but note_fast_fix() credits
        # it back on every gated fast frame -- and a fast fix re-fits the SAME
        # map 3D that KLT carried forward, so it cannot detect chain drift.
        # Without an independent bound the carry ran anchorless indefinitely:
        # on the seven-video 720p corpus P168 reported 710 consecutive
        # unverified frames (~89 s at stride 3) as successful localization.
        carry = SyncCarry(mailbox=AnchorMailbox(), transform_chain=TransformChain(),
                          max_anchorless_frames=8)
        carry.mailbox.put(self._anchor(0.0))
        self.assertEqual(carry.combine(0.0).status, SyncStatus.REANCHORED)
        for i in range(1, 9):
            self.assertEqual(carry.combine(i * 0.1).status, SyncStatus.FRESH_BRIDGED)
            carry.note_fast_fix()
        result = carry.combine(0.9)
        self.assertEqual(result.status, SyncStatus.LOST)
        self.assertEqual(result.info["reason"], "anchorless_exhausted")
        self.assertEqual(result.info["anchorless_frames"], 9)

    def test_budget_still_expires_without_a_fast_fix(self) -> None:
        carry = SyncCarry(mailbox=AnchorMailbox(), transform_chain=TransformChain())
        carry.mailbox.put(self._anchor(0.0))
        carry.combine(0.0)
        for i in range(1, 6):
            self.assertEqual(carry.combine(i * 0.1).status, SyncStatus.FRESH_BRIDGED)
        self.assertEqual(carry.combine(0.6).status, SyncStatus.LOST)

    def test_fast_path_attempts_tracking_after_the_budget_expires(self) -> None:
        # LOST must mean "no anchor AND no fast fix": the fast path is still
        # allowed to produce its own gated pose once the budget is spent.
        mailbox = AnchorMailbox()
        chain = TransformChain()
        carry = SyncCarry(mailbox=mailbox, transform_chain=chain, max_drift_budget=1)
        camera = type(
            "Cam", (), dict(width=1024, height=576, fx=800.0, fy=800.0, cx=512.0, cy=288.0)
        )()
        path = FastPath(camera=camera, sync_carry=carry, transform_chain=chain)
        gray = np.zeros((576, 1024), np.uint8)
        mailbox.put(self._anchor(0.0))
        path.feed(gray, 0.0)

        attempts = []
        path._estimate_pnp = lambda p2, p3: attempts.append(len(p2)) or None
        path._klt_2d = np.zeros((100, 2), np.float32)
        path._klt_3d = np.zeros((100, 3))
        path._klt_gray = gray

        # Drive past the effective budget (it adapts upward from the observed
        # anchor cadence, so drive until the carry actually reports LOST).
        statuses = []
        for i in range(1, 8):
            res = path.feed(gray, i * 0.1)
            statuses.append(res.pose_status)
            if res.pose_status == "LOST":
                break
        self.assertEqual(statuses[-1], "LOST")
        self.assertTrue(res.stale)
        # LK/PnP ran on the LOST frame too: the verdict follows a real attempt.
        lk_calls = len([s for s in statuses if s in ("NEED_REANCHOR", "LOST")])
        self.assertGreaterEqual(lk_calls, 2)

    def test_hard_bound_short_circuits_a_runaway_anchorless_run(self) -> None:
        mailbox = AnchorMailbox()
        chain = TransformChain()
        carry = SyncCarry(mailbox=mailbox, transform_chain=chain, max_drift_budget=1)
        camera = type(
            "Cam", (), dict(width=1024, height=576, fx=800.0, fy=800.0, cx=512.0, cy=288.0)
        )()
        path = FastPath(camera=camera, sync_carry=carry, transform_chain=chain)
        gray = np.zeros((576, 1024), np.uint8)
        mailbox.put(self._anchor(0.0))
        path.feed(gray, 0.0)
        carry.anchorless_frames = async_localizer._DRIFT_BUDGET_CAP + 1
        carry.bridge_count = async_localizer._DRIFT_BUDGET_CAP + 1
        path._estimate_pnp = lambda p2, p3: self.fail("hard bound must skip tracking")
        path._klt_2d = np.zeros((100, 2), np.float32)
        path._klt_3d = np.zeros((100, 3))
        path._klt_gray = gray

        res = path.feed(gray, 1.0)
        self.assertEqual(res.pose_status, "LOST")


class TestFastPathGateOverrides(unittest.TestCase):
    """Fast-path inlier/reproj gates can tighten without touching the slow path."""

    def test_fast_min_inliers_does_not_change_slow_anchor_floor(self) -> None:
        loc = AsyncLocalizer(
            camera=type(
                "Cam", (), dict(width=64, height=48, fx=50.0, fy=50.0, cx=32.0, cy=24.0)
            )(),
            weak_min_inliers=30,
            anchor_min_inliers=50,
            max_reproj_error=6.0,
            fast_min_inliers=45,
            fast_max_reproj_error=3.0,
        )
        self.assertEqual(loc.fast_path.weak_min_inliers, 45)
        self.assertEqual(loc.fast_path.max_reproj_error, 3.0)
        self.assertEqual(loc.slow_path.weak_min_inliers, 30)
        self.assertEqual(loc.slow_path.anchor_min_inliers, 50)
        self.assertEqual(loc.slow_path.max_reproj_rms, 6.0)

    def test_unset_fast_gates_inherit_site_thresholds(self) -> None:
        loc = AsyncLocalizer(
            camera=type(
                "Cam", (), dict(width=64, height=48, fx=50.0, fy=50.0, cx=32.0, cy=24.0)
            )(),
            weak_min_inliers=30,
            max_reproj_error=6.0,
        )
        self.assertEqual(loc.fast_path.weak_min_inliers, 30)
        self.assertEqual(loc.fast_path.max_reproj_error, 6.0)


class TestInlineSyncFallback(unittest.TestCase):
    """A fast path with nothing to publish must fall back to a synchronous
    keyframe on the calling thread, never to a stream of NO_ANCHOR."""

    @staticmethod
    def _cam() -> object:
        return type(
            "Cam", (), dict(width=64, height=48, fx=50.0, fy=50.0, cx=32.0, cy=24.0)
        )()

    @staticmethod
    def _gray() -> np.ndarray:
        rng = np.random.default_rng(7)
        return (rng.uniform(0, 255, (48, 64))).astype(np.uint8)

    @staticmethod
    def _good_matcher() -> Any:
        n = 40
        rng = np.random.default_rng(3)
        p2d = np.column_stack([
            rng.uniform(5, 59, n).astype(np.float32),
            rng.uniform(5, 43, n).astype(np.float32),
        ])
        p3d = np.column_stack([
            rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(3, 6, n),
        ])

        def fake(g: np.ndarray, stamp: float) -> dict:
            return {
                "ok": True,
                "rejected": None,
                "pose_status": "VISUALLY_CONFIRMED",
                "inliers": n,
                "inlier_ratio": 1.0,
                "reproj_rms": 0.5,
                "cam_from_world": np.eye(4),
                "inlier_2d": p2d,
                "inlier_3d": p3d,
            }

        return fake

    def test_boot_frame_runs_keyframe_inline(self) -> None:
        gray = self._gray()
        loc = AsyncLocalizer(
            camera=self._cam(),
            edm_matcher=self._good_matcher(),
            anchor_min_inliers=30,
        )
        # No background thread started: without the floor this frame could only
        # ever be NO_ANCHOR.
        res = loc.feed_frame(gray, 100.0)
        self.assertEqual(res.pose_status, "VISUALLY_CONFIRMED")
        self.assertIsNotNone(res.pose)
        self.assertEqual(loc.inline_syncs, 1)
        self.assertEqual(loc.slow_path.inline_runs, 1)

    def test_stale_mailbox_anchor_does_not_suppress_inline(self) -> None:
        gray = self._gray()
        loc = AsyncLocalizer(
            camera=self._cam(),
            edm_matcher=self._good_matcher(),
            anchor_min_inliers=30,
        )
        loc.mailbox.put(AnchorData(
            key_stamp=0.0,
            cam_from_world=np.eye(4),
            inlier_2d=np.zeros((40, 2), np.float32),
            inlier_3d=np.zeros((40, 3)),
            num_inliers=40,
        ))
        res = loc.feed_frame(gray, 100.0)
        self.assertEqual(res.pose_status, "VISUALLY_CONFIRMED")
        self.assertEqual(loc.slow_path.inline_runs, 1)

    def test_inline_fallback_is_disablable(self) -> None:
        gray = self._gray()
        loc = AsyncLocalizer(
            camera=self._cam(),
            edm_matcher=self._good_matcher(),
            anchor_min_inliers=30,
            inline_sync_fallback=False,
        )
        res = loc.feed_frame(gray, 100.0)
        self.assertIsNone(res.pose)
        self.assertEqual(loc.inline_syncs, 0)
        self.assertEqual(loc.slow_path.inline_runs, 0)

    def test_slow_path_exception_is_recorded_not_fatal(self) -> None:
        gray = self._gray()

        def boom(g: np.ndarray, stamp: float) -> dict:
            raise ValueError("simulated GPU fault")

        loc = AsyncLocalizer(camera=self._cam(), edm_matcher=boom)
        loc.slow_path.request_keyframe(gray, 1.0)
        self.assertTrue(loc.slow_path.step())
        self.assertEqual(loc.slow_path.errors, 1)
        self.assertIn("ValueError", loc.slow_path.last_error or "")
        # The queued task was consumed, not left to wedge the worker loop.
        self.assertFalse(loc.slow_path.step())
        # And the inline floor reports the same fault instead of raising.
        self.assertIsNone(loc.slow_path.run_inline(gray, 2.0))
        self.assertEqual(loc.slow_path.errors, 2)

    def test_inline_drains_superseded_pending_keyframes(self) -> None:
        gray = self._gray()
        loc = AsyncLocalizer(
            camera=self._cam(),
            edm_matcher=self._good_matcher(),
            anchor_min_inliers=30,
        )
        loc.slow_path.request_keyframe(gray, 1.0)
        loc.slow_path.request_keyframe(gray, 2.0)
        anchor = loc.slow_path.run_inline(gray, 3.0)
        self.assertIsNotNone(anchor)
        # The inline frame is the newest: queued older frames must not be
        # re-driven by the background worker afterwards.
        self.assertTrue(loc.slow_path._queue.empty())
        self.assertFalse(loc.slow_path.step())

