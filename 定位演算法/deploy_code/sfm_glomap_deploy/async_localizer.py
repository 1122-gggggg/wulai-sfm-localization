"""Asynchronous Fast/Slow Localizer with decoupled CPU tracking and GPU keyframe matching.

Architecture Blueprint (PipelineSys / RTX 5060 deployment):
- Fast Thread (30Hz, CPU only):
    - KLT 2D tracking (cv2.calcOpticalFlowPyrLK with bidirectional forward-backward
      error gate tau < 1.0px and median-drop outlier rejection).
    - pycolmap PnP pose estimation.
    - Jump gate protection.
    - Emits KLT_BRIDGED with candidate_mode="klt_fast".
    - Maintains relative transform chain T(cur<-prev).
    - When tracked < 20 or PnP fails: emits NEED_REANCHOR, marks pose stale, triggers re-anchor.
- Slow Thread (5~10Hz, GPU):
    - EDM feature extraction and matching.
    - Anchor threshold gating (inliers >= max(2*weak_min, 60), ratio >= 0.66, reproj_rms <= max).
    - Rejection of KLT_BRIDGED/PREDICTED_ONLY anchors (pose_guided/quality.py semantics).
    - Prioritized single-GPU task executor: MegaLoc global retrieval runs with lower
      priority and NEVER blocks EDM keyframes (priority preemption).
- AnchorMailbox:
    - Capacity-1 mailbox storing latest visual anchor.
    - Overwrite on put; take_if_fresh(max_age_s=0.2) discards stale keyframes (>200ms).
    - Thread-safe via threading.Lock.
- SyncCarry:
    - SE(3) composition: T(now<-M) = T(now<-key) @ T(key<-M).
    - Jump gate validation: residual must pass max_jump before updating active anchor.
    - On jump gate pass: switch active anchor and clear drift budget.
    - On jump gate failure: retain old anchor, do NOT clear drift budget.
    - Drift budget enforcement: consecutive bridge frames capped (default 5).
      When budget is exhausted, honestly reports LOST (no unbounded KLT_BRIDGED).
- Three-level Keyframe Scheduler:
    - TRACK: 300ms periodic interval.
    - WEAK: 100ms periodic interval.
    - NEED_REANCHOR: 0ms immediate trigger (re-anchor preempts periodic).
    - Full decision logging for record-replay determinism and multi-seed statistical analysis.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

import cv2
import numpy as np

try:
    import pycolmap
except ImportError:  # pragma: no cover
    pycolmap = None  # type: ignore[assignment]


# --- Default Constants & Thresholds ---
_KLT_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
_KLT_FB_PX: float = 1.0
_KLT_MIN_TRACK: int = 20
_DEFAULT_MAX_JUMP: float = 0.50  # meters
_DEFAULT_MAX_YAW_JUMP_DEG: float = 45.0  # degrees
_DEFAULT_MAX_DRIFT_BUDGET: int = 5  # consecutive bridged frames (floor; see AnchorSupply)
_DEFAULT_MAX_ANCHOR_AGE_S: float = 1.0  # slow BOOT/recovery latency cover; chain coverage gates composition
_DEFAULT_WEAK_MIN_INLIERS: int = 15
_DEFAULT_ANCHOR_MIN_INLIERS: int = 60
_DEFAULT_ANCHOR_MIN_RATIO: float = 0.66
_DEFAULT_MAX_REPROJ_ERROR: float = 4.0  # px
_DEFAULT_GRID_SIZE: int = 8
# Anchor-supply adaptation. The fast path spends drift budget per frame and the
# mailbox expires anchors per stamp-second, but both are set by how fast the
# slow GPU path can publish: on the 2026-09-04 production-path gate one anchor
# took ~25 ms of wall time, i.e. ~6 fast frames and (at stride 3) ~0.6 s of
# source-stamp time, so a fixed 5-frame budget was spent before every anchor
# arrived and recovery keyframes (145 ms) aged past the fixed 1.0 s mailbox
# bound. Both bounds now follow the measured cadence, with a floor at the
# configured value and a hard cap so a stalled slow path still reports LOST.
_ANCHOR_SUPPLY_HISTORY: int = 8
_DRIFT_BUDGET_MARGIN: float = 1.5
_DRIFT_BUDGET_CAP: int = 30
_ANCHOR_AGE_MARGIN: float = 2.0
_ANCHOR_AGE_CAP_FACTOR: float = 8.0
# Consecutive fast-path misses tolerated before the KLT seed is dropped. A
# single PnP miss used to zero a 100-point seed; the WEAK hysteresis in the
# sync tracker keeps the track and only downweights it, so does this.
_FAST_SOFT_DEGRADE_MISSES: int = 2
_SCHEDULER_INTERVAL_TRACK: float = 0.0  # every frame; slot dedups when busy
_SCHEDULER_INTERVAL_WEAK: float = 0.0  # every frame; slot dedups when busy
_SCHEDULER_INTERVAL_REANCHOR: float = 0.0  # immediate
_DECISION_LOG_CAP: int = 1024


# --- Enums & Contracts ---

class SyncStatus(str, Enum):
    """Synchronization status returned by SyncCarry and FastPath."""
    FRESH_BRIDGED = "FRESH_BRIDGED"
    REANCHORED = "REANCHORED"
    LOST = "LOST"
    NO_ANCHOR = "NO_ANCHOR"
    NEED_REANCHOR = "NEED_REANCHOR"


class SchedulerTriggerState(str, Enum):
    """Scheduler state determining keyframe trigger cadence."""
    TRACK = "TRACK"
    WEAK = "WEAK"
    NEED_REANCHOR = "NEED_REANCHOR"


class TaskPriority(int, Enum):
    """Execution priority for single-GPU slow path. Lower value = higher priority."""
    REANCHOR = 0
    PERIODIC = 1
    MEGALOC = 2


# --- Data Structures ---

@dataclass
class AnchorData:
    """Visual anchor produced by slow thread (EDM / Visual confirmation)."""
    key_stamp: float
    cam_from_world: np.ndarray  # 4x4 SE(3) float matrix
    inlier_2d: np.ndarray  # (N, 2)
    inlier_3d: np.ndarray  # (N, 3)
    num_inliers: int
    seq: int = 0
    reproj_rms: float | None = None
    inlier_ratio: float = 1.0
    inlier_grid_cells: int = 0
    gray: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cam_from_world = cam_from_world_to_matrix(self.cam_from_world)
        self.inlier_2d = np.asarray(self.inlier_2d, dtype=np.float32).reshape(-1, 2)
        self.inlier_3d = np.asarray(self.inlier_3d, dtype=np.float64).reshape(-1, 3)
        self.num_inliers = int(self.num_inliers)


@dataclass
class SyncCarryResult:
    """Result of combining anchor with relative transform chain."""
    status: SyncStatus
    pose: np.ndarray | None = None  # 4x4 SE(3) float matrix
    center: np.ndarray | None = None  # (3,)
    rotation: np.ndarray | None = None  # (3, 3)
    yaw: float | None = None
    drift_count: int = 0
    key_stamp: float | None = None
    residual: float | None = None
    jump_gate_passed: bool = False
    is_stale: bool = False
    anchor: AnchorData | None = None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalizerResult:
    """Result payload adhering to live_localizer_worker.py contract."""
    pose: dict[str, Any] | None
    pose_status: str  # "VISUALLY_CONFIRMED", "KLT_BRIDGED", "NEED_REANCHOR", "LOST", "NO_ANCHOR"
    candidate_mode: str | None  # "klt_fast", "track", "relocalize", etc.
    inliers: int
    tracked: int
    stale: bool
    stamp: float
    drift_count: int
    sync_status: str
    reproj_rms: float | None = None
    limited_jump: dict[str, Any] | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Convert to production worker contract dictionary."""
        return {
            "pose": self.pose,
            "pose_status": self.pose_status,
            "candidate_mode": self.candidate_mode,
            "inliers": self.inliers,
            "tracked": self.tracked,
            "stale": self.stale,
            "stamp": self.stamp,
            "drift_count": self.drift_count,
            "sync_status": self.sync_status,
            "reproj_rms": self.reproj_rms,
            "limited_jump": self.limited_jump,
            "info": self.info,
        }


# --- SE(3) Math Utilities ---

def angle_diff(a: float, b: float) -> float:
    """Smallest signed angle difference in radians [-pi, pi]."""
    diff = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return float(diff)


def cam_from_world_to_matrix(transform: Any) -> np.ndarray:
    """Convert Rigid3d or array-like to 4x4 float64 SE(3) matrix."""
    if isinstance(transform, np.ndarray):
        arr = np.asarray(transform, dtype=float)
        if arr.shape == (4, 4):
            return arr.copy()
        if arr.shape == (3, 4):
            mat = np.eye(4, dtype=float)
            mat[:3, :4] = arr
            return mat
    if hasattr(transform, "rotation") and hasattr(transform, "translation"):
        R = np.asarray(transform.rotation.matrix(), dtype=float)
        t = np.asarray(transform.translation, dtype=float).reshape(3)
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = R
        mat[:3, 3] = t
        return mat
    raise ValueError(f"Cannot convert transform of type {type(transform)} to 4x4 matrix")


def matrix_to_pose_components(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract rotation (3,3), camera center (3,), and yaw (radians) from 4x4 SE(3)."""
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    center = -R.T @ t
    forward = R.T @ np.array([0.0, 0.0, 1.0], dtype=float)
    yaw = float(math.atan2(forward[1], forward[0]))
    return R, center, yaw


def pose_components_to_matrix(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Construct 4x4 SE(3) matrix (cam_from_world) from rotation R and center C."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = -R @ C
    return T


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Closed-form exact inverse of 4x4 SE(3) matrix."""
    T_inv = np.eye(4, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    R_T = R.T
    T_inv[:3, :3] = R_T
    T_inv[:3, 3] = -R_T @ t
    return T_inv


def se3_multiply(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """Closed-form exact composition of two SE(3) matrices: T1 @ T2."""
    T = np.eye(4, dtype=float)
    R1 = T1[:3, :3]
    t1 = T1[:3, 3]
    R2 = T2[:3, :3]
    t2 = T2[:3, 3]
    T[:3, :3] = R1 @ R2
    T[:3, 3] = R1 @ t2 + t1
    return T


def compute_reprojection_metrics(
    cam_from_world_mat: np.ndarray,
    pts2d: np.ndarray,
    pts3d: np.ndarray,
    camera: Any,
    grid: int = _DEFAULT_GRID_SIZE,
) -> dict[str, Any]:
    """Compute reprojection RMS error, inlier ratio, and spatial coverage grid cells."""
    pts2d = np.asarray(pts2d, dtype=float)
    pts3d = np.asarray(pts3d, dtype=float)
    count = len(pts2d)
    if count == 0:
        return {"reproj_rms": None, "inlier_ratio": 0.0, "inlier_grid_cells": 0}

    R = cam_from_world_mat[:3, :3]
    t = cam_from_world_mat[:3, 3]
    pts_cam = (R @ pts3d.T).T + t
    valid = np.isfinite(pts_cam).all(axis=1) & (pts_cam[:, 2] > 1e-6)
    if not np.any(valid):
        return {"reproj_rms": None, "inlier_ratio": 0.0, "inlier_grid_cells": 0}

    observed = pts2d[valid]
    pts_cam_valid = pts_cam[valid]

    # Project via pycolmap camera or pinhole model
    if hasattr(camera, "img_from_cam"):
        projected = np.asarray(camera.img_from_cam(pts_cam_valid), dtype=float)
    else:
        fx = getattr(camera, "fx", 500.0)
        fy = getattr(camera, "fy", 500.0)
        cx = getattr(camera, "cx", 320.0)
        cy = getattr(camera, "cy", 240.0)
        u = fx * (pts_cam_valid[:, 0] / pts_cam_valid[:, 2]) + cx
        v = fy * (pts_cam_valid[:, 1] / pts_cam_valid[:, 2]) + cy
        projected = np.column_stack([u, v])

    valid_proj = np.isfinite(projected).all(axis=1)
    if not np.any(valid_proj):
        return {"reproj_rms": None, "inlier_ratio": 0.0, "inlier_grid_cells": 0}

    diff = projected[valid_proj] - observed[valid_proj]
    rms = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

    # Spatial coverage grid cells
    obs_val = observed[valid_proj]
    w = max(1.0, float(getattr(camera, "width", 640)))
    h = max(1.0, float(getattr(camera, "height", 480)))
    gx = np.clip((obs_val[:, 0] / w * grid).astype(int), 0, grid - 1)
    gy = np.clip((obs_val[:, 1] / h * grid).astype(int), 0, grid - 1)
    cells = len(set(zip(gx.tolist(), gy.tolist())))

    return {
        "reproj_rms": rms,
        "inlier_ratio": float(np.count_nonzero(valid_proj)) / float(count),
        "inlier_grid_cells": cells,
    }


# --- 1. AnchorMailbox ---

class AnchorMailbox:
    """Thread-safe capacity-1 mailbox retaining only the latest visual anchor.

    - put() overwrites existing anchor with incremented sequence number.
    - take_if_fresh(max_age_s=0.2) returns and clears slot if fresh;
      if keyframe age > max_age_s or invalid, discards and returns None.
    - Protected by threading.Lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slot: AnchorData | None = None
        self._seq: int = 0
        # Delivery accounting: an anchor overwritten before it was taken, or
        # expired on age, is a fix the slow path paid for and nobody used.
        self.stats: Counter[str] = Counter()

    def put(self, anchor: AnchorData) -> None:
        """Overwrite mailbox with new anchor."""
        with self._lock:
            self._seq += 1
            anchor.seq = self._seq
            if self._slot is not None:
                self.stats["overwritten"] += 1
            self._slot = anchor
            self.stats["put"] += 1

    def take_if_fresh(
        self, now_stamp: float | None = None, max_age_s: float = _DEFAULT_MAX_ANCHOR_AGE_S
    ) -> AnchorData | None:
        """Take anchor from mailbox if fresh. Discard and return None if expired."""
        with self._lock:
            if self._slot is None:
                return None
            if now_stamp is not None:
                age = float(now_stamp) - float(self._slot.key_stamp)
                if not math.isfinite(age) or age < 0.0 or age > max_age_s:
                    self._slot = None  # discard stale anchor
                    self.stats["expired"] += 1
                    return None
            res = self._slot
            self._slot = None  # consumed
            return res

    def peek_if_fresh(
        self, now_stamp: float | None = None, max_age_s: float = _DEFAULT_MAX_ANCHOR_AGE_S
    ) -> AnchorData | None:
        """Peek latest anchor without consuming, if fresh."""
        with self._lock:
            if self._slot is None:
                return None
            if now_stamp is not None:
                age = float(now_stamp) - float(self._slot.key_stamp)
                if not math.isfinite(age) or age < 0.0 or age > max_age_s:
                    return None
            return self._slot

    def clear(self) -> None:
        """Clear mailbox contents."""
        with self._lock:
            self._slot = None

    def is_empty(self) -> bool:
        """Return True if no anchor in mailbox."""
        with self._lock:
            return self._slot is None

    @property
    def seq(self) -> int:
        """Current monotonically increasing sequence number."""
        with self._lock:
            return self._seq


# --- TransformChain ---

class TransformChain:
    """Bounded history of relative transforms T(cur<-prev) and poses for sync-carry."""

    def __init__(self, max_history: int = 120) -> None:
        self.max_history = max_history
        self._lock = threading.Lock()
        # Sequence of (prev_stamp, cur_stamp, T_cur_prev)
        self._steps: list[tuple[float, float, np.ndarray]] = []
        # Stored map-frame poses: stamp -> T(cur<-M)
        self._poses: dict[float, np.ndarray] = {}

    def record_step(
        self,
        prev_stamp: float,
        cur_stamp: float,
        T_cur_prev: np.ndarray,
        pose_cur: np.ndarray | None = None,
    ) -> None:
        """Record relative step between frames and optional current pose."""
        if not math.isfinite(prev_stamp) or not math.isfinite(cur_stamp):
            return
        if not np.isfinite(T_cur_prev).all():
            return

        with self._lock:
            self._steps.append((float(prev_stamp), float(cur_stamp), T_cur_prev.copy()))
            if pose_cur is not None and np.isfinite(pose_cur).all():
                self._poses[float(cur_stamp)] = pose_cur.copy()

            if len(self._steps) > self.max_history:
                discard = self._steps.pop(0)
                self._poses.pop(discard[0], None)

    def get_relative_transform(
        self, from_stamp: float, to_stamp: float, tol_s: float = 0.005
    ) -> np.ndarray | None:
        """Compute relative transform T(to<-from) such that X_to = T(to<-from) @ X_from."""
        from_s = float(from_stamp)
        to_s = float(to_stamp)
        if abs(from_s - to_s) < 1e-6:
            return np.eye(4, dtype=float)

        with self._lock:
            # Route 1: Directly through stored map poses if available
            # T(to<-from) = T(to<-M) @ T(from<-M)^-1
            p_from = self._find_pose_near(from_s, tol_s)
            p_to = self._find_pose_near(to_s, tol_s)
            if p_from is not None and p_to is not None:
                T_from_inv = se3_inverse(p_from)
                T_rel = se3_multiply(p_to, T_from_inv)
                if np.isfinite(T_rel).all():
                    return T_rel

            # Route 2: Chain intermediate steps
            if from_s < to_s:
                return self._chain_forward(from_s, to_s, tol_s)
            else:
                inv_rel = self._chain_forward(to_s, from_s, tol_s)
                if inv_rel is not None:
                    return se3_inverse(inv_rel)
                return None

    def _find_pose_near(self, stamp: float, tol_s: float) -> np.ndarray | None:
        if stamp in self._poses:
            return self._poses[stamp]
        best_diff = float("inf")
        best_pose = None
        for s, p in self._poses.items():
            diff = abs(s - stamp)
            if diff <= tol_s and diff < best_diff:
                best_diff = diff
                best_pose = p
        return best_pose

    def _chain_forward(
        self, from_stamp: float, to_stamp: float, tol_s: float
    ) -> np.ndarray | None:
        if not self._steps:
            return None

        # Filter steps covering [from_stamp, to_stamp]
        accum = np.eye(4, dtype=float)
        cur_t = from_stamp
        matched_any = False

        for p_s, c_s, T in self._steps:
            if c_s < from_stamp - tol_s:
                continue
            if p_s > to_stamp + tol_s:
                break
            # Check continuity
            if abs(p_s - cur_t) <= tol_s:
                accum = se3_multiply(T, accum)
                cur_t = c_s
                matched_any = True
                if abs(cur_t - to_stamp) <= tol_s or cur_t >= to_stamp:
                    break

        if matched_any and (abs(cur_t - to_stamp) <= tol_s or cur_t >= to_stamp):
            return accum if np.isfinite(accum).all() else None
        return None

    def clear(self) -> None:
        """Clear transform chain history."""
        with self._lock:
            self._steps.clear()
            self._poses.clear()


# --- AnchorSupply ---

class AnchorSupply:
    """Measured slow-path anchor cadence, in fast frames and in stamp seconds.

    Single-threaded: only the fast thread touches it (inside SyncCarry.combine).
    ``frame_gap`` / ``stamp_gap`` return the worst recent inter-anchor gap,
    because the bound that matters is the one that has to survive the slowest
    anchor, not the average one.
    """

    def __init__(self, history: int = _ANCHOR_SUPPLY_HISTORY) -> None:
        self.history = max(1, int(history))
        self._frame_gaps: deque[int] = deque(maxlen=self.history)
        self._stamp_gaps: deque[float] = deque(maxlen=self.history)
        self._pending_frames: int = 0
        self._last_key_stamp: float | None = None

    def reset(self) -> None:
        self._frame_gaps.clear()
        self._stamp_gaps.clear()
        self._pending_frames = 0
        self._last_key_stamp = None

    def observe_frame(self) -> None:
        """Count one fast frame served without a fresh anchor."""
        self._pending_frames += 1

    def observe_anchor(self, key_stamp: float) -> None:
        """Record the gap that this adopted anchor closed."""
        if self._pending_frames > 0:
            self._frame_gaps.append(int(self._pending_frames))
        stamp = float(key_stamp)
        if (
            self._last_key_stamp is not None
            and math.isfinite(stamp)
            and math.isfinite(self._last_key_stamp)
        ):
            delta = stamp - self._last_key_stamp
            if delta > 0.0:
                self._stamp_gaps.append(float(delta))
        if math.isfinite(stamp):
            self._last_key_stamp = stamp
        self._pending_frames = 0

    def frame_gap(self) -> int | None:
        return max(self._frame_gaps) if self._frame_gaps else None

    def stamp_gap(self) -> float | None:
        return max(self._stamp_gaps) if self._stamp_gaps else None


# --- 4. SyncCarry ---

class SyncCarry:
    """Sync-carry pose composition T(now<-M) = T(now<-key) @ T(key<-M).

    - Residual passes jump gate: update anchor + clear drift budget.
    - Residual fails jump gate: retain old anchor, do NOT clear drift budget.
    - Drift budget: consecutive bridge frames capped. ``max_drift_budget`` is
      the floor; the effective cap follows the measured anchor cadence
      (AnchorSupply) up to _DRIFT_BUDGET_CAP frames, so a slow path that needs
      six frames per anchor does not report LOST between every anchor.
      When the effective budget is exhausted, honestly reports LOST.
    - Mailbox freshness follows the same measurement in stamp seconds.
    - Explicit tri-state output: FRESH_BRIDGED / REANCHORED / LOST.
    """

    def __init__(
        self,
        mailbox: AnchorMailbox,
        transform_chain: TransformChain,
        max_drift_budget: int = _DEFAULT_MAX_DRIFT_BUDGET,
        max_jump: float = _DEFAULT_MAX_JUMP,
        max_yaw_jump_deg: float = _DEFAULT_MAX_YAW_JUMP_DEG,
        max_anchor_age_s: float = _DEFAULT_MAX_ANCHOR_AGE_S,
        max_anchorless_frames: int = _DRIFT_BUDGET_CAP,
    ) -> None:
        self.mailbox = mailbox
        self.transform_chain = transform_chain
        self.max_drift_budget = max_drift_budget
        self.max_jump = max_jump
        self.max_yaw_jump_deg = max_yaw_jump_deg
        self.max_anchor_age_s = max_anchor_age_s
        self.max_anchorless_frames = max(1, int(max_anchorless_frames))

        self.active_anchor: AnchorData | None = None
        self.bridge_count: int = 0
        # Consecutive frames served without adopting a fresh anchor. The drift
        # budget is credited back by a gated fast fix, so this is the only hard
        # bound left on how long the carry may run anchorless.
        self.anchorless_frames: int = 0
        self.last_accepted_pose: np.ndarray | None = None
        self.last_accepted_stamp: float | None = None
        # Adoption accounting: an anchor the slow path produced but the carry
        # never adopts is a lost fix, so every outcome is counted.
        self.stats: Counter[str] = Counter()
        self.supply = AnchorSupply()

    def reset(self) -> None:
        """Drop active anchor and drift budget (mirror of worker reset_to_boot)."""
        self.active_anchor = None
        self.bridge_count = 0
        self.anchorless_frames = 0
        self.last_accepted_pose = None
        self.last_accepted_stamp = None
        self.stats.clear()
        self.supply.reset()

    def note_fast_fix(self) -> None:
        """Credit the drift budget for a fast-path fix that passed every gate.

        The budget exists to bound UNVERIFIED carry. A fast frame that produced
        its own PnP pose over map 3D, cleared the inlier / reprojection / jump
        gates and re-seeded the tracks is a visual fix, not dead reckoning, so
        it must not spend budget -- otherwise a slow path that is itself in a
        recovery valley forces LOST after five frames no matter how well the
        fast path is tracking.
        """
        self.stats["fast_fix"] += 1
        self.bridge_count = 0

    def effective_drift_budget(self) -> int:
        """Consecutive bridged frames allowed before reporting LOST."""
        gap = self.supply.frame_gap()
        if gap is None:
            return int(self.max_drift_budget)
        covered = int(math.ceil(_DRIFT_BUDGET_MARGIN * float(gap)))
        return int(min(_DRIFT_BUDGET_CAP, max(int(self.max_drift_budget), covered)))

    def effective_anchor_max_age_s(self) -> float:
        """Mailbox freshness bound, widened to cover the observed anchor cadence."""
        base = float(self.max_anchor_age_s)
        gap = self.supply.stamp_gap()
        if gap is None:
            return base
        return float(min(_ANCHOR_AGE_CAP_FACTOR * base, max(base, _ANCHOR_AGE_MARGIN * gap)))

    def combine(
        self, now_stamp: float, current_fast_pose: np.ndarray | None = None
    ) -> SyncCarryResult:
        """Execute sync-carry composition with mailbox check, jump gate, and drift budget."""
        now_s = float(now_stamp)
        self.supply.observe_frame()
        drift_budget = self.effective_drift_budget()
        new_anchor = self.mailbox.take_if_fresh(
            now_stamp=now_s, max_age_s=self.effective_anchor_max_age_s()
        )

        # 1. Attempt re-anchor if fresh anchor is available
        if new_anchor is not None:
            # Reacquisition semantics mirror ProductionEDMTracker: once the
            # drift budget is spent the carry is LOST and the jump gate cannot
            # apply across the gap, so a LOST carry adopts the anchor pose and
            # skips the gate -- exactly what the sync tracker's LOST
            # reacquisition does.
            reacquiring = self.bridge_count > drift_budget
            T_now_key = self.transform_chain.get_relative_transform(new_anchor.key_stamp, now_s)
            chain_covered = T_now_key is not None
            if not chain_covered:
                # The chain cannot span [key_stamp, now]: no fast poses were
                # recorded (BOOT, dropped seed, or a miss run). Bridging would
                # carry a stale pose, so adopt the anchor pose directly -- the
                # lag error is bounded by the motion during the anchor age and
                # is corrected on the next anchor, and the jump gate below
                # still vets the result against the last accepted pose.
                # Discarding it instead deadlocked the carry: the mailbox take
                # is destructive, so every anchor was consumed and thrown away
                # until the drift budget expired.
                T_now_key = np.eye(4, dtype=float)
            if np.isfinite(T_now_key).all():
                T_prop = se3_multiply(T_now_key, new_anchor.cam_from_world)
                if np.isfinite(T_prop).all():
                    R_prop, C_prop, yaw_prop = matrix_to_pose_components(T_prop)

                    # Jump gate verification. The gate exists to catch a bad
                    # COMPOSITION, not to re-litigate the visual fix: the anchor
                    # already passed the slow tracker's inlier / reprojection /
                    # continuous-trajectory / acquire-jump / yaw / stale gates in
                    # map frame, whereas the pose it is compared against is our
                    # own bridged pose. So a failed gate discards the CHAIN and
                    # re-anchors on the anchor's own map pose, instead of keeping
                    # a drifted bridge and throwing the fix away (the mailbox
                    # take is destructive, so a discarded anchor is a lost fix:
                    # the 2026-09-04 diagnostic run produced 606 anchors but
                    # delivered only 561 VISUALLY_CONFIRMED frames).
                    jump_override = False
                    if self.last_accepted_pose is not None and not reacquiring:
                        _, C_last, yaw_last = matrix_to_pose_components(self.last_accepted_pose)
                        step = float(np.linalg.norm(C_prop - C_last))
                        yaw_diff_deg = float(math.degrees(abs(angle_diff(yaw_prop, yaw_last))))
                        if step > self.max_jump or yaw_diff_deg > self.max_yaw_jump_deg:
                            jump_override = True
                            T_prop = new_anchor.cam_from_world
                            R_prop, C_prop, yaw_prop = matrix_to_pose_components(T_prop)
                    else:
                        step = 0.0
                        yaw_diff_deg = 0.0
                        # Initial anchor and LOST reacquisition are always accepted:
                        # the visual anchor already passed the slow tracker's full
                        # inlier / reprojection / trajectory gates.

                    # Re-anchor: adopt anchor, reset drift budget
                    self.supply.observe_anchor(new_anchor.key_stamp)
                    self.stats["adopted"] += 1
                    if jump_override:
                        self.stats["jump_override"] += 1
                    if not chain_covered:
                        self.stats["chain_uncovered"] += 1
                    self.active_anchor = new_anchor
                    self.anchorless_frames = 0
                    self.bridge_count = 0
                    self.last_accepted_pose = T_prop
                    self.last_accepted_stamp = now_s
                    return SyncCarryResult(
                        status=SyncStatus.REANCHORED,
                        pose=T_prop,
                        center=C_prop,
                        rotation=R_prop,
                        yaw=yaw_prop,
                        drift_count=0,
                        key_stamp=new_anchor.key_stamp,
                        residual=step,
                        jump_gate_passed=not jump_override,
                        is_stale=False,
                        anchor=new_anchor,
                        info={
                            "jump_residual": step,
                            "yaw_diff_deg": yaw_diff_deg,
                            "reanchored": True,
                            "drift_budget": int(drift_budget),
                            "anchor_age_s": float(now_s - float(new_anchor.key_stamp)),
                            "chain_covered": bool(chain_covered),
                            "jump_override": jump_override,
                        },
                    )

        # 2. No re-anchor: bridge using active anchor or report NO_ANCHOR / LOST
        if self.active_anchor is None:
            return SyncCarryResult(
                status=SyncStatus.NO_ANCHOR,
                pose=None,
                drift_count=self.bridge_count,
                is_stale=True,
                info={"reason": "no_anchor_available"},
            )

        # Increment drift budget counter
        self.anchorless_frames += 1
        self.bridge_count += 1
        anchorless_exhausted = self.anchorless_frames > self.max_anchorless_frames
        if self.bridge_count > drift_budget or anchorless_exhausted:
            # Drift budget exhausted: honestly report LOST.
            #
            # anchorless_frames is the backstop for note_fast_fix(): a fast PnP
            # re-fits the SAME map 3D points that KLT carried forward, so its
            # inlier and reprojection numbers measure the chain's internal
            # consistency, not agreement with the map. Crediting the drift
            # budget on every such frame made the budget unreachable, and the
            # only hard bound on anchorless carry sat inside the branch that
            # bound was supposed to backstop. Measured on the seven-video 720p
            # corpus that let P168 report 710 consecutive unverified frames
            # (~89 s at stride 3) as successful localization.
            return SyncCarryResult(
                status=SyncStatus.LOST,
                pose=self.last_accepted_pose,
                center=matrix_to_pose_components(self.last_accepted_pose)[1]
                if self.last_accepted_pose is not None
                else None,
                rotation=matrix_to_pose_components(self.last_accepted_pose)[0]
                if self.last_accepted_pose is not None
                else None,
                yaw=matrix_to_pose_components(self.last_accepted_pose)[2]
                if self.last_accepted_pose is not None
                else None,
                drift_count=self.bridge_count,
                key_stamp=self.active_anchor.key_stamp,
                is_stale=True,
                anchor=self.active_anchor,
                info={
                    "reason": ("anchorless_exhausted" if anchorless_exhausted
                               else "drift_budget_exhausted"),
                    "anchorless_frames": self.anchorless_frames,
                    "max_anchorless_frames": self.max_anchorless_frames,
                    "bridge_count": self.bridge_count,
                    "max_budget": int(drift_budget),
                    "configured_budget": int(self.max_drift_budget),
                    "anchor_frame_gap": self.supply.frame_gap(),
                },
            )

        # Within drift budget: calculate bridged pose
        if current_fast_pose is not None and np.isfinite(current_fast_pose).all():
            T_bridged = current_fast_pose
        else:
            T_now_key = self.transform_chain.get_relative_transform(
                self.active_anchor.key_stamp, now_s
            )
            if T_now_key is not None and np.isfinite(T_now_key).all():
                T_bridged = se3_multiply(T_now_key, self.active_anchor.cam_from_world)
            else:
                T_bridged = self.last_accepted_pose

        if T_bridged is not None and np.isfinite(T_bridged).all():
            R, C, yaw = matrix_to_pose_components(T_bridged)
            self.last_accepted_pose = T_bridged
            self.last_accepted_stamp = now_s
            return SyncCarryResult(
                status=SyncStatus.FRESH_BRIDGED,
                pose=T_bridged,
                center=C,
                rotation=R,
                yaw=yaw,
                drift_count=self.bridge_count,
                key_stamp=self.active_anchor.key_stamp,
                residual=0.0,
                jump_gate_passed=True,
                is_stale=False,
                anchor=self.active_anchor,
                info={"bridge_count": self.bridge_count},
            )

        return SyncCarryResult(
            status=SyncStatus.LOST,
            pose=None,
            drift_count=self.bridge_count,
            is_stale=True,
            info={"reason": "invalid_bridged_pose"},
        )


# --- 2. FastPath ---

class FastPath:
    """Fast-frequency (30Hz) visual tracker operating exclusively on CPU.

    - LK tracking (cv2.calcOpticalFlowPyrLK + bidirectional FB < 1.0px + median drop).
    - pycolmap PnP pose estimation.
    - Jump gate verification.
    - Relative transform chain updates T(cur<-prev).
    - Emits KLT_BRIDGED / klt_fast.
    - tracked < 20 or PnP failure emits NEED_REANCHOR with stale last-known pose.
    """

    def __init__(
        self,
        camera: Any,
        sync_carry: SyncCarry,
        transform_chain: TransformChain,
        scheduler: Scheduler | None = None,
        max_jump: float = _DEFAULT_MAX_JUMP,
        weak_min_inliers: int = _DEFAULT_WEAK_MIN_INLIERS,
        max_reproj_error: float = _DEFAULT_MAX_REPROJ_ERROR,
        pnp_ransac_seed: int = 42,
    ) -> None:
        self.camera = camera
        self.sync_carry = sync_carry
        self.transform_chain = transform_chain
        self.scheduler = scheduler
        self.max_jump = max_jump
        self.weak_min_inliers = weak_min_inliers
        self.max_reproj_error = max_reproj_error
        self.pnp_ransac_seed = pnp_ransac_seed

        # Tracking state
        self._klt_2d: np.ndarray | None = None
        self._klt_3d: np.ndarray | None = None
        self._klt_gray: np.ndarray | None = None
        self._last_stamp: float | None = None
        self._last_known_pose: np.ndarray | None = None
        self._last_known_result: LocalizerResult | None = None
        # Consecutive tracking/PnP misses. One miss no longer zeroes the seed:
        # it survives _FAST_SOFT_DEGRADE_MISSES frames so LK can retry from the
        # last good frame (the sync tracker's WEAK hysteresis, fast-path side).
        self._consecutive_misses: int = 0

        # Setup pycolmap camera and options if available
        self._pcam = None
        self._pnp_options = None
        self._init_camera_and_pnp()

    def _init_camera_and_pnp(self) -> None:
        if pycolmap is None:
            return
        if isinstance(self.camera, pycolmap.Camera):
            self._pcam = self.camera
        else:
            w = int(getattr(self.camera, "width", 640))
            h = int(getattr(self.camera, "height", 480))
            fx = float(getattr(self.camera, "fx", 500.0))
            fy = float(getattr(self.camera, "fy", 500.0))
            cx = float(getattr(self.camera, "cx", 320.0))
            cy = float(getattr(self.camera, "cy", 240.0))
            self._pcam = pycolmap.Camera(
                model="PINHOLE", width=w, height=h, params=[fx, fy, cx, cy]
            )

        self._pnp_options = pycolmap.AbsolutePoseEstimationOptions()
        self._pnp_options.ransac.max_error = float(self.max_reproj_error)
        self._pnp_options.ransac.random_seed = int(self.pnp_ransac_seed)
        self._pnp_options.ransac.num_threads = 1  # deterministic CPU execution

    def reset(self) -> None:
        """Drop all fast tracking state (mirror of worker reset_to_boot)."""
        self._klt_2d = None
        self._klt_3d = None
        self._klt_gray = None
        self._last_stamp = None
        self._last_known_pose = None
        self._last_known_result = None
        self._consecutive_misses = 0

    def feed(self, gray: np.ndarray, stamp: float) -> LocalizerResult:
        """Process incoming video frame on the fast thread."""
        stamp = float(stamp)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = np.ascontiguousarray(gray, dtype=np.uint8)

        # 1. Sync-carry: check for fresh anchor or update drift budget
        sync_res = self.sync_carry.combine(stamp)

        # Case A: Re-anchored with fresh visual anchor
        if sync_res.status == SyncStatus.REANCHORED and sync_res.anchor is not None:
            anchor = sync_res.anchor
            self._seed_tracking_from_anchor(anchor, gray, stamp, sync_res.pose)
            self._consecutive_misses = 0
            R, C, yaw = matrix_to_pose_components(sync_res.pose)
            self._last_stamp = stamp
            self._last_known_pose = sync_res.pose

            res = LocalizerResult(
                pose=self._make_pose_dict(sync_res.pose, R, C, yaw, stamp),
                pose_status="VISUALLY_CONFIRMED",
                candidate_mode="track",
                inliers=anchor.num_inliers,
                tracked=len(anchor.inlier_2d),
                stale=False,
                stamp=stamp,
                drift_count=0,
                sync_status=sync_res.status.value,
                reproj_rms=anchor.reproj_rms,
            )
            self._last_known_result = res
            return res

        # Case B: No anchor available anywhere
        if sync_res.status == SyncStatus.NO_ANCHOR:
            return LocalizerResult(
                pose=None,
                pose_status="NO_ANCHOR",
                candidate_mode=None,
                inliers=0,
                tracked=0,
                stale=True,
                stamp=stamp,
                drift_count=0,
                sync_status=sync_res.status.value,
            )

        # Case C: the carry is out of drift budget. LOST means "no anchor AND no
        # fast fix", so the fast path still gets to try: its LK+PnP result faces
        # the same inlier / reprojection / jump gates the sync tracker's KLT
        # prior faces, and a fix that passes them is a visual fix, not drift.
        # Only a hard bound on consecutive anchorless frames short-circuits it.
        carry_lost = sync_res.status == SyncStatus.LOST
        if carry_lost:
            if self.scheduler:
                self.scheduler.check_and_trigger(gray, stamp, force_reanchor=True)
            if self.sync_carry.anchorless_frames > _DRIFT_BUDGET_CAP:
                return self._make_stale_result(
                    stamp,
                    pose_status="LOST",
                    sync_status=sync_res.status.value,
                    drift_count=sync_res.drift_count,
                )

        # 2. Tracking: If we lack tracking points, attempt re-seed or trigger re-anchor
        if (
            self._klt_2d is None
            or self._klt_3d is None
            or self._klt_gray is None
            or len(self._klt_2d) < _KLT_MIN_TRACK
        ):
            if self.sync_carry.active_anchor is not None:
                self._seed_tracking_from_anchor(
                    self.sync_carry.active_anchor, gray, stamp, self._last_known_pose
                )

        if (
            self._klt_2d is None
            or self._klt_3d is None
            or self._klt_gray is None
            or len(self._klt_2d) < _KLT_MIN_TRACK
        ):
            if self.scheduler:
                self.scheduler.check_and_trigger(gray, stamp, force_reanchor=True)
            return self._make_stale_result(
                stamp,
                pose_status="LOST" if carry_lost else "NEED_REANCHOR",
                sync_status=sync_res.status.value
                if carry_lost
                else SyncStatus.NEED_REANCHOR.value,
                drift_count=sync_res.drift_count,
            )

        # 3. Bidirectional KLT with FB gate and median drop
        p0 = np.asarray(self._klt_2d, dtype=np.float32).reshape(-1, 1, 2)
        nxt, stf, _ = cv2.calcOpticalFlowPyrLK(self._klt_gray, gray, p0, None, **_KLT_LK_PARAMS)
        if nxt is None or stf is None:
            return self._on_tracking_failure(gray, stamp, sync_res.drift_count, carry_lost)

        back, stb, _ = cv2.calcOpticalFlowPyrLK(gray, self._klt_gray, nxt, None, **_KLT_LK_PARAMS)
        if back is None or stb is None:
            return self._on_tracking_failure(gray, stamp, sync_res.drift_count, carry_lost)

        fb = np.linalg.norm((p0 - back).reshape(-1, 2), axis=1)
        good = (stf.ravel() == 1) & (stb.ravel() == 1) & (fb < _KLT_FB_PX)

        if int(np.count_nonzero(good)) < _KLT_MIN_TRACK:
            return self._on_tracking_failure(gray, stamp, sync_res.drift_count, carry_lost)

        n2d = nxt.reshape(-1, 2)[good]
        n3d = np.asarray(self._klt_3d, dtype=float)[good]
        fbg = fb[good]

        # Median drop outlier rejection
        try:
            median_fbg = float(np.median(fbg)) if len(fbg) else 0.0
        except Exception:
            median_fbg = 0.0
        drop = (fbg > 2.0 * (median_fbg + 1e-6)) & (fbg > 0.5)

        if bool(np.any(drop)) and int(np.count_nonzero(~drop)) >= _KLT_MIN_TRACK:
            s2d = n2d[~drop]
            s3d = n3d[~drop]
        else:
            s2d, s3d = n2d, n3d

        if s2d.shape[0] < 6:
            return self._on_tracking_failure(gray, stamp, sync_res.drift_count, carry_lost)

        # 4. PnP Pose Estimation
        estimate = self._estimate_pnp(s2d, s3d)
        if estimate is None:
            return self._on_tracking_failure(gray, stamp, sync_res.drift_count, carry_lost)

        inliers = int(estimate.get("num_inliers", 0))
        if inliers < self.weak_min_inliers:
            return self._on_tracking_failure(gray, stamp, sync_res.drift_count, carry_lost)

        T_cur = cam_from_world_to_matrix(estimate["cam_from_world"])
        if not np.isfinite(T_cur).all():
            return self._on_tracking_failure(gray, stamp, sync_res.drift_count, carry_lost)

        metrics = compute_reprojection_metrics(T_cur, s2d, s3d, self._pcam or self.camera)
        reproj_rms = metrics.get("reproj_rms")
        if reproj_rms is None or not math.isfinite(float(reproj_rms)) or float(reproj_rms) > self.max_reproj_error:
            return self._on_tracking_failure(gray, stamp, sync_res.drift_count, carry_lost)

        R_cur, C_cur, yaw_cur = matrix_to_pose_components(T_cur)

        # 5. Jump Gate
        if self._last_known_pose is not None:
            _, C_prev, _ = matrix_to_pose_components(self._last_known_pose)
            step = float(np.linalg.norm(C_cur - C_prev))
            if step > self.max_jump:
                if self.scheduler:
                    self.scheduler.check_and_trigger(gray, stamp, force_reanchor=True)
                return self._make_stale_result(
                    stamp,
                    pose_status="LOST" if carry_lost else "NEED_REANCHOR",
                    sync_status=sync_res.status.value
                    if carry_lost
                    else SyncStatus.NEED_REANCHOR.value,
                    drift_count=sync_res.drift_count,
                    limited_jump={"raw_step": step, "limit": self.max_jump},
                )

        # 6. Pose Accepted: update transform chain and tracking state
        if self._last_stamp is not None and self._last_known_pose is not None:
            T_prev_inv = se3_inverse(self._last_known_pose)
            T_cur_prev = se3_multiply(T_cur, T_prev_inv)
            self.transform_chain.record_step(self._last_stamp, stamp, T_cur_prev, T_cur)

        self._klt_2d = s2d.copy()
        self._klt_3d = s3d.copy()
        self._klt_gray = gray.copy()
        self._last_stamp = stamp
        self._last_known_pose = T_cur
        self._consecutive_misses = 0
        # This frame carried a fix that passed the inlier / reprojection / jump
        # gates, so it is not unverified drift: give the budget back.
        self.sync_carry.note_fast_fix()

        res = LocalizerResult(
            pose=self._make_pose_dict(T_cur, R_cur, C_cur, yaw_cur, stamp),
            pose_status="KLT_BRIDGED",
            candidate_mode="klt_fast",
            inliers=inliers,
            tracked=len(s2d),
            stale=False,
            stamp=stamp,
            drift_count=sync_res.drift_count,
            sync_status=sync_res.status.value,
            reproj_rms=reproj_rms,
        )
        self._last_known_result = res
        return res

    def _seed_tracking_from_anchor(
        self,
        anchor: AnchorData,
        gray: np.ndarray,
        stamp: float,
        pose_prop: np.ndarray | None = None,
    ) -> None:
        """Initialize or refresh KLT 2D/3D tracking points from visual anchor."""
        if abs(stamp - anchor.key_stamp) < 1e-4 and anchor.gray is not None:
            self._klt_2d = anchor.inlier_2d.copy()
            self._klt_3d = anchor.inlier_3d.copy()
            self._klt_gray = anchor.gray.copy()
            return

        # Reproject anchor 3D points to current frame using propagated pose
        T = pose_prop if pose_prop is not None else anchor.cam_from_world
        R = T[:3, :3]
        t = T[:3, 3]
        p3d = anchor.inlier_3d
        pts_cam = (R @ p3d.T).T + t
        valid_z = pts_cam[:, 2] > 0.1

        if not np.any(valid_z):
            self._klt_2d = anchor.inlier_2d.copy()
            self._klt_3d = anchor.inlier_3d.copy()
            self._klt_gray = gray.copy()
            return

        cam = self._pcam or self.camera
        if hasattr(cam, "img_from_cam"):
            proj = np.asarray(cam.img_from_cam(pts_cam[valid_z]), dtype=float)
        else:
            fx = getattr(cam, "fx", 500.0)
            fy = getattr(cam, "fy", 500.0)
            cx = getattr(cam, "cx", 320.0)
            cy = getattr(cam, "cy", 240.0)
            u = fx * (pts_cam[valid_z, 0] / pts_cam[valid_z, 2]) + cx
            v = fy * (pts_cam[valid_z, 1] / pts_cam[valid_z, 2]) + cy
            proj = np.column_stack([u, v])

        w = float(getattr(cam, "width", 640))
        h = float(getattr(cam, "height", 480))
        in_bounds = (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)

        if np.count_nonzero(in_bounds) >= _KLT_MIN_TRACK:
            self._klt_2d = proj[in_bounds].astype(np.float32)
            self._klt_3d = p3d[valid_z][in_bounds]
        else:
            self._klt_2d = anchor.inlier_2d.copy()
            self._klt_3d = anchor.inlier_3d.copy()

        self._klt_gray = gray.copy()

    def _estimate_pnp(self, pts2d: np.ndarray, pts3d: np.ndarray) -> dict[str, Any] | None:
        """Estimate camera pose using pycolmap or fallback solver."""
        if pycolmap is not None and self._pcam is not None:
            try:
                ret = pycolmap.estimate_and_refine_absolute_pose(
                    np.asarray(pts2d, dtype=float),
                    np.asarray(pts3d, dtype=float),
                    self._pcam,
                    self._pnp_options,
                )
                if ret is not None and "cam_from_world" in ret:
                    return ret
            except Exception:
                pass

        # Fallback to OpenCV solvePnPRansac
        try:
            cam = self._pcam or self.camera
            fx = float(getattr(cam, "fx", 500.0))
            fy = float(getattr(cam, "fy", 500.0))
            cx = float(getattr(cam, "cx", 320.0))
            cy = float(getattr(cam, "cy", 240.0))
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.asarray(pts3d, dtype=np.float32),
                np.asarray(pts2d, dtype=np.float32),
                K,
                None,
                reprojectionError=float(self.max_reproj_error),
                iterationsCount=100,
            )
            if not ok or inliers is None:
                return None
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4, dtype=float)
            T[:3, :3] = R
            T[:3, 3] = tvec.ravel()
            return {
                "cam_from_world": T,
                "num_inliers": len(inliers),
                "inlier_mask": np.isin(np.arange(len(pts2d)), inliers),
            }
        except Exception:
            return None

    def _on_tracking_failure(
        self, gray: np.ndarray, stamp: float, drift_count: int, carry_lost: bool = False
    ) -> LocalizerResult:
        """Soft-degrade a miss: keep the seed, only drop it after repeated misses.

        A single PnP miss used to null a ~100-point 2D/3D seed, so the next
        frame had nothing to track and could only wait for a slow anchor. The
        seed and its reference frame (``_klt_gray``) stay paired and untouched,
        so the next frame simply retries LK across a longer baseline. The frame
        itself is still reported as a stale NEED_REANCHOR and still requests an
        immediate keyframe -- no pose is invented.
        """
        self._consecutive_misses += 1
        if self._consecutive_misses > _FAST_SOFT_DEGRADE_MISSES:
            self._klt_2d = None
            self._klt_3d = None
            self._klt_gray = None
        if self.scheduler:
            self.scheduler.check_and_trigger(gray, stamp, force_reanchor=True)
        return self._make_stale_result(
            stamp,
            pose_status="LOST" if carry_lost else "NEED_REANCHOR",
            sync_status=SyncStatus.LOST.value
            if carry_lost
            else SyncStatus.NEED_REANCHOR.value,
            drift_count=drift_count,
        )

    def _make_stale_result(
        self,
        stamp: float,
        pose_status: str,
        sync_status: str,
        drift_count: int,
        limited_jump: dict[str, Any] | None = None,
    ) -> LocalizerResult:
        pose_payload = None
        if self._last_known_pose is not None:
            R, C, yaw = matrix_to_pose_components(self._last_known_pose)
            pose_payload = self._make_pose_dict(self._last_known_pose, R, C, yaw, stamp)

        return LocalizerResult(
            pose=pose_payload,
            pose_status=pose_status,
            candidate_mode="klt_fast",
            inliers=0,
            tracked=0,
            stale=True,
            stamp=stamp,
            drift_count=drift_count,
            sync_status=sync_status,
            limited_jump=limited_jump,
        )

    @staticmethod
    def _make_pose_dict(
        T: np.ndarray, R: np.ndarray, C: np.ndarray, yaw: float, stamp: float
    ) -> dict[str, Any]:
        return {
            "x": float(C[0]),
            "y": float(C[1]),
            "z": float(C[2]),
            "yaw": float(yaw),
            "stamp": float(stamp),
            "cam_from_world": T,
            "rotation": R,
            "center": C,
        }


# --- 3. SlowPath ---

class SlowPath:
    """Slow-frequency (5~10Hz) GPU worker handling EDM keyframe matching and MegaLoc.

    - Single-GPU serial execution with priority queue.
    - Tasks: TaskPriority.REANCHOR (0) > TaskPriority.PERIODIC (1) > TaskPriority.MEGALOC (2).
    - MegaLoc NEVER blocks EDM keyframes (preempted by higher priority keyframes).
    - Results passing anchor threshold are placed into AnchorMailbox.
    """

    def __init__(
        self,
        mailbox: AnchorMailbox,
        edm_matcher: Callable[[np.ndarray, float], dict[str, Any] | None] | None = None,
        megaloc_fn: Callable[[Any, float], dict[str, Any] | None] | None = None,
        weak_min_inliers: int = _DEFAULT_WEAK_MIN_INLIERS,
        anchor_min_inliers: int = _DEFAULT_ANCHOR_MIN_INLIERS,
        anchor_min_ratio: float = _DEFAULT_ANCHOR_MIN_RATIO,
        max_reproj_rms: float = _DEFAULT_MAX_REPROJ_ERROR,
    ) -> None:
        self.mailbox = mailbox
        self.edm_matcher = edm_matcher
        self.megaloc_fn = megaloc_fn
        self.weak_min_inliers = weak_min_inliers
        # An anchor must be at least as good as a weak fix, and otherwise
        # exactly what the caller asked for. Doubling the weak floor here used
        # to invent a bar above the site's own accept thresholds and silently
        # dropped legitimate slow-tracker accepts (the 2026-09-04 gate lost
        # every recovery anchor that way).
        self.anchor_min_inliers = max(int(weak_min_inliers), int(anchor_min_inliers))
        self.anchor_min_ratio = anchor_min_ratio
        self.max_reproj_rms = max_reproj_rms
        self.anchor_reject_reason: str | None = None
        self.anchor_rejects: dict[str, int] = {}
        self.anchor_accepts: int = 0

        # Single GPU priority queue: elements are (priority, entry_id, task_type, args, callback)
        self._queue: queue.PriorityQueue[tuple[int, int, str, tuple, Any]] = queue.PriorityQueue()
        self._entry_id: int = 0
        self._lock = threading.Lock()
        # The stateful tracker behind edm_matcher must be driven in frame order.
        # Without this, an inline (caller-thread) keyframe and a background
        # keyframe interleave: whichever runs second drives an older frame with
        # a newer prior (or vice versa) and recovery degrades. The drive lock
        # makes background and inline keyframes mutually exclusive, and the
        # inline side additionally drains superseded pendings so the background
        # never re-drives an older frame afterwards. Lock order everywhere is
        # _inline_lock -> _drive_lock -> adapter state lock.
        self._drive_lock = threading.Lock()
        # Serializes inline (caller-thread) keyframes against each other.
        self._inline_lock = threading.Lock()
        self._running = False
        self._worker_thread: threading.Thread | None = None
        # A slow path that died mid-flight and a slow path that never matched
        # look identical from the fast side (permanent NO_ANCHOR). Count and
        # surface faults instead of losing them with the thread.
        self.errors: int = 0
        self.last_error: str | None = None
        self.inline_runs: int = 0

    def request_keyframe(
        self,
        gray: np.ndarray,
        stamp: float,
        priority: TaskPriority = TaskPriority.PERIODIC,
        callback: Callable[[AnchorData | None], None] | None = None,
    ) -> None:
        """Enqueue keyframe matching request, dropping superseded pendings.

        The slow tracker steps densely (every fed frame); when it lags, only
        the newest pending keyframe is kept so recovery never works stale
        frames. MEGALOC tasks are left queued (lower priority anyway).
        """
        with self._lock:
            self._drop_pending_keyframes_locked()
            self._entry_id += 1
            self._queue.put((
                int(priority),
                self._entry_id,
                "keyframe",
                (np.ascontiguousarray(gray), float(stamp)),
                callback,
            ))

    def _drop_pending_keyframes_locked(self) -> None:
        """Drop queued (not in-flight) keyframe tasks. Caller holds _lock.

        Note: dropped tasks are dequeued without task_done(), matching the
        pre-existing request_keyframe purge semantics (nothing joins this
        queue).
        """
        keep: list = []
        try:
            while True:
                task = self._queue.get_nowait()
                if task[2] != "keyframe":
                    keep.append(task)
        except queue.Empty:
            pass
        for task in keep:
            self._queue.put(task)

    def request_megaloc(
        self,
        query: Any,
        stamp: float,
        callback: Callable[[dict[str, Any] | None], None] | None = None,
    ) -> None:
        """Enqueue MegaLoc retrieval request (always lower priority than keyframes)."""
        with self._lock:
            self._entry_id += 1
            task = (
                int(TaskPriority.MEGALOC),
                self._entry_id,
                "megaloc",
                (query, float(stamp)),
                callback,
            )
            self._queue.put(task)

    def step(self) -> bool:
        """Execute one task from priority queue. Useful for deterministic testing."""
        try:
            priority, _, task_type, args, callback = self._queue.get_nowait()
        except queue.Empty:
            return False

        try:
            if task_type == "keyframe":
                gray, stamp = args
                # Mutually exclusive with run_inline: the tracker behind
                # edm_matcher is stateful and must see frames in order.
                with self._drive_lock:
                    anchor = self._process_keyframe(gray, stamp)
                if callback:
                    callback(anchor)
            elif task_type == "megaloc":
                query, stamp = args
                result = self._process_megaloc(query, stamp)
                if callback:
                    callback(result)
        except Exception as exc:  # keep the worker alive; a dead thread is silent
            self.errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._queue.task_done()
        return True
    def run_inline(self, gray: np.ndarray, stamp: float,
                   max_age_s: float | None = None) -> AnchorData | None:
        """Run one keyframe on the calling thread and publish it to the mailbox.

        The async floor: when the fast path has nothing to publish, paying the
        synchronous keyframe cost is strictly better than emitting no pose. The
        drive lock makes this mutually exclusive with a background keyframe, so
        the stateful tracker always sees frames in order; queued keyframes older
        than this frame are superseded pendings and are dropped, and if the
        background just delivered a fresh anchor for a nearby frame it is used
        instead of spending a second forward.
        """
        if self.edm_matcher is None:
            return None
        with self._inline_lock:
            with self._drive_lock:
                # Peek with the same freshness bound the carry will apply on
                # take: an anchor older than that would be discarded on consume,
                # so it must not suppress the inline keyframe. Waiting on the
                # drive lock above already let an in-flight background keyframe
                # finish and publish, so this peek also absorbs that case.
                fresh = self.mailbox.peek_if_fresh(float(stamp), float(max_age_s)
                                                  if max_age_s is not None else float("inf"))
                if fresh is not None:
                    return fresh
                # This frame is the newest the caller has: anything still queued
                # is an older frame the tracker must not re-drive afterwards.
                with self._lock:
                    self._drop_pending_keyframes_locked()
                self.inline_runs += 1
                try:
                    return self._process_keyframe(np.ascontiguousarray(gray), float(stamp))
                except Exception as exc:
                    self.errors += 1
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    return None

    def _process_keyframe(self, gray: np.ndarray, stamp: float) -> AnchorData | None:
        if self.edm_matcher is None:
            return None

        result = self.edm_matcher(gray, stamp)
        if not self._is_safe_visual_anchor(result):
            return None

        # Format and put into mailbox
        cam_from_world = cam_from_world_to_matrix(result["cam_from_world"])
        inlier_2d = np.asarray(result.get("inlier_2d", []), dtype=np.float32)
        inlier_3d = np.asarray(result.get("inlier_3d", []), dtype=np.float64)
        num_inliers = int(result.get("inliers", len(inlier_2d)))

        anchor = AnchorData(
            key_stamp=stamp,
            cam_from_world=cam_from_world,
            inlier_2d=inlier_2d,
            inlier_3d=inlier_3d,
            num_inliers=num_inliers,
            reproj_rms=result.get("reproj_rms"),
            inlier_ratio=float(result.get("inlier_ratio", 1.0)),
            inlier_grid_cells=int(result.get("inlier_grid_cells", 0)),
            gray=gray,
            meta=result,
        )
        self.mailbox.put(anchor)
        return anchor

    def _process_megaloc(self, query: Any, stamp: float) -> dict[str, Any] | None:
        if self.megaloc_fn is None:
            return None
        return self.megaloc_fn(query, stamp)

    def _is_safe_visual_anchor(self, res: Mapping[str, Any] | None) -> bool:
        """Validate result against anchor thresholds and pose_guided/quality.py semantics.

        Records ``anchor_reject_reason`` / ``anchor_rejects`` so a starved fast
        path can be told apart from a slow path that never produced anything.
        """
        reason = self._anchor_reject_reason(res)
        self.anchor_reject_reason = reason
        if reason is None:
            self.anchor_accepts += 1
            return True
        self.anchor_rejects[reason] = self.anchor_rejects.get(reason, 0) + 1
        return False

    def _anchor_reject_reason(self, res: Mapping[str, Any] | None) -> str | None:
        if res is None:
            return "no_result"
        if res.get("ok") is not True:
            return "not_ok"
        if res.get("rejected"):
            return f"rejected:{res.get('rejected')}"

        # quality.py semantics: reject KLT_BRIDGED or PREDICTED_ONLY as visual anchors
        pose_status = str(res.get("pose_status", ""))
        if pose_status in ("PREDICTED_ONLY", "KLT_BRIDGED"):
            return "not_visually_confirmed"
        if res.get("candidate_mode") == "klt_bridge" or res.get("bridge"):
            return "bridged"

        inliers = res.get("inliers")
        if inliers is None or int(inliers) < self.anchor_min_inliers:
            return "inliers"

        ratio = res.get("inlier_ratio")
        if ratio is None or float(ratio) < self.anchor_min_ratio:
            return "inlier_ratio"

        reproj = res.get("reproj_rms")
        if reproj is not None:
            if not math.isfinite(float(reproj)) or float(reproj) > self.max_reproj_rms:
                return "reproj_rms"

        return None

    def start(self) -> None:
        """Start background worker thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._worker_thread = threading.Thread(
                target=self._worker_loop, daemon=True, name="async-slow-path"
            )
            self._worker_thread.start()

    def stop(self) -> None:
        """Stop background worker thread."""
        with self._lock:
            self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

    def _worker_loop(self) -> None:
        while self._running:
            if not self.step():
                time.sleep(0.005)


# --- 5. Scheduler ---

class Scheduler:
    """Three-level keyframe trigger scheduler with priority and decision logging.

    - Cadence:
        - TRACK: 300ms
        - WEAK: 100ms
        - NEED_REANCHOR: 0ms (immediate preemption)
    - Re-anchor prioritizes over periodic.
    - Records decision log for record-replay verification and multi-seed statistical analysis.
    """

    def __init__(
        self,
        slow_path: SlowPath,
        state: SchedulerTriggerState = SchedulerTriggerState.TRACK,
        decision_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.slow_path = slow_path
        self.state = state
        self.decision_callback = decision_callback

        self.last_trigger_stamp: float | None = None
        self.decision_log: list[dict[str, Any]] = []

    def set_state(self, state: SchedulerTriggerState | str) -> None:
        """Update scheduler triggering state."""
        if isinstance(state, str):
            state = SchedulerTriggerState(state)
        self.state = state

    def reset(self) -> None:
        """Clear keyframe cadence (next frame triggers immediately)."""
        self.last_trigger_stamp = None

    def check_and_trigger(
        self,
        gray: np.ndarray,
        stamp: float,
        state: SchedulerTriggerState | str | None = None,
        force_reanchor: bool = False,
    ) -> bool:
        """Evaluate keyframe condition and trigger slow path if due."""
        stamp = float(stamp)
        if state is not None:
            self.set_state(state)

        # 1. Immediate Re-anchor
        if force_reanchor or self.state == SchedulerTriggerState.NEED_REANCHOR:
            priority = TaskPriority.REANCHOR
            self.slow_path.request_keyframe(gray, stamp, priority=priority)
            self.last_trigger_stamp = stamp
            self._record_decision(stamp, self.state.value, True, "reanchor_immediate", priority)
            return True

        # 2. Cadence Interval
        interval = (
            _SCHEDULER_INTERVAL_WEAK
            if self.state == SchedulerTriggerState.WEAK
            else _SCHEDULER_INTERVAL_TRACK
        )

        triggered = False
        reason = "interval_not_met"
        priority = TaskPriority.PERIODIC

        if self.last_trigger_stamp is None:
            triggered = True
            reason = "initial_trigger"
        else:
            elapsed = stamp - self.last_trigger_stamp
            if elapsed >= interval:
                triggered = True
                reason = f"elapsed_{elapsed:.3f}s_gte_{interval:.3f}s"

        if triggered:
            self.slow_path.request_keyframe(gray, stamp, priority=priority)
            self.last_trigger_stamp = stamp

        self._record_decision(stamp, self.state.value, triggered, reason, priority)
        return triggered

    def _record_decision(
        self,
        stamp: float,
        state: str,
        triggered: bool,
        reason: str,
        priority: TaskPriority,
    ) -> None:
        entry = {
            "stamp": stamp,
            "state": state,
            "triggered": triggered,
            "reason": reason,
            "priority": priority.name,
        }
        self.decision_log.append(entry)
        while len(self.decision_log) > _DECISION_LOG_CAP:
            del self.decision_log[0]
        if self.decision_callback:
            self.decision_callback(entry)


# --- 6. AsyncLocalizer ---

class AsyncLocalizer:
    """High-level dual-thread asynchronous visual localizer coordinator."""

    def __init__(
        self,
        camera: Any,
        edm_matcher: Callable[[np.ndarray, float], dict[str, Any] | None] | None = None,
        megaloc_fn: Callable[[Any, float], dict[str, Any] | None] | None = None,
        max_jump: float = _DEFAULT_MAX_JUMP,
        max_yaw_jump_deg: float = _DEFAULT_MAX_YAW_JUMP_DEG,
        max_drift_budget: int = _DEFAULT_MAX_DRIFT_BUDGET,
        max_anchor_age_s: float = _DEFAULT_MAX_ANCHOR_AGE_S,
        max_anchorless_frames: int = _DRIFT_BUDGET_CAP,
        weak_min_inliers: int = _DEFAULT_WEAK_MIN_INLIERS,
        anchor_min_inliers: int = _DEFAULT_ANCHOR_MIN_INLIERS,
        anchor_min_ratio: float = _DEFAULT_ANCHOR_MIN_RATIO,
        max_reproj_error: float = _DEFAULT_MAX_REPROJ_ERROR,
        pnp_ransac_seed: int = 42,
        fast_min_inliers: int | None = None,
        fast_max_reproj_error: float | None = None,
        inline_sync_fallback: bool = True,
    ) -> None:
        self.inline_sync_fallback = bool(inline_sync_fallback)
        self.inline_syncs: int = 0
        self.mailbox = AnchorMailbox()
        self.transform_chain = TransformChain()
        self.sync_carry = SyncCarry(
            mailbox=self.mailbox,
            transform_chain=self.transform_chain,
            max_jump=max_jump,
            max_yaw_jump_deg=max_yaw_jump_deg,
            max_anchor_age_s=max_anchor_age_s,
            max_anchorless_frames=max_anchorless_frames,
        )
        self.slow_path = SlowPath(
            mailbox=self.mailbox,
            edm_matcher=edm_matcher,
            megaloc_fn=megaloc_fn,
            weak_min_inliers=weak_min_inliers,
            anchor_min_inliers=anchor_min_inliers,
            anchor_min_ratio=anchor_min_ratio,
            max_reproj_rms=max_reproj_error,
        )
        self.scheduler = Scheduler(slow_path=self.slow_path)
        self.fast_path = FastPath(
            camera=camera,
            sync_carry=self.sync_carry,
            transform_chain=self.transform_chain,
            scheduler=self.scheduler,
            max_jump=max_jump,
            weak_min_inliers=(
                weak_min_inliers if fast_min_inliers is None else fast_min_inliers
            ),
            max_reproj_error=(
                max_reproj_error
                if fast_max_reproj_error is None
                else fast_max_reproj_error
            ),
            pnp_ransac_seed=pnp_ransac_seed,
        )

    def feed_frame(
        self, gray: np.ndarray, stamp: float, state: str | None = None
    ) -> LocalizerResult:
        """Process one incoming camera frame."""
        if state is not None:
            self.scheduler.set_state(state)
        # Check scheduler for periodic keyframe trigger
        self.scheduler.check_and_trigger(gray, stamp)
        # Fast path processing
        res = self.fast_path.feed(gray, stamp)
        if res.pose_status in ("VISUALLY_CONFIRMED", "KLT_BRIDGED"):
            return res
        if not self.inline_sync_fallback:
            return res
        # Nothing publishable (BOOT before the first anchor lands, a total
        # track loss, or a stale NEED_REANCHOR/LOST/NO_ANCHOR -- stale results
        # still carry the last-known pose payload, so the trigger must be the
        # status, not pose presence). The scheduler has already queued a
        # keyframe, but a caller that consumes frames faster than the slow path
        # can answer would otherwise never see a pose at all. Run that keyframe
        # on this thread instead: the frame then costs what the synchronous
        # tracker costs, which is the floor async must never fall below.
        # (A stand-down that skips inline after consecutive failures was tried
        # and reverted: failed keyframes still advance the slow state machine,
        # so skipping them starves recovery -- sequential 151 collapsed to 111
        # with LOST 7 ballooning to 72.)
        if self.slow_path.run_inline(
                gray, stamp,
                max_age_s=self.sync_carry.effective_anchor_max_age_s()) is None:
            return res
        self.inline_syncs += 1
        retry = self.fast_path.feed(gray, stamp)
        return retry if retry.pose_status in ("VISUALLY_CONFIRMED", "KLT_BRIDGED") else res

    def reset(self) -> None:
        """Drop all fast/slow tracking state (mirror of worker reset_to_boot)."""
        self.mailbox.clear()
        self.transform_chain.clear()
        self.sync_carry.reset()
        self.fast_path.reset()
        self.scheduler.reset()


    def start(self) -> None:
        """Start asynchronous slow path background thread."""
        self.slow_path.start()

    def stop(self) -> None:
        """Stop asynchronous slow path background thread."""
        self.slow_path.stop()
