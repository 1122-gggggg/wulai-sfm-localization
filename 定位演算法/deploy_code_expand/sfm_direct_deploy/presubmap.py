"""Bounded independent monocular submap; strong map poses establish its scale."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from workmap_submaps import (
    ANCHOR_CENTER_TOL_U,
    AnchorPair,
    Sim3,
    anchor_consistency,
    fit_similarity,
)

FEATURES = 400
MIN_DISTANCE = 8
FB_MAX_PX = 1.0
INIT_MIN_FLOW_PX = 20.0
INIT_MIN_INLIERS = 80
INIT_MIN_POINTS = 50
TRI_MIN_ANGLE_DEG = 1.5
TRI_MAX_REPROJ_PX = 2.0
RESEED_BELOW = 250
PNP_REPROJ_PX = 3.0
PNP_MIN_INLIERS = 20
#: Only every STRIDE-th frame is tracked: half the cost, and poses/keyframes stay on even ordinals.
STRIDE = 2
BATCH_EVERY = 8
BATCH_MAX_AGE = 90
KEYFRAME_EVERY = 8
#: Pure monocular VO drifts 10-30 % of the path, so only the last HISTORY_FRAMES before
#: the merge are handed over, bounded in points and keyframes.
HISTORY_FRAMES = 240
MAX_MERGE_POINTS = 4000
MAX_MERGE_KEYFRAMES = 30
QUEUE_DEPTH = 24
#: Anchors: enough strong poses, close enough in time that the local drift between them is small.
ANCHOR_COUNT = 8
ANCHOR_MAX_SPAN = 48
ANCHOR_MIN_SPAN = 8
GIVE_UP_AFTER_LOST = 45
#: Skip brief in-map starts; a pre-fix monocular submap needs an observable baseline.
MIN_STRETCH_S = 1.25
GIVE_UP_AFTER_ANCHOR = 150
_LK = dict(
    winSize=(15, 15),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)


def _klt(prev: np.ndarray, cur: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p0 = xy.astype(np.float32).reshape(-1, 1, 2)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **_LK)
    back, stb, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **_LK)
    fwd = p1.reshape(-1, 2).astype(np.float64)
    keep = (
        (st.ravel() == 1)
        & (stb.ravel() == 1)
        & (np.linalg.norm(back.reshape(-1, 2) - xy, axis=1) <= FB_MAX_PX)
        & (fwd[:, 0] >= 0)
        & (fwd[:, 0] < cur.shape[1])
        & (fwd[:, 1] >= 0)
        & (fwd[:, 1] < cur.shape[0])
    )
    return fwd, keep


def _detect(gray: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    mask = np.full(gray.shape, 255, dtype=np.uint8)
    points = np.asarray(occupied, dtype=int).reshape(-1, 2)
    if len(points):
        inside = (
            (points[:, 0] >= 0) & (points[:, 0] < gray.shape[1])
            & (points[:, 1] >= 0) & (points[:, 1] < gray.shape[0])
        )
        for x, y in points[inside]:
            cv2.circle(mask, (int(x), int(y)), MIN_DISTANCE, 0, -1)
    corners = cv2.goodFeaturesToTrack(gray, FEATURES, 0.01, MIN_DISTANCE, mask=mask)
    return np.zeros((0, 2)) if corners is None else corners.reshape(-1, 2).astype(np.float64)


def _center(pose: np.ndarray) -> np.ndarray:
    return -pose[:, :3].T @ pose[:, 3]


def _angles_deg(points: np.ndarray, pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    ray_a, ray_b = points - _center(pose_a), points - _center(pose_b)
    norm = np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1) + 1e-12
    return np.degrees(np.arccos(np.clip((ray_a * ray_b).sum(1) / norm, -1.0, 1.0)))


def _triangulate(
    K: np.ndarray, pose_a: np.ndarray, pose_b: np.ndarray, xy_a: np.ndarray, xy_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = cv2.triangulatePoints(
        K @ pose_a, K @ pose_b, xy_a.T.astype(np.float64), xy_b.T.astype(np.float64)
    )
    w = homogeneous[3]
    good = np.abs(w) > 1e-12
    world = np.zeros((len(xy_a), 3))
    world[good] = (homogeneous[:3, good] / w[good]).T
    for pose, xy in ((pose_a, xy_a), (pose_b, xy_b)):
        camera = world @ pose[:, :3].T + pose[:, 3]
        depth = camera[:, 2]
        good &= depth > 1e-6
        projected = camera @ K.T
        safe = np.where(depth[:, None] > 1e-6, projected[:, 2:3], 1.0)
        good &= np.linalg.norm(projected[:, :2] / safe - xy, axis=1) < TRI_MAX_REPROJ_PX
    return world, good


def _rgb_sampler(rgb: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """3x3-mean colour lookup at fast-loop pixels of an RGB frame of the same size."""
    height, width = rgb.shape[:2]

    def sample(xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        columns, rows = np.rint(xy[:, 0]).astype(int), np.rint(xy[:, 1]).astype(int)
        total = np.zeros((len(xy), 3), dtype=np.float64)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                total += rgb[np.clip(rows + dy, 0, height - 1), np.clip(columns + dx, 0, width - 1)]
        return np.clip(np.rint(total / 9.0), 0, 255).astype(np.uint8)

    return sample


@dataclass
class LocalPoint:
    seed_ordinal: int
    seed_stamp: float
    seed_xy: np.ndarray
    seed_pose: np.ndarray
    ordinal: int
    stamp: float
    xy: np.ndarray
    pose: np.ndarray
    xyz: np.ndarray
    parallax_deg: float
    rgb: list[int] | None


@dataclass
class LocalKeyframe:
    ordinal: int
    stamp: float
    gray: np.ndarray | None
    point_ids: np.ndarray
    xy: np.ndarray


@dataclass
class MergeResult:
    scale: float
    residual_u: float
    anchors: int
    points: dict[int, LocalPoint]
    keyframes: list[LocalKeyframe]
    poses_local: dict[int, np.ndarray]
    similarity: tuple[float, np.ndarray, np.ndarray]
    stamps: dict[int, float] = field(default_factory=dict)
    anchor_views: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    anchor_poses: dict[int, np.ndarray] = field(default_factory=dict)
    poses_map: dict[int, np.ndarray] = field(default_factory=dict)
    points_map: dict[int, np.ndarray] = field(default_factory=dict)


class PreAnchorSubmap:
    """Local-frame VO from the first frame; ``try_merge`` once strong anchors exist."""

    def __init__(self, K: np.ndarray) -> None:
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.state = "idle"        # idle -> tracking -> merged | failed
        self.poses: dict[int, np.ndarray] = {}
        self.stamps: dict[int, float] = {}
        self.points: dict[int, LocalPoint] = {}
        self.keyframes: list[LocalKeyframe] = []
        self._xy = np.zeros((0, 2))
        self._xyz = np.zeros((0, 3))
        self._ids = np.zeros(0, dtype=np.int64)
        self._batches: list[dict] = []
        self._next_id = 0
        self._prev: np.ndarray | None = None
        self._init: dict | None = None
        self._lost = 0
        self._anchors: list[tuple[int, np.ndarray]] = []
        self._anchor_views: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._first_stamp: float | None = None
        self._first_anchor: int | None = None
        self.last_ms = 0.0
        self._rejections: list[str] = []
        self.dropped = 0
        self._lock = threading.Lock()
        self._jobs: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
        self._thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        return self.state in ("idle", "tracking")

    def close(self, state: str = "failed") -> None:
        self.state = state
        self.keyframes = []
        self._batches = []
        self._prev = None
        self._init = None
        try:
            self._jobs.put_nowait(None)          # lets a worker thread exit
        except queue.Full:
            pass

    def pop_rejections(self) -> list[str]:
        """Consistency rejections since the last call (the caller logs them)."""
        reasons, self._rejections = self._rejections, []
        return reasons

    # ------------------------------------------------------------- threaded use
    def wants(self, ordinal: int) -> bool:
        """Whether frame ``ordinal`` is tracked (the caller only builds inputs for these)."""
        return self.active and (not self.poses or ordinal % STRIDE == 0)

    def enqueue(self, ordinal: int, stamp: float, gray: np.ndarray, rgb: np.ndarray | None) -> None:
        """Hand a frame to the background tracker; never blocks the caller."""
        if not self.wants(ordinal):
            return
        if self._first_stamp is None:
            self._first_stamp = float(stamp)
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="presubmap-vo", daemon=True)
            self._thread.start()
        try:
            self._jobs.put_nowait((ordinal, stamp, gray, rgb))
        except queue.Full:
            self.dropped += 1

    def wait_idle(self, timeout: float = 30.0) -> None:
        """Block until queued frames are tracked (tests and offline replays)."""
        import time

        deadline = time.monotonic() + timeout
        while not self._jobs.empty() and time.monotonic() < deadline:
            time.sleep(0.005)
        with self._lock:
            pass

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            ordinal, stamp, gray, rgb = job
            with self._lock:
                if not self.active:
                    continue
                try:
                    self.step(ordinal, stamp, gray, None if rgb is None else _rgb_sampler(rgb))
                except Exception:  # noqa: BLE001 - the submap is optional
                    self.close("failed")

    # ---------------------------------------------------------------- tracking
    def step(
        self,
        ordinal: int,
        stamp: float,
        gray: np.ndarray,
        sample_rgb: Callable[[np.ndarray], np.ndarray | None] | None = None,
    ) -> None:
        if not self.active or (ordinal % STRIDE and self.poses):
            return
        if self._first_stamp is None:
            self._first_stamp = float(stamp)
        self.state = "tracking"
        self.stamps[ordinal] = float(stamp)
        if len(self.poses) == 0:
            self._try_init(ordinal, stamp, gray, sample_rgb)
            if len(self.stamps) > HISTORY_FRAMES:
                self.stamps = {
                    key: value for key, value in self.stamps.items()
                    if key >= ordinal - HISTORY_FRAMES
                }
            return
        pose = self._track(ordinal, stamp, gray, sample_rgb)
        if pose is None:
            self._lost += 1
            if self._lost >= GIVE_UP_AFTER_LOST:
                self.close("failed")
            return
        self._lost = 0
        if any(anchor == ordinal for anchor, _ in self._anchors):
            self._anchor_views[ordinal] = (self._ids.copy(), self._xy.copy())
        if ordinal % KEYFRAME_EVERY == 0:
            self._record_keyframe(ordinal, stamp, gray)
            self._prune(ordinal)

    def _try_init(self, ordinal, stamp, gray, sample_rgb) -> None:
        if self._init is None:
            xy = _detect(gray, np.zeros((0, 2)))
            self._init = dict(o0=ordinal, t0=stamp, xy0=xy, idx=np.arange(len(xy)), cur=xy.copy())
            self._prev = gray
            return
        state = self._init
        fwd, keep = _klt(self._prev, gray, state["cur"])
        state["cur"], state["idx"] = fwd[keep], state["idx"][keep]
        self._prev = gray
        if len(state["idx"]) < INIT_MIN_INLIERS:
            self._init = None
            return
        a, b = state["xy0"][state["idx"]], state["cur"]
        if float(np.median(np.linalg.norm(b - a, axis=1))) < INIT_MIN_FLOW_PX:
            return
        essential, mask = cv2.findEssentialMat(
            a, b, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if essential is None:
            return
        count, rotation, translation, mask = cv2.recoverPose(essential, a, b, self.K, mask=mask)
        if count < INIT_MIN_INLIERS:
            return
        pose_a = np.hstack([np.eye(3), np.zeros((3, 1))])
        pose_b = np.hstack([rotation, translation.reshape(3, 1)])
        inliers = mask.ravel().astype(bool)
        world, good = _triangulate(self.K, pose_a, pose_b, a[inliers], b[inliers])
        good &= _angles_deg(world, pose_a, pose_b) > TRI_MIN_ANGLE_DEG
        if int(good.sum()) < INIT_MIN_POINTS:
            return
        self.poses[state["o0"]], self.poses[ordinal] = pose_a, pose_b
        self.stamps[state["o0"]] = float(state["t0"])
        xy, xyz = b[inliers][good], world[good]
        seed_xy = a[inliers][good]
        colors = None if sample_rgb is None else sample_rgb(xy)
        ids = np.arange(len(xy), dtype=np.int64)
        for row, key in enumerate(ids):
            self._make_point(
                int(key), (state["o0"], state["t0"], seed_xy[row], pose_a),
                (ordinal, stamp, xy[row], pose_b), xyz[row],
                float(_angles_deg(xyz[row : row + 1], pose_a, pose_b)[0]),
                None if colors is None else colors[row],
            )
        self._xy, self._xyz, self._ids = xy, xyz, ids
        self._next_id = len(ids)
        self._init = None
        self._record_keyframe(ordinal, stamp, gray)

    def _make_point(self, key, seed, current, xyz, parallax, color) -> None:
        self.points[key] = LocalPoint(
            seed_ordinal=int(seed[0]), seed_stamp=float(seed[1]), seed_xy=np.array(seed[2]),
            seed_pose=np.array(seed[3]), ordinal=int(current[0]), stamp=float(current[1]),
            xy=np.array(current[2]), pose=np.array(current[3]), xyz=np.array(xyz),
            parallax_deg=parallax, rgb=None if color is None else [int(v) for v in color],
        )

    def _track(self, ordinal, stamp, gray, sample_rgb) -> np.ndarray | None:
        fwd, keep = _klt(self._prev, gray, self._xy)
        xy, xyz, ids = fwd[keep], self._xyz[keep], self._ids[keep]
        pose = None
        if len(xy) >= PNP_MIN_INLIERS:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                xyz, xy, self.K, None, reprojectionError=PNP_REPROJ_PX,
                iterationsCount=100, flags=cv2.SOLVEPNP_EPNP,
            )
            if ok and inl is not None and len(inl) >= PNP_MIN_INLIERS:
                inl = inl.ravel()
                rvec, tvec = cv2.solvePnPRefineLM(xyz[inl], xy[inl], self.K, None, rvec, tvec)
                rot, _ = cv2.Rodrigues(rvec)
                pose = np.hstack([rot, tvec.reshape(3, 1)])
                xy, xyz, ids = xy[inl], xyz[inl], ids[inl]
        self._prev = gray
        if pose is None:
            self._xy, self._xyz, self._ids = xy, xyz, ids
            return None
        self.poses[ordinal] = pose
        alive = []
        for batch in self._batches:
            moved, ok = _klt(batch["prev"], gray, batch["xy"])
            batch["xy"], batch["seed"], batch["prev"] = moved[ok], batch["seed"][ok], gray
            if not len(batch["xy"]) or ordinal - batch["ordinal"] > BATCH_MAX_AGE:
                continue
            world, good = _triangulate(self.K, batch["pose"], pose, batch["seed"], batch["xy"])
            parallax = _angles_deg(world, batch["pose"], pose)
            take = good & (parallax > TRI_MIN_ANGLE_DEG)
            if take.any():
                colors = None if sample_rgb is None else sample_rgb(batch["xy"][take])
                rows = np.flatnonzero(take)
                new_ids = np.arange(self._next_id, self._next_id + len(rows), dtype=np.int64)
                self._next_id += len(rows)
                for n, row in enumerate(rows):
                    self._make_point(
                        int(new_ids[n]),
                        (batch["ordinal"], self.stamps.get(batch["ordinal"], 0.0),
                         batch["seed"][row], batch["pose"]),
                        (ordinal, stamp, batch["xy"][row], pose), world[row],
                        float(parallax[row]), None if colors is None else colors[n],
                    )
                xy = np.vstack([xy, batch["xy"][take]])
                xyz = np.vstack([xyz, world[take]])
                ids = np.concatenate([ids, new_ids])
                batch["xy"], batch["seed"] = batch["xy"][~take], batch["seed"][~take]
            if len(batch["xy"]):
                alive.append(batch)
        self._batches = alive
        if ordinal % BATCH_EVERY == 0 or len(xy) < RESEED_BELOW:
            occupied = np.vstack([xy] + [b["xy"] for b in self._batches]) if self._batches else xy
            fresh = _detect(gray, occupied)
            if len(fresh):
                self._batches.append(
                    dict(ordinal=ordinal, pose=pose, seed=fresh, xy=fresh.copy(), prev=gray)
                )
        self._xy, self._xyz, self._ids = xy, xyz, ids
        return pose

    def _record_keyframe(self, ordinal: int, stamp: float, gray: np.ndarray) -> None:
        self.keyframes.append(
            LocalKeyframe(ordinal, float(stamp), gray.copy(), self._ids.copy(), self._xy.copy())
        )

    def _prune(self, ordinal: int) -> None:
        """Forget submap history older than HISTORY_FRAMES; tracking itself is unaffected."""
        cutoff = ordinal - HISTORY_FRAMES
        if len(self.keyframes) > MAX_MERGE_KEYFRAMES or (self.keyframes and self.keyframes[0].ordinal < cutoff):
            self.keyframes = [k for k in self.keyframes if k.ordinal >= cutoff][-MAX_MERGE_KEYFRAMES:]
        if len(self.points) > 2 * MAX_MERGE_POINTS:
            self.points = {
                key: p for key, p in self.points.items() if p.ordinal >= cutoff
            }
            if len(self.points) > MAX_MERGE_POINTS:
                newest = sorted(self.points, key=lambda k: self.points[k].ordinal)[-MAX_MERGE_POINTS:]
                self.points = {key: self.points[key] for key in newest}
        anchored = {a for a, _ in self._anchors}
        for old in list(self.poses):
            if old < cutoff and old not in anchored:
                del self.poses[old]
                self.stamps.pop(old, None)
        for old in list(self._anchor_views):
            if old < cutoff:
                del self._anchor_views[old]

    # ------------------------------------------------------------------- merge
    def add_anchor(
        self, ordinal: int, cam_from_world: np.ndarray, stamp: float | None = None
    ) -> None:
        """A strong frozen-map pose (``cam_from_world``, 3x4) for a frame the submap tracked.

        The first anchor closes the submap as ``unneeded`` when fewer than ``MIN_STRETCH_S``
        seconds preceded it: the flight started inside the map and there is nothing to cover.
        """
        if not self.active:
            return
        if self._first_anchor is None:
            self._first_anchor = int(ordinal)
            if (
                stamp is not None
                and self._first_stamp is not None
                and float(stamp) - self._first_stamp < MIN_STRETCH_S
            ):
                self.close("unneeded")
                return
        if ordinal % STRIDE == 0:                # other frames are never tracked
            self._anchors.append((int(ordinal), np.asarray(cam_from_world, dtype=np.float64)))
            if ordinal in self.poses and ordinal not in self._anchor_views:
                view = next(
                    ((frame.point_ids.copy(), frame.xy.copy()) for frame in self.keyframes if frame.ordinal == ordinal),
                    None,
                )
                if view is not None:
                    self._anchor_views[ordinal] = view

    def try_merge(self) -> MergeResult | None:
        """Fit the similarity once enough anchors accumulated; None until then."""
        if not self.active:
            return None
        if (
            self._first_anchor is not None
            and self._anchors
            and self._anchors[-1][0] - self._first_anchor > GIVE_UP_AFTER_ANCHOR
        ):
            self.close("unmerged")
            return None
        if len(self._anchors) < ANCHOR_COUNT:
            return None
        if not self._lock.acquire(blocking=False):
            return None                          # the tracker thread is mid-step; try next frame
        try:
            return self._merge_locked()
        finally:
            self._lock.release()

    def _merge_locked(self) -> MergeResult | None:
        # Anchors ahead of the tracker thread are kept: it may simply not have reached them yet.
        tracked = max(self.poses) if self.poses else -1
        self._anchors = [a for a in self._anchors if a[0] in self.poses or a[0] > tracked]
        usable = [a for a in self._anchors if a[0] in self.poses]
        if len(usable) < ANCHOR_COUNT:
            return None
        window = [a for a in usable if a[0] >= usable[-1][0] - ANCHOR_MAX_SPAN]
        if len(window) < ANCHOR_COUNT or window[-1][0] - window[0][0] < ANCHOR_MIN_SPAN:
            return None
        local = [(self.poses[o][:, :3], _center(self.poses[o])) for o, _ in window]
        mapped = [(p[:, :3], _center(p)) for _, p in window]
        scale, rotation, translation, residual = fit_similarity(local, mapped)
        if not (scale > 0 and np.isfinite(scale)) or residual > ANCHOR_CENTER_TOL_U:
            self._anchors = self._anchors[-ANCHOR_COUNT:]
            return None
        pairs = [
            AnchorPair(o, p, self.poses[o], np.zeros(0, dtype=np.int64), np.zeros((0, 2)))
            for o, p in window
        ]
        reason = anchor_consistency(pairs, Sim3(scale, rotation, translation), scale_bounds=None)
        if reason is not None:
            self._rejections.append(reason)
            self._anchors = self._anchors[-ANCHOR_COUNT:]
            return None
        cutoff = usable[-1][0] - HISTORY_FRAMES
        points = {k: p for k, p in self.points.items() if p.ordinal >= cutoff}
        if len(points) > MAX_MERGE_POINTS:
            newest = sorted(points, key=lambda k: points[k].ordinal)[-MAX_MERGE_POINTS:]
            points = {k: points[k] for k in newest}
        result = MergeResult(
            scale=scale, residual_u=residual, anchors=len(window), points=points,
            keyframes=[k for k in self.keyframes if k.ordinal >= cutoff][-MAX_MERGE_KEYFRAMES:],
            poses_local=dict(self.poses), similarity=(scale, rotation, translation),
            stamps=dict(self.stamps),
            anchor_views={o: self._anchor_views[o] for o, _ in window if o in self._anchor_views},
            anchor_poses={o: p for o, p in window},
        )
        self.close("merged")
        return result
