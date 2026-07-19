#!/usr/bin/env python3
"""Production EDM tracker: the state machine of ProductionXFeatTracker, EDM local matching.

Same shape as the XFeat tracker (BOOT_INIT / TRACK / WEAK_TRACK / LOST, MegaLoc once at
BOOT and once per LOST episode, pose-prior + covisibility for tracking), because it keeps
the per-frame cost down -- and it matters MORE for EDM, not less:

  XFeat amortises the map side (descriptors are precomputed, so an extra candidate
  reference costs one cheap LighterGlue call). EDM cannot: it is detector-free, so every
candidate reference is a full network forward. The number of candidates per frame is
therefore the dominant cost term, so normal TRACK stays at one reference.

Measured on the RTX 5060 replay: TRACK topk=1 is about 25.6 FPS; topk=2 increases
inliers but roughly doubles median latency. LOW/WEAK temporarily uses topk=3 instead.

What is deliberately NOT ported from the XFeat tracker:
  - the temporal anchor cache. It caches descriptors of inlier 3D anchors to match the
    next frame cheaply; a detector-free matcher has no descriptors to cache. The natural
    EDM equivalent (match against the previous FRAME and carry its 3D) is left out until
    it is shown to be needed -- at 1-2 references, matching is no longer the bottleneck.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pycolmap

from edm_matcher import GRID_H, GRID_W, EDMMatcher
from reloc_localizer_edm import Camera, EDMLocalizer, EDMRelocMap


@dataclass
class EDMConfig:
    # acquisition (BOOT_INIT / LOST): MegaLoc retrieval
    boot_global_topk: int = 10
    acquire_min_inliers: int = 80
    # Run MegaLoc once at BOOT and once on entry to each LOST episode. LOW/WEAK
    # remains geometry-only and raises only the local EDM candidate count.
    global_retrieval_policy: str = "boot_and_lost_once"
    lost_local_topk: int = 5
    lost_local_grace_frames: int = 12
    recovery_bank_size: int = 192
    recovery_scan_topk: int = 2
    # Bound EDM's reference batch so boot_global_topk=10 does not allocate a
    # batch-of-10 activation peak.  All retrieved references are still evaluated.
    match_batch_size: int = 2
    use_temporal_reference: bool = False
    temporal_map_topk: int = 1
    # tracking: geometry-only candidates, no MegaLoc
    local_topk: int = 1
    weak_local_topk: int = 3
    near_pool: int = 24
    covis_per_ref: int = 20
    radius: float = 0.8
    max_yaw_diff_deg: float = 90.0
    track_min_inliers: int = 50
    weak_min_inliers: int = 30
    # gates
    max_reproj_error_acquire: float = 5.0
    max_reproj_error_track: float = 6.0
    pnp_ransac_max_error: float = 5.0
    max_jump: float = 2.0
    adaptive_jump_factor: float = 8.0
    adaptive_jump_floor: float = 0.003
    adaptive_jump_bootstrap: float = 0.02
    adaptive_jump_ceiling: float = 0.008
    adaptive_jump_min_history: int = 20
    weak_after: int = 2
    lost_after: int = 2
    # EDM emits thousands of correspondences; PnP cost is linear in them and RANSAC
    # gains nothing past a well-spread ~900. Cap spatially so the cap does not bias
    # the pose toward whichever image region happened to match densely.
    max_corr_total: int = 900
    corr_grid: int = 8


@dataclass
class RuntimeState:
    state: str = "BOOT_INIT"
    center: np.ndarray | None = None
    yaw: float | None = None
    velocity: np.ndarray | None = None
    last_refs: list = field(default_factory=list)
    misses: int = 0
    frame: int = 0
    accepted_step_norms: list[float] = field(default_factory=list)
    boot_refs: list[str] = field(default_factory=list)
    global_retrieval_calls: int = 0
    lost_global_retrieval_done: bool = False
    lost_frames: int = 0
    recovery_cursor: int = 0


def _angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def build_recovery_bank(ref_names: list[str], max_refs: int) -> list[str]:
    """Create a bounded, sequence-aware reference bank for EDM-only global recovery.

    Slots are allocated roughly in proportion to each sequence size, sampled uniformly
    within the sequence, then interleaved so a cyclic scan reaches every route early.
    """
    if max_refs <= 0 or not ref_names:
        return []
    if len(ref_names) <= max_refs:
        return list(ref_names)
    groups: dict[str, list[str]] = {}
    for name in ref_names:
        groups.setdefault(name.split("/", 1)[0], []).append(name)
    total = len(ref_names)
    keys = list(groups)
    raw = {key: max_refs * len(groups[key]) / total for key in keys}
    allocation = {key: min(len(groups[key]), max(1, int(math.floor(raw[key])))) for key in keys}
    while sum(allocation.values()) > max_refs:
        candidates = [key for key in keys if allocation[key] > 1]
        if not candidates:
            break
        key = max(candidates, key=lambda item: allocation[item] - raw[item])
        allocation[key] -= 1
    while sum(allocation.values()) < max_refs:
        candidates = [key for key in keys if allocation[key] < len(groups[key])]
        if not candidates:
            break
        key = max(candidates, key=lambda item: raw[item] - allocation[item])
        allocation[key] += 1
    sampled: dict[str, list[str]] = {}
    for key in keys:
        group = groups[key]
        count = allocation[key]
        indices = np.linspace(0, len(group) - 1, num=count, dtype=int)
        sampled[key] = [group[int(idx)] for idx in indices]
    bank: list[str] = []
    for offset in range(max(len(values) for values in sampled.values())):
        for key in keys:
            if offset < len(sampled[key]):
                bank.append(sampled[key][offset])
    return bank[:max_refs]


def build_temporal_lut(
    points2d_camera: np.ndarray,
    points3d: np.ndarray,
    inlier_mask: np.ndarray,
    camera_to_edm_scale: float,
) -> np.ndarray:
    """Bind successful query-frame PnP inliers to that frame's EDM coarse cells."""
    lut = np.full((GRID_W * GRID_H, 3), np.nan, dtype=np.float32)
    mask = np.asarray(inlier_mask, dtype=bool)
    if not mask.any():
        return lut
    points_edm = np.asarray(points2d_camera, dtype=np.float32)[mask] / camera_to_edm_scale
    cells = EDMMatcher.cell_ids(points_edm)
    lut[cells] = np.asarray(points3d, dtype=np.float32)[mask]
    return lut


def adaptive_jump_limit(step_history: list[float], cfg: EDMConfig) -> float:
    """Return a trajectory-scale envelope, bounded by the hard teleport gate."""
    if len(step_history) < cfg.adaptive_jump_min_history:
        return min(
            cfg.max_jump,
            cfg.adaptive_jump_ceiling,
            cfg.adaptive_jump_bootstrap,
        )
    typical = float(np.median(np.asarray(step_history, dtype=float)))
    return min(
        cfg.max_jump,
        cfg.adaptive_jump_ceiling,
        max(cfg.adaptive_jump_floor, cfg.adaptive_jump_factor * typical),
    )


def limit_center_step(
    previous: np.ndarray,
    candidate: np.ndarray,
    step_history: list[float],
    cfg: EDMConfig,
) -> tuple[np.ndarray, dict]:
    """Limit an isolated PnP center spike without turning it into a teleport."""
    previous = np.asarray(previous, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    delta = candidate - previous
    raw_step = float(np.linalg.norm(delta))
    limit = adaptive_jump_limit(step_history, cfg)
    if raw_step <= limit or raw_step == 0.0:
        return candidate, {"limited": False, "raw_step": raw_step, "limit": limit}
    center = previous + delta * (limit / raw_step)
    return center, {"limited": True, "raw_step": raw_step, "limit": limit}


def spatially_cap(pts2d: np.ndarray, pts3d: np.ndarray, max_total: int,
                  width: int, height: int, grid: int = 8):
    """Keep <=max_total correspondences, spread evenly over an grid x grid image grid."""
    n = len(pts2d)
    if n <= max_total:
        return pts2d, pts3d
    gx = np.clip((pts2d[:, 0] / width * grid).astype(int), 0, grid - 1)
    gy = np.clip((pts2d[:, 1] / height * grid).astype(int), 0, grid - 1)
    cell = gy * grid + gx
    keep: list[np.ndarray] = []
    per_cell = max(1, max_total // (grid * grid))
    rng = np.random.default_rng(0)
    for c in np.unique(cell):
        idx = np.nonzero(cell == c)[0]
        if len(idx) > per_cell:
            idx = rng.choice(idx, per_cell, replace=False)
        keep.append(idx)
    sel = np.concatenate(keep)
    if len(sel) > max_total:
        sel = rng.choice(sel, max_total, replace=False)
    return pts2d[sel], pts3d[sel]


class ProductionEDMTracker:
    def __init__(self, reloc_map: EDMRelocMap, camera: Camera, cfg: EDMConfig | None = None,
                 matcher: EDMMatcher | None = None, megaloc=None):
        self.cfg = cfg or EDMConfig()
        self.map = reloc_map
        self.cam = camera
        self.loc = EDMLocalizer(reloc_map, camera, matcher=matcher, megaloc=megaloc,
                                pnp_max_error=self.cfg.pnp_ransac_max_error)
        self.st = RuntimeState()
        self.centers = None if reloc_map.ref_centers is None else np.asarray(reloc_map.ref_centers, np.float32)
        self.yaws = None if reloc_map.ref_yaws is None else np.asarray(reloc_map.ref_yaws, np.float32)
        self.name_of = {i: n for i, n in enumerate(reloc_map.ref_names)}
        self.idx_of = {n: i for i, n in enumerate(reloc_map.ref_names)}
        self.recovery_bank = build_recovery_bank(
            list(reloc_map.ref_names), self.cfg.recovery_bank_size
        )
        self.temporal_gray: np.ndarray | None = None
        self.temporal_xyz_by_cell: np.ndarray | None = None

    # ---------- candidate selection ----------
    def _predict_center(self):
        if self.st.center is None:
            return None
        if self.st.velocity is None:
            return self.st.center
        return self.st.center + self.st.velocity

    def _track_candidates(self, topk: int) -> list[str]:
        """Geometry only: nearest references to the predicted pose, plus their covisibles."""
        C = self._predict_center()
        if C is None or self.centers is None:
            return []
        d = np.linalg.norm(self.centers - C[None, :], axis=1)
        finite = np.isfinite(d)
        max_yaw = math.radians(self.cfg.max_yaw_diff_deg)

        near = []
        for i in np.argsort(np.where(finite, d, np.inf)):
            i = int(i)
            if not finite[i]:
                continue
            if d[i] > self.cfg.radius and len(near) >= self.cfg.near_pool:
                break
            if self.st.yaw is not None and self.yaws is not None and np.isfinite(self.yaws[i]):
                if abs(_angle_diff(float(self.yaws[i]), self.st.yaw)) > max_yaw:
                    continue
            near.append(i)
            if len(near) >= self.cfg.near_pool:
                break

        covis_refs: list[int] = []
        if self.map.covis:
            for idx in list(self.st.last_refs)[:2] + near[:2]:
                covis_refs.extend(int(j) for j in self.map.covis.get(self.name_of[idx], [])[:self.cfg.covis_per_ref])

        union = list(dict.fromkeys(near + covis_refs + list(self.st.last_refs)))
        last = set(self.st.last_refs)

        def score(i: int) -> float:
            s = 2.0 * math.exp(-float(d[i]) / max(self.cfg.radius, 1e-6))
            if self.st.yaw is not None and self.yaws is not None and np.isfinite(self.yaws[i]):
                s += 0.8 * math.exp(-abs(_angle_diff(float(self.yaws[i]), self.st.yaw)) / math.radians(45.0))
            else:
                s += 0.4
            return s + (1.0 if i in last else 0.0)

        union = [i for i in union if 0 <= i < len(self.map.ref_names) and finite[i]]
        union.sort(key=score, reverse=True)
        return [self.name_of[i] for i in union[:topk]]

    def _recovery_candidates(self, topk: int) -> list[str]:
        if topk <= 0 or not self.recovery_bank:
            return []
        start = self.st.recovery_cursor % len(self.recovery_bank)
        refs = [
            self.recovery_bank[(start + offset) % len(self.recovery_bank)]
            for offset in range(min(topk, len(self.recovery_bank)))
        ]
        self.st.recovery_cursor = (start + len(refs)) % len(self.recovery_bank)
        return refs

    # ---------- one frame ----------
    def localize(self, frame_bgr: np.ndarray) -> dict:
        import cv2
        cfg = self.st_cfg = self.cfg
        t0 = time.perf_counter()
        self.st.frame += 1
        gray = EDMMatcher.load_gray(frame_bgr)

        acquiring = self.st.state in ("BOOT_INIT", "LOST")
        t_vpr = 0.0
        candidate_mode = "edm_track"
        if acquiring:
            if cfg.global_retrieval_policy == "boot_and_lost_once":
                run_global = (
                    (self.st.state == "BOOT_INIT" and self.st.global_retrieval_calls == 0)
                    or (self.st.state == "LOST" and not self.st.lost_global_retrieval_done)
                )
            elif cfg.global_retrieval_policy == "boot_once":
                run_global = self.st.global_retrieval_calls == 0
            else:
                run_global = True
            if not run_global:
                if (
                    self.st.state == "LOST"
                    and self.st.lost_frames > cfg.lost_local_grace_frames
                ):
                    refs = self._recovery_candidates(cfg.recovery_scan_topk)
                    candidate_mode = "edm_map_scan"
                elif self.st.state == "LOST":
                    refs = self._track_candidates(cfg.lost_local_topk)
                    candidate_mode = "edm_local_recovery"
                else:
                    refs = list(self.st.boot_refs)
                    candidate_mode = "edm_boot_refs"
                if not refs:
                    refs = list(self.st.boot_refs[: cfg.lost_local_topk])
                    candidate_mode = "edm_boot_refs"
            else:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                tv = time.perf_counter()
                refs = self.loc.retrieve(rgb, cfg.boot_global_topk)
                t_vpr = (time.perf_counter() - tv) * 1e3
                self.st.global_retrieval_calls += 1
                if self.st.state == "LOST":
                    self.st.lost_global_retrieval_done = True
                if not self.st.boot_refs:
                    self.st.boot_refs = list(refs)
                candidate_mode = (
                    "megaloc_lost" if self.st.state == "LOST" else "megaloc_boot"
                )
            min_inl = cfg.acquire_min_inliers
        else:
            temporal_ready = (
                cfg.use_temporal_reference
                and self.temporal_gray is not None
                and self.temporal_xyz_by_cell is not None
            )
            topk = (
                cfg.temporal_map_topk
                if self.st.state == "TRACK" and temporal_ready
                else cfg.local_topk if self.st.state == "TRACK" else cfg.weak_local_topk
            )
            refs = self._track_candidates(topk)
            candidate_mode = "edm_temporal_map" if temporal_ready else "edm_track"
            if not refs:                       # prior unusable -> acquisition candidate set
                if (
                    cfg.global_retrieval_policy in {"boot_once", "boot_and_lost_once"}
                    and self.st.global_retrieval_calls > 0
                ):
                    refs = list(self.st.boot_refs[: cfg.lost_local_topk])
                    candidate_mode = "edm_boot_refs"
                else:
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    tv = time.perf_counter()
                    refs = self.loc.retrieve(rgb, cfg.boot_global_topk)
                    t_vpr = (time.perf_counter() - tv) * 1e3
                    self.st.global_retrieval_calls += 1
                    if not self.st.boot_refs:
                        self.st.boot_refs = list(refs)
                    candidate_mode = "megaloc_reacquire"
            min_inl = (cfg.track_min_inliers if self.st.state == "TRACK" else cfg.weak_min_inliers)

        tm = time.perf_counter()
        by_ref = None
        temporal_used = (
            candidate_mode in ("edm_temporal_map", "edm_local_recovery")
            and cfg.use_temporal_reference
            and self.temporal_gray is not None
            and self.temporal_xyz_by_cell is not None
        )
        if temporal_used:
            source_images = [self.temporal_gray] + [self.map.images[name] for name in refs]
            source_luts = [self.temporal_xyz_by_cell] + [self.map.xyz_by_cell[name] for name in refs]
            by_source = self.loc.correspondences_for_sources(
                gray, source_images, source_luts, batch_size=cfg.match_batch_size
            )
            p2_parts = [row[0] for row in by_source if len(row[0])]
            p3_parts = [row[1] for row in by_source if len(row[1])]
            p2 = np.concatenate(p2_parts) if p2_parts else np.zeros((0, 2))
            p3 = np.concatenate(p3_parts) if p3_parts else np.zeros((0, 3))
            per_ref = [row[2] for row in by_source]
        elif candidate_mode == "edm_map_scan":
            by_ref = self.loc.correspondences_by_ref(
                gray, refs, batch_size=cfg.match_batch_size
            )
            p2_parts = [row[0] for row in by_ref if len(row[0])]
            p3_parts = [row[1] for row in by_ref if len(row[1])]
            p2 = np.concatenate(p2_parts) if p2_parts else np.zeros((0, 2))
            p3 = np.concatenate(p3_parts) if p3_parts else np.zeros((0, 3))
            per_ref = [row[2] for row in by_ref]
        else:
            p2, p3, per_ref = self.loc.correspondences(
                gray, refs, batch_size=cfg.match_batch_size
            )
        t_match = (time.perf_counter() - tm) * 1e3

        tp = time.perf_counter()
        ret = None
        selected_ref = None
        pose_points2d = np.zeros((0, 2))
        pose_points3d = np.zeros((0, 3))
        pcam = pycolmap.Camera(model=self.cam.model, width=self.cam.width,
                               height=self.cam.height, params=self.cam.params)
        opts = pycolmap.AbsolutePoseEstimationOptions()
        opts.ransac.max_error = cfg.pnp_ransac_max_error

        def estimate_pose(points2d: np.ndarray, points3d: np.ndarray):
            if len(points3d) < 6:
                return None
            p2c, p3c = spatially_cap(
                points2d, points3d, cfg.max_corr_total, self.cam.width,
                self.cam.height, cfg.corr_grid
            )
            estimate = pycolmap.estimate_and_refine_absolute_pose(
                np.asarray(p2c, float), np.asarray(p3c, float), pcam, opts
            )
            return None if estimate is None else (estimate, p2c, p3c)

        if by_ref is not None:
            for name, (ref_p2, ref_p3, _) in zip(refs, by_ref):
                candidate_pose = estimate_pose(ref_p2, ref_p3)
                if candidate_pose is not None and (
                    ret is None or int(candidate_pose[0]["num_inliers"]) > int(ret["num_inliers"])
                ):
                    ret, pose_points2d, pose_points3d = candidate_pose
                    selected_ref = name
        else:
            candidate_pose = estimate_pose(p2, p3)
            if candidate_pose is not None:
                ret, pose_points2d, pose_points3d = candidate_pose
        t_pnp = (time.perf_counter() - tp) * 1e3

        info = {
            "frame": self.st.frame, "state_in": self.st.state, "refs": refs,
            "candidate_mode": candidate_mode,
            "temporal_used": temporal_used,
            "selected_ref": selected_ref,
            "n_corr": int(len(p3)), "per_ref": per_ref,
            "global_retrieval_calls": self.st.global_retrieval_calls,
            "lost_global_retrieval_done": self.st.lost_global_retrieval_done,
            "vpr_ms": t_vpr, "match_ms": t_match, "pnp_ms": t_pnp,
            "total_ms": (time.perf_counter() - t0) * 1e3,
        }

        if ret is None or int(ret["num_inliers"]) < min_inl:
            info["inliers"] = 0 if ret is None else int(ret["num_inliers"])
            self._on_miss(info)
            return info

        T = ret["cam_from_world"]
        R = T.rotation.matrix()
        C = -R.T @ np.asarray(T.translation)
        fwd = R.T @ np.array([0, 0, 1.0])
        yaw = float(math.atan2(fwd[1], fwd[0]))
        info["inliers"] = int(ret["num_inliers"])

        # Two-layer trajectory gate. Extreme centers are rejected. Smaller isolated
        # PnP spikes are limited to a robust trajectory-scale envelope so they cannot
        # become accepted ghost teleports or force an otherwise healthy track LOST.
        if self.st.center is not None and info["state_in"] != "LOST":
            raw_step = float(np.linalg.norm(C - self.st.center))
            if raw_step > cfg.max_jump:
                info["rejected"] = "jump"
                self._on_miss(info)
                return info
            C, jump_info = limit_center_step(
                self.st.center, C, self.st.accepted_step_norms, cfg
            )
            if jump_info["limited"]:
                info["limited_jump"] = {
                    "raw_step": jump_info["raw_step"],
                    "limit": jump_info["limit"],
                }

        vel = (
            None
            if self.st.center is None or info["state_in"] == "LOST"
            else (C - self.st.center)
        )
        if vel is not None:
            self.st.accepted_step_norms.append(float(np.linalg.norm(vel)))
        self.st.velocity = vel if self.st.velocity is None or vel is None else 0.5 * (self.st.velocity + vel)
        self.st.center, self.st.yaw = C, yaw
        accepted_refs = [selected_ref] if selected_ref is not None else refs
        self.st.last_refs = [self.idx_of[n] for n in accepted_refs]
        self.st.misses = 0
        self.st.lost_frames = 0
        self.st.lost_global_retrieval_done = False
        self.st.state = "TRACK"
        inlier_mask = np.asarray(
            ret.get("inlier_mask", np.ones(len(pose_points3d), dtype=bool)),
            dtype=bool,
        )
        self.temporal_xyz_by_cell = build_temporal_lut(
            pose_points2d, pose_points3d, inlier_mask,
            camera_to_edm_scale=self.loc.scale,
        )
        self.temporal_gray = gray.copy()
        info.update({"state_out": "TRACK", "center": C, "yaw": yaw, "R": R, "ok": True})
        return info

    def _on_miss(self, info: dict):
        self.st.misses += 1
        if self.st.state == "TRACK" and self.st.misses >= self.cfg.weak_after:
            self.st.state = "WEAK_TRACK"
        elif self.st.state == "WEAK_TRACK" and self.st.misses >= self.cfg.weak_after + self.cfg.lost_after:
            self.st.state = "LOST"
            self.st.velocity = None
            self.st.lost_global_retrieval_done = False
        if self.st.state == "LOST":
            self.st.lost_frames += 1
        else:
            self.st.lost_frames = 0
        info.update({"state_out": self.st.state, "ok": False})


if __name__ == "__main__":
    print(__doc__)
