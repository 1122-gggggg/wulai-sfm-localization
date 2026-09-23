"""Offline cross-video evaluation of map-anchored STA relocalization.

Research branch only: no EDM, KLT or PnP in the STA path, no runtime or
profile change, no hardware.  Every sideview map keyframe is used as a query;
its own video is removed from retrieval, so references always come from a
different flight.  Ground truth is the frozen GlueMap pose of the query.

Subcommands (run in this order):

  selfcheck  xformers shim equivalence, crop mapping vs ViSTA preprocessing,
             and the STA pose convention on same-video neighbours (never used
             by the evaluation, which is cross-video only)
  sta        BoQ bank retrieval -> STA(query, ref) -> gamma from map depth ->
             pose composition -> pose consensus; also records STA timings
  edm        production EDM + PnP on the same top-k references (baseline)
  report     metrics and the pre-registered decision in
             docs/sta_map_anchored_eval_20260923.md
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import types
from typing import Any

import numpy as np

from sta_map_anchored import (
    Hypothesis,
    camera_center,
    compose_query_pose,
    estimate_gamma,
    pose_consensus,
    rotation_angle_deg,
    sta_crop,
)


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "地圖檔/場域/river_site/releases/river_gluemap_all8_direct_20260908"
VISTA = ROOT / "執行環境/models/vista_slam"
OUT = ROOT / "outputs/analysis/sta_map_anchored_20260923"
LIVE_SIZE = (1280, 720)  # site_profile query_camera; keyframe K scaled by 1/2.1

# Frozen before any accuracy result was read (docs/sta_map_anchored_eval_20260923.md).
PARAMS = {
    "top_k": 5,
    "edm_top_k": 2,
    "pose_conf_min": 0.75,
    "min_scale_samples": 30,
    "max_scale_log_mad": 0.2,
    "consensus_position_u": 0.10,
    "consensus_rotation_deg": 5.0,
    "min_frame_gap": 72,
    "gross_error_u": 0.3,
}
CRITERIA = {
    "position_p50_u": 0.05,
    "position_p95_u": 0.12,
    "rotation_p95_deg": 5.0,
    "gross_rate_max": 0.01,
    "accept_rate_min": 0.5,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _video(name: str) -> str:
    return name.split("/", 1)[0]


def _frame(name: str) -> int:
    return int(Path(name).stem.split("_")[-1])


class MapData:
    """Frozen GlueMap poses, EDM-retriangulated points and the BoQ sideview bank."""

    def __init__(self, release: Path):
        import pycolmap

        self.release = release
        self.recon = pycolmap.Reconstruction(str(release / "model"))
        self.by_name = {image.name: image for image in self.recon.images.values()}
        camera = next(iter(self.recon.cameras.values()))
        self.width, self.height = int(camera.width), int(camera.height)
        loc = release / "localization"
        self.names: list[str] = json.loads((loc / "boq_references_sideview.names.json").read_text())
        descriptors = np.load(loc / "boq_references_sideview.npy").astype(np.float32)
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        self.descriptors = descriptors / np.maximum(norms, 1e-12)
        self.videos = np.array([_video(name) for name in self.names])
        self.centers = np.array([camera_center(self.pose(name)) for name in self.names])

    def pose(self, name: str) -> np.ndarray:
        return np.asarray(self.by_name[name].cam_from_world().matrix(), dtype=float)

    def image_path(self, name: str) -> Path:
        return self.release / "keyframes/images" / name

    @lru_cache(maxsize=2048)
    def observations(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        image = self.by_name[name]
        xy, xyz = [], []
        for point in image.points2D:
            if point.has_point3D():
                xy.append(point.xy)
                xyz.append(self.recon.points3D[point.point3D_id].xyz)
        return np.asarray(xy, dtype=float).reshape(-1, 2), np.asarray(xyz, dtype=float).reshape(-1, 3)

    def ranked(self, query_index: int, count: int) -> list[tuple[int, float]]:
        similarity = self.descriptors @ self.descriptors[query_index]
        similarity[self.videos == self.videos[query_index]] = -np.inf
        order = np.argsort(-similarity, kind="stable")[:count]
        return [(int(index), float(similarity[index])) for index in order]

    def nearest_other_video(self, query_index: int) -> float:
        other = self.videos != self.videos[query_index]
        return float(np.min(np.linalg.norm(self.centers[other] - self.centers[query_index], axis=1)))


def _load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"cannot read {path}")
    if size is not None:
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _install_xformers_shim() -> None:
    """Route ``xformers.ops.memory_efficient_attention`` to torch SDPA.

    The vendored STA blocks always wrap attention in xformers; installing
    xformers would pin a different torch into the production venv.  Same maths:
    inputs/outputs are (B, N, H, K) for xformers and (B, H, N, K) for SDPA.
    """

    try:
        import xformers.ops  # noqa: F401

        return
    except ImportError:
        pass
    import torch.nn.functional as F

    def memory_efficient_attention(q, k, v, scale=None, p=0.0):
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=p, scale=scale
        )
        return out.transpose(1, 2)

    package = types.ModuleType("xformers")
    ops = types.ModuleType("xformers.ops")
    ops.memory_efficient_attention = memory_efficient_attention
    package.ops = ops
    sys.modules["xformers"] = package
    sys.modules["xformers.ops"] = ops


def _vista_imports(vista: Path) -> None:
    _install_xformers_shim()
    source = str(vista / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


class StaRunner:
    def __init__(self, vista: Path, device: str = "cuda"):
        import torch

        _vista_imports(vista)
        from vista_slam.sta_model.sta_model import SymmetricTwoViewAssociation

        self.torch = torch
        self.device = device
        self.weights = vista / "pretrains/frontend_sta_weights.pth"
        model = SymmetricTwoViewAssociation()
        checkpoint = torch.load(self.weights, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        self.model = model.to(device).eval()
        self.shape = torch.tensor([[224, 224]])
        self._encoded: dict[str, Any] = {}

    def preprocess(self, rgb: np.ndarray):
        """ViSTA ``SLAM_image_only.process_image`` with the vendored cropping helpers."""

        import PIL.Image
        import torchvision.transforms as tvf
        from vista_slam.utils import cropping

        image = PIL.Image.fromarray(rgb)
        crop = sta_crop(image.size[0], image.size[1])
        bbox = tuple(int(value) for value in (crop.left, crop.top, crop.right, crop.bottom))
        image, _, _ = cropping.crop_image_depthmap(image, None, None, bbox)
        image, _, _ = cropping.rescale_image_depthmap(image, None, None, np.array([224, 224]))
        box = (int(crop.left2), int(crop.top2), int(crop.left2) + 224, int(crop.top2) + 224)
        image, _, _ = cropping.crop_image_depthmap(image, None, None, box)
        norm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        return norm(image).unsqueeze(0).to(self.device)

    def encode(self, rgb: np.ndarray):
        with self.torch.no_grad():
            return self.model._encode_image(self.preprocess(rgb), self.shape, normalize=False)

    def encode_cached(self, key: str, loader) -> tuple[Any, float]:
        if key in self._encoded:
            return self._encoded[key], 0.0
        started = self._sync()
        value = self.encode(loader())
        elapsed = (self._sync() - started) * 1000.0
        if len(self._encoded) >= 512:
            self._encoded.pop(next(iter(self._encoded)))
        self._encoded[key] = value
        return value, elapsed

    def _sync(self) -> float:
        self.torch.cuda.synchronize()
        return time.perf_counter()

    def pair(self, query, reference) -> dict[str, Any]:
        """STA(query=i, reference=j): ``T_{q,ref}`` and the reference-side pointmap."""

        torch = self.torch
        (q_feat, q_pos), (r_feat, r_pos) = query, reference
        started = self._sync()
        with torch.no_grad():
            dec_q, dec_r = self.model._decode_stereo(q_feat, r_feat, q_pos, r_pos)
            with torch.amp.autocast("cuda", enabled=False):
                pose = self.model.head_pose_s(dec_q[-1][:, 0, :])
                points = self.model.head_pts(
                    [r_feat] + [token[:, 1:, :].float() for token in dec_r], self.shape
                )
        elapsed = (self._sync() - started) * 1000.0
        return {
            "T": pose["pose"][0].double().cpu().numpy(),
            "conf": float(pose["conf"][0]),
            "depth": points["pts3d"][0, :, :, 2].float().cpu().numpy(),
            "depth_conf": points["conf"][0].float().cpu().numpy(),
            "decode_ms": elapsed,
        }


def _pose_errors(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    return (
        float(np.linalg.norm(camera_center(estimate) - camera_center(truth))),
        rotation_angle_deg(estimate[:3, :3], truth[:3, :3]),
    )


def _hypothesis(data: MapData, crop, ref: str, result: dict[str, Any], truth: np.ndarray) -> tuple[dict, Hypothesis | None]:
    xy, xyz = data.observations(ref)
    ref_pose = data.pose(ref)
    row: dict[str, Any] = {
        "reference": ref,
        "pose_conf": result["conf"],
        "ref_distance_u": float(np.linalg.norm(camera_center(ref_pose) - camera_center(truth))),
    }
    if len(xy):
        uv = np.floor(crop.to_crop(xy)).astype(int)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < 224) & (uv[:, 1] >= 0) & (uv[:, 1] < 224)
        z_map = (xyz[inside] @ ref_pose[:3, :3].T + ref_pose[:3, 3])[:, 2]
        uv = uv[inside]
        fit = estimate_gamma(
            z_map,
            result["depth"][uv[:, 1], uv[:, 0]],
            result["depth_conf"][uv[:, 1], uv[:, 0]],
            min_samples=PARAMS["min_scale_samples"],
        )
    else:
        fit = estimate_gamma(np.zeros(0), np.zeros(0), np.zeros(0), min_samples=PARAMS["min_scale_samples"])
    row.update(gamma=fit.gamma, scale_samples=fit.samples, scale_log_mad=fit.log_mad)
    if fit.gamma is None:
        row["valid"] = False
        return row, None
    estimate = compose_query_pose(ref_pose, result["T"], fit.gamma)
    row["position_error_u"], row["rotation_error_deg"] = _pose_errors(estimate, truth)
    row["sta_translation_u"] = float(fit.gamma * np.linalg.norm(result["T"][:3, 3]))
    row["valid"] = bool(
        result["conf"] >= PARAMS["pose_conf_min"] and fit.log_mad <= PARAMS["max_scale_log_mad"]
    )
    if not row["valid"]:
        return row, None
    return row, Hypothesis(ref, _video(ref), _frame(ref), estimate, result["conf"])


def _gpu_apps() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
    except OSError:
        return "unavailable"


def _completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["query"] for line in path.read_text().splitlines() if line.strip()}


def run_sta(args) -> None:
    import torch

    args.out.mkdir(parents=True, exist_ok=True)
    data = MapData(args.release)
    runner = StaRunner(args.vista)
    crop = sta_crop(data.width, data.height)
    path = args.out / "sta_results.jsonl"
    done = _completed(path)
    (args.out / "sta_provenance.json").write_text(json.dumps({
        "vista_commit": subprocess.run(["git", "-C", str(args.vista / "src"), "rev-parse", "HEAD"],
                                       capture_output=True, text=True, check=False).stdout.strip(),
        "sta_weights_sha256": _sha256(runner.weights),
        "params": PARAMS,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_apps_at_start": _gpu_apps(),
        "query_resize": LIVE_SIZE,
    }, indent=2))
    indices = range(len(data.names)) if args.limit is None else range(min(args.limit, len(data.names)))
    torch.cuda.reset_peak_memory_stats()
    with path.open("a") as sink:
        for query_index in indices:
            query = data.names[query_index]
            if query in done:
                continue
            truth = data.pose(query)
            started = runner._sync()
            q_enc = runner.encode(_load_rgb(data.image_path(query), LIVE_SIZE))
            query_encode_ms = (runner._sync() - started) * 1000.0
            rows, valid, decode_ms, ref_encode_ms = [], [], [], []
            for rank, (ref_index, similarity) in enumerate(data.ranked(query_index, PARAMS["top_k"])):
                ref = data.names[ref_index]
                r_enc, encode_ms = runner.encode_cached(ref, lambda ref=ref: _load_rgb(data.image_path(ref)))
                if encode_ms:
                    ref_encode_ms.append(encode_ms)
                result = runner.pair(q_enc, r_enc)
                decode_ms.append(result["decode_ms"])
                row, hypothesis = _hypothesis(data, crop, ref, result, truth)
                row.update(rank=rank, similarity=similarity)
                rows.append(row)
                if hypothesis is not None:
                    valid.append(hypothesis)
            record: dict[str, Any] = {
                "query": query,
                "video": _video(query),
                "nearest_other_video_u": data.nearest_other_video(query_index),
                "hypotheses": rows,
                "timing_ms": {
                    "query_encode": query_encode_ms,
                    "reference_encode": ref_encode_ms,
                    "pair_decode": decode_ms,
                },
            }
            first = next((row for row in rows if row["valid"]), None)
            record["top1"] = None if first is None else {
                key: first[key] for key in ("reference", "rank", "position_error_u", "rotation_error_deg", "ref_distance_u")
            }
            consensus = pose_consensus(
                valid,
                max_position=PARAMS["consensus_position_u"],
                max_rotation_deg=PARAMS["consensus_rotation_deg"],
                min_frame_gap=PARAMS["min_frame_gap"],
            )
            record["consensus"] = {"accepted": consensus.accepted, "members": list(consensus.members), "reason": consensus.reason}
            if consensus.accepted:
                position, rotation = _pose_errors(consensus.cam_from_world, truth)
                record["consensus"].update(position_error_u=position, rotation_error_deg=rotation)
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            print(f"{query_index + 1}/{len(data.names)} {query} consensus={consensus.accepted}", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**20
    (args.out / "sta_memory.json").write_text(json.dumps({"peak_allocated_mib": peak}, indent=2))


def run_edm(args) -> None:
    """Production EDM + PnP forced onto the same top-k cross-video references."""

    import cv2

    for relative in ("定位演算法/flight_control", "定位演算法/deploy_code/sfm_glomap_deploy",
                     "定位演算法/deploy_code/sfm_direct_deploy", "控制介面程式"):
        sys.path.insert(0, str(ROOT / relative))
    from production_localizer_factory import build_production_localizer
    from real_path_follow_controller import load_map_frame
    from river_map_quality.official_edm_adapter import prepare_official_megadepth_image_from_array

    args.out.mkdir(parents=True, exist_ok=True)
    site_path = ROOT / "地圖檔/場域/river_site/site_profile.json"
    site = json.loads(site_path.read_text())

    def asset(value: str) -> Path:
        return (site_path.parent / value).resolve()

    camera = site["query_camera"]
    built = build_production_localizer(
        backend="direct",
        bundle=asset(site["assets"]["localization_bundle"]),
        bundle_sha256=site["asset_sha256"]["localization_bundle"],
        production_profile=asset(site["localizer_profile"]),
        production_profile_sha256=site["asset_sha256"]["localizer_profile"],
        camera_tuple=(camera["model"], camera["width"], camera["height"], camera["params"]),
        frame_source=lambda: None,
        map_frame=load_map_frame(asset(site["map_align"])),
    )
    tracker = built.tracker.trk
    provider = tracker.provider
    provider.ensure_models()
    provider_index = {name: index for index, name in enumerate(provider._reference_names)}
    data = MapData(args.release)
    path = args.out / "edm_results.jsonl"
    done = _completed(path)
    indices = range(len(data.names)) if args.limit is None else range(min(args.limit, len(data.names)))
    with path.open("a") as sink:
        for query_index in indices:
            query = data.names[query_index]
            if query in done:
                continue
            refs = [data.names[i] for i, _ in data.ranked(query_index, PARAMS["edm_top_k"])]
            rgb = _load_rgb(data.image_path(query), LIVE_SIZE)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            gray = np.ascontiguousarray(cv2.resize(gray, (tracker._w, tracker._h), interpolation=cv2.INTER_AREA))
            provider.last_stage_ms = {}
            provider._query_matches.clear()
            started = time.perf_counter()
            prepared = prepare_official_megadepth_image_from_array(gray, provider._matcher_runtime())
            fix = provider._solve_references(
                gray, prepared, tuple(provider_index[name] for name in refs), started, False
            )
            record: dict[str, Any] = {
                "query": query,
                "references": refs,
                "status": fix.status,
                "ok": bool(fix.ok),
                "inliers": int(fix.inliers),
                "runtime_ms": (time.perf_counter() - started) * 1000.0,
            }
            if fix.cam_from_world is not None:
                position, rotation = _pose_errors(np.asarray(fix.cam_from_world), data.pose(query))
                record.update(position_error_u=position, rotation_error_deg=rotation)
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            print(f"{query_index + 1}/{len(data.names)} {query} {fix.status}", flush=True)
    provider.close()


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=float)
    return {"n": int(array.size), **{f"p{q}": float(np.percentile(array, q)) for q in (50, 90, 95)}}


def _accuracy(rows: list[dict[str, Any]], total: int) -> dict[str, Any]:
    positions = [row["position_error_u"] for row in rows]
    return {
        "accepted": len(rows),
        "accept_rate": len(rows) / total if total else 0.0,
        "position_u": _stats(positions),
        "rotation_deg": _stats([row["rotation_error_deg"] for row in rows]),
        "gross_rate": (
            sum(value > PARAMS["gross_error_u"] for value in positions) / len(positions) if positions else None
        ),
    }


def run_report(args) -> None:
    sta = [json.loads(line) for line in (args.out / "sta_results.jsonl").read_text().splitlines() if line.strip()]
    edm_path = args.out / "edm_results.jsonl"
    edm = {}
    if edm_path.exists():
        for line in edm_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                edm[row["query"]] = row
    total = len(sta)
    consensus = [row["consensus"] for row in sta if row["consensus"]["accepted"]]
    top1 = [row["top1"] for row in sta if row["top1"] is not None]
    report: dict[str, Any] = {
        "queries": total,
        "params": PARAMS,
        "criteria": CRITERIA,
        "sta_consensus": _accuracy(consensus, total),
        "sta_top1_valid": _accuracy(top1, total),
        "sta_all_hypotheses_with_scale": _accuracy(
            [h for row in sta for h in row["hypotheses"] if h.get("position_error_u") is not None],
            sum(len(row["hypotheses"]) for row in sta),
        ),
    }
    bins = [(0.0, 0.1), (0.1, 0.3), (0.3, math.inf)]
    report["sta_top1_by_ref_distance"] = {
        f"{low}-{high}": _accuracy(
            [row for row in top1 if low <= row["ref_distance_u"] < high],
            sum(low <= row["ref_distance_u"] < high for row in top1),
        )
        for low, high in bins
    }
    if edm:
        strong = [row for row in edm.values() if row["ok"]]
        report["edm_strong"] = _accuracy(strong, len(edm))
        paired = [
            (row, edm[row["query"]]) for row in sta
            if row["consensus"]["accepted"] and edm.get(row["query"], {}).get("ok")
        ]
        report["paired_both_accepted"] = {
            "n": len(paired),
            "sta_position_u": _stats([s["consensus"]["position_error_u"] for s, _ in paired]),
            "edm_position_u": _stats([e["position_error_u"] for _, e in paired]),
        }
        rescue = [row for row in sta if row["query"] in edm and not edm[row["query"]]["ok"]]
        rescued = [row["consensus"] for row in rescue if row["consensus"]["accepted"]]
        report["edm_failures_sta_consensus"] = {"edm_failures": len(rescue), **_accuracy(rescued, len(rescue))}
        report["edm_runtime_ms"] = _stats([row["runtime_ms"] for row in edm.values()])
    decode = [value for row in sta for value in row["timing_ms"]["pair_decode"]]
    report["sta_timing_ms"] = {
        "query_encode": _stats([row["timing_ms"]["query_encode"] for row in sta]),
        "reference_encode": _stats([v for row in sta for v in row["timing_ms"]["reference_encode"]]),
        "pair_decode": _stats(decode),
    }
    head = report["sta_consensus"]
    position, rotation = head["position_u"], head["rotation_deg"]
    checks = {
        "position_p50": position.get("p50", math.inf) <= CRITERIA["position_p50_u"],
        "position_p95": position.get("p95", math.inf) <= CRITERIA["position_p95_u"],
        "rotation_p95": rotation.get("p95", math.inf) <= CRITERIA["rotation_p95_deg"],
        "gross_rate": (head["gross_rate"] if head["gross_rate"] is not None else 1.0) <= CRITERIA["gross_rate_max"],
        "accept_rate": head["accept_rate"] >= CRITERIA["accept_rate_min"],
    }
    report["decision"] = {"checks": checks, "go": all(checks.values())}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def run_selfcheck(args) -> None:
    import torch

    _vista_imports(args.vista)
    from vista_slam.sta_model.blocks.sta_blocks import Attention, XFormer_Attention

    checks: dict[str, Any] = {}
    torch.manual_seed(0)
    plain = Attention(64, num_heads=4, qkv_bias=True).cuda().eval()
    wrapped = XFormer_Attention(plain).cuda().eval()
    x = torch.randn(2, 50, 64, device="cuda")
    with torch.no_grad():
        difference = float((plain(x, None) - wrapped(x, None)).abs().max())
    checks["xformers_shim_max_abs_diff"] = difference

    runner = StaRunner(args.vista)
    crop_errors = []
    for width, height in ((2688, 1512), (1280, 720)):
        crop = sta_crop(width, height)
        for fx, fy in ((0.5, 0.5), (0.3, 0.2), (0.7, 0.8), (0.62, 0.05)):
            image = np.zeros((height, width, 3), dtype=np.uint8)
            x0, y0 = int(fx * width), int(fy * height)
            image[y0 - 8 : y0 + 9, x0 - 8 : x0 + 9] = 255
            # Blob centroid, not argmax: Lanczos ringing moves the peak pixel.
            weight = np.clip(runner.preprocess(image)[0, 0].cpu().numpy() + 1.0, 0.0, None)
            rows, cols = np.mgrid[0:224, 0:224]
            centroid = np.array([(weight * (cols + 0.5)).sum(), (weight * (rows + 0.5)).sum()]) / weight.sum()
            expected = crop.to_crop(np.array([[x0 + 0.5, y0 + 0.5]]))[0]
            if 0 <= expected[0] < 224 and 0 <= expected[1] < 224:
                crop_errors.append(float(np.linalg.norm(centroid - expected)))
    checks["crop_mapping_max_px"] = max(crop_errors)
    checks["crop_mapping_cases"] = len(crop_errors)

    data = MapData(args.release)
    as_is, transposed, t_angle, t_angle_inverse = [], [], [], []
    names = set(data.names)
    for name in data.names[::31]:
        neighbour = f"{_video(name)}/frame_{_frame(name) + 24:08d}.jpg"
        if neighbour not in names:
            continue
        query, ref = data.pose(name), data.pose(neighbour)
        truth_R = ref[:, :3] @ query[:, :3].T
        truth_t = ref[:, 3] - truth_R @ query[:, 3]
        result = runner.pair(runner.encode(_load_rgb(data.image_path(name))), runner.encode(_load_rgb(data.image_path(neighbour))))
        R, t = result["T"][:3, :3], result["T"][:3, 3]
        as_is.append(rotation_angle_deg(R, truth_R))
        transposed.append(rotation_angle_deg(R.T, truth_R))

        def angle(a, b):
            return math.degrees(math.acos(float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1))))

        t_angle.append(angle(t, truth_t))
        t_angle_inverse.append(angle(-R.T @ t, truth_t))
    checks["convention_pairs"] = len(as_is)
    checks["rotation_error_deg_median"] = {"T_is_ref_from_query": float(np.median(as_is)), "transposed": float(np.median(transposed))}
    checks["translation_dir_error_deg_median"] = {"T_is_ref_from_query": float(np.median(t_angle)), "inverse": float(np.median(t_angle_inverse))}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selfcheck.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("selfcheck", "sta", "edm", "report"))
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--vista", type=Path, default=VISTA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=None, help="first N bank queries only (smoke)")
    args = parser.parse_args(argv)
    {"selfcheck": run_selfcheck, "sta": run_sta, "edm": run_edm, "report": run_report}[args.command](args)


if __name__ == "__main__":
    main()
