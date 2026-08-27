#!/usr/bin/env python3
"""Run the EDM tracker over a held-out query video. Site-agnostic: bundle and camera come
from config.json, so it works on any packaged map.

These videos were never mapped, so there is NO ground-truth pose to score against. What is
measurable without one, and is reported here:

  - localization rate and the tracker's state distribution (TRACK / WEAK_TRACK / LOST)
  - PnP inliers and correspondence counts
  - latency breakdown (retrieval / matching / PnP)
  - trajectory continuity: the per-frame step in map units. At 24 fps a real flight moves a
    small, smooth amount per frame, so a heavy tail here means the pose is jumping around --
    the one failure mode a rate-only summary would hide.

Nothing here proves absolute accuracy. It shows whether the map SUPPORTS this flight.

`--stress-matrix` replays the same query under a fixed 2D appearance / image-plane transform
matrix. Those results are stress tests, not proof of 3D parallax generalization. Transforms
never write map or profile assets and must keep the camera contract (model/size/params) and
exact frame dimensions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy"))
from production_edm_tracker import EDMConfig, ProductionEDMTracker  # noqa: E402
from reloc_localizer_edm import Camera, EDMRelocMap  # noqa: E402


STRESS_SEED = 20260826
STRESS_SCHEMA = "edm-video-stress-matrix/v1"
ROBUSTNESS_KIND = "2d_appearance_image_plane"
ROBUSTNESS_NOTE = (
    "2D appearance/image-plane robustness only. "
    "Synthetic transforms are stress tests, not proof of 3D parallax generalization."
)
_TRACK_STATES = ("TRACK", "WEAK_TRACK", "LOST")


def _op(name: str, **params) -> dict:
    return {"op": name, **params}


STRESS_MATRIX: tuple[dict, ...] = (
    {"id": "identity", "ops": ()},
    {"id": "hcrop_0.10", "ops": (_op("crop", axis="horizontal", fraction=0.10),)},
    {"id": "vcrop_0.10", "ops": (_op("crop", axis="vertical", fraction=0.10),)},
    {"id": "rotate_3", "ops": (_op("rotate", degrees=3.0),)},
    {"id": "exposure_1.25", "ops": (_op("exposure", gain=1.25),)},
    {"id": "gamma_0.75", "ops": (_op("gamma", gamma=0.75),)},
    {"id": "contrast_1.20", "ops": (_op("contrast", factor=1.20),)},
    {"id": "blur_k5_s1.25", "ops": (_op("blur", ksize=5, sigma=1.25),)},
    {"id": "jpeg_70", "ops": (_op("jpeg", quality=70),)},
    {
        "id": "hcrop_0.10+jpeg_70",
        "ops": (_op("crop", axis="horizontal", fraction=0.10), _op("jpeg", quality=70)),
    },
    {
        "id": "vcrop_0.10+rotate_3",
        "ops": (_op("crop", axis="vertical", fraction=0.10), _op("rotate", degrees=3.0)),
    },
    {
        "id": "rotate_3+blur_k5_s1.25",
        "ops": (_op("rotate", degrees=3.0), _op("blur", ksize=5, sigma=1.25)),
    },
    {
        "id": "exposure_1.25+contrast_1.20",
        "ops": (_op("exposure", gain=1.25), _op("contrast", factor=1.20)),
    },
    {
        "id": "gamma_0.75+jpeg_70",
        "ops": (_op("gamma", gamma=0.75), _op("jpeg", quality=70)),
    },
    {
        "id": "blur_k5_s1.25+jpeg_70",
        "ops": (_op("blur", ksize=5, sigma=1.25), _op("jpeg", quality=70)),
    },
)


def camera_contract(camera) -> dict:
    params = [float(value) for value in list(camera.params)]
    if any(not math.isfinite(value) for value in params):
        raise ValueError("camera params must be finite")
    return {
        "model": str(camera.model),
        "width": int(camera.width),
        "height": int(camera.height),
        "params": params,
    }


def require_frame_camera_contract(frame: np.ndarray, camera) -> None:
    height, width = frame.shape[:2]
    if int(width) != int(camera.width) or int(height) != int(camera.height):
        raise ValueError(
            f"frame {width}x{height} does not match camera "
            f"{int(camera.width)}x{int(camera.height)}"
        )


def frame_sha256(frame: np.ndarray) -> str:
    payload = np.ascontiguousarray(frame)
    return hashlib.sha256(payload.tobytes()).hexdigest()


def sequence_sha256(digests: list[str], seed: int) -> str:
    hasher = hashlib.sha256()
    hasher.update(STRESS_SCHEMA.encode("ascii"))
    hasher.update(int(seed).to_bytes(8, "little", signed=True))
    for digest in digests:
        hasher.update(bytes.fromhex(digest))
    return hasher.hexdigest()


def _to_u8(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 255.0).astype(np.uint8)


def _crop_resize(frame: np.ndarray, axis: str, fraction: float) -> np.ndarray:
    height, width = frame.shape[:2]
    amount = float(fraction)
    if not math.isfinite(amount) or amount <= 0.0 or amount >= 0.5:
        raise ValueError("crop fraction must be finite and in (0, 0.5)")
    if axis == "horizontal":
        cut = min(max(int(round(width * amount * 0.5)), 1), (width - 1) // 2)
        cropped = frame[:, cut:width - cut]
    elif axis == "vertical":
        cut = min(max(int(round(height * amount * 0.5)), 1), (height - 1) // 2)
        cropped = frame[cut:height - cut, :]
    else:
        raise ValueError(f"unsupported crop axis: {axis}")
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def _rotate(frame: np.ndarray, degrees: float) -> np.ndarray:
    angle = float(degrees)
    if not math.isfinite(angle):
        raise ValueError("rotation degrees must be finite")
    height, width = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _exposure(frame: np.ndarray, gain: float) -> np.ndarray:
    value = float(gain)
    if not math.isfinite(value):
        raise ValueError("exposure gain must be finite")
    return _to_u8(frame.astype(np.float32) * value)


def _gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
    value = float(gamma)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("gamma must be finite and positive")
    return _to_u8(np.power(frame.astype(np.float32) / 255.0, value) * 255.0)


def _contrast(frame: np.ndarray, factor: float) -> np.ndarray:
    value = float(factor)
    if not math.isfinite(value):
        raise ValueError("contrast factor must be finite")
    return _to_u8((frame.astype(np.float32) - 127.5) * value + 127.5)


def _blur(frame: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    size = int(ksize)
    spread = float(sigma)
    if size < 1 or size % 2 == 0:
        raise ValueError("blur ksize must be a positive odd integer")
    if not math.isfinite(spread) or spread < 0.0:
        raise ValueError("blur sigma must be finite and non-negative")
    return cv2.GaussianBlur(
        frame, (size, size), spread, borderType=cv2.BORDER_REPLICATE,
    )


def _jpeg(frame: np.ndarray, quality: int) -> np.ndarray:
    level = int(quality)
    if level < 1 or level > 100:
        raise ValueError("jpeg quality must be in [1, 100]")
    ok, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), level],
    )
    if not ok:
        raise RuntimeError("jpeg encode failed")
    flag = cv2.IMREAD_GRAYSCALE if frame.ndim == 2 else cv2.IMREAD_COLOR
    decoded = cv2.imdecode(buffer, flag)
    if decoded is None:
        raise RuntimeError("jpeg decode failed")
    return decoded


def _apply_op(frame: np.ndarray, op: dict) -> np.ndarray:
    name = op["op"]
    if name == "crop":
        return _crop_resize(frame, op["axis"], op["fraction"])
    if name == "rotate":
        return _rotate(frame, op["degrees"])
    if name == "exposure":
        return _exposure(frame, op["gain"])
    if name == "gamma":
        return _gamma(frame, op["gamma"])
    if name == "contrast":
        return _contrast(frame, op["factor"])
    if name == "blur":
        return _blur(frame, op["ksize"], op["sigma"])
    if name == "jpeg":
        return _jpeg(frame, op["quality"])
    raise ValueError(f"unsupported stress op: {name}")


def apply_stress_transform(
    frame: np.ndarray,
    spec: dict,
    camera=None,
) -> np.ndarray:
    """Apply a finite, deterministic 2D transform. Output matches input HxW exactly."""
    source = np.ascontiguousarray(frame)
    output = source.copy()
    for op in spec.get("ops") or ():
        output = np.ascontiguousarray(_apply_op(output, op))
        if output.shape != source.shape:
            raise RuntimeError(
                f"{spec.get('id', op['op'])} changed shape {source.shape} -> {output.shape}"
            )
        if output.dtype != np.uint8:
            output = _to_u8(output)
    if camera is not None:
        require_frame_camera_contract(output, camera)
    return output


def transform_record(
    frame_in: np.ndarray,
    frame_out: np.ndarray,
    spec: dict,
    seed: int = STRESS_SEED,
) -> dict:
    return {
        "id": spec["id"],
        "ops": [dict(op) for op in spec.get("ops") or ()],
        "seed": int(seed),
        "input_sha256": frame_sha256(frame_in),
        "output_sha256": frame_sha256(frame_out),
    }


def longest_run(values: list[str], target: str) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value == target else 0
        best = max(best, current)
    return best


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    value = float(np.median(np.asarray(values, dtype=float)))
    if not math.isfinite(value):
        raise ValueError("metric median is not finite")
    return value


def _delta(current, baseline):
    if current is None or baseline is None:
        return None
    return float(current) - float(baseline)


def summarize_stress_frames(infos: list[dict]) -> dict:
    n = len(infos)
    accepted = [row for row in infos if row.get("ok")]
    states = [str(row.get("state_out", "UNKNOWN")) for row in infos]
    counts = Counter(states)
    state_counts = {name: int(counts.get(name, 0)) for name in _TRACK_STATES}
    state_counts["WEAK"] = state_counts["WEAK_TRACK"]
    inliers = [
        float(row["inliers"])
        for row in accepted
        if row.get("inliers") is not None and math.isfinite(float(row["inliers"]))
    ]
    reproj = [
        float(row["reproj_rms"])
        for row in accepted
        if row.get("reproj_rms") is not None and math.isfinite(float(row["reproj_rms"]))
    ]
    latency = [
        float(row["total_ms"])
        for row in infos
        if row.get("total_ms") is not None and math.isfinite(float(row["total_ms"]))
    ]
    return {
        "frames": n,
        "accepted": len(accepted),
        "accepted_rate": (len(accepted) / n) if n else 0.0,
        "states": state_counts,
        "longest_lost": longest_run(states, "LOST"),
        "inliers_median": _median(inliers),
        "reproj_rms_median": _median(reproj),
        "latency_median_ms": _median(latency),
    }


def stress_deltas(metrics: dict, baseline: dict) -> dict:
    state_delta = {
        name: int(metrics["states"].get(name, 0)) - int(baseline["states"].get(name, 0))
        for name in (*_TRACK_STATES, "WEAK")
    }
    return {
        "accepted_rate": _delta(metrics["accepted_rate"], baseline["accepted_rate"]),
        "longest_lost": _delta(metrics["longest_lost"], baseline["longest_lost"]),
        "inliers_median": _delta(metrics["inliers_median"], baseline["inliers_median"]),
        "reproj_rms_median": _delta(metrics["reproj_rms_median"], baseline["reproj_rms_median"]),
        "latency_median_ms": _delta(metrics["latency_median_ms"], baseline["latency_median_ms"]),
        "states": state_delta,
    }


def build_stress_report(
    *,
    video: str,
    bundle: str,
    camera,
    seed: int,
    rows: list[dict],
) -> dict:
    if not rows or rows[0]["id"] != "identity":
        raise ValueError("stress matrix must start with the unmodified identity baseline")
    baseline_metrics = rows[0]["metrics"]
    transforms = []
    for row in rows:
        item = {
            "id": row["id"],
            "ops": row["ops"],
            "seed": int(seed),
            "input_sha256": row["input_sha256"],
            "output_sha256": row["output_sha256"],
            "probe": row["probe"],
            **row["metrics"],
            "delta_vs_baseline": stress_deltas(row["metrics"], baseline_metrics),
        }
        transforms.append(item)
    return {
        "schema": STRESS_SCHEMA,
        "robustness_kind": ROBUSTNESS_KIND,
        "note": ROBUSTNESS_NOTE,
        "seed": int(seed),
        "video": str(video),
        "bundle": str(bundle),
        "camera": camera_contract(camera),
        "frames": int(baseline_metrics["frames"]),
        "transforms": transforms,
    }


def _resize_to_camera(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if (frame.shape[1], frame.shape[0]) != (width, height):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def load_query_frames(
    video: str,
    width: int,
    height: int,
    max_frames: int,
    stride: int,
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    frames: list[np.ndarray] = []
    index = seen = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and seen >= max_frames):
                break
            index += 1
            if (index - 1) % stride:
                continue
            seen += 1
            frames.append(np.ascontiguousarray(_resize_to_camera(frame, width, height)))
    finally:
        cap.release()
    return frames


def run_stress_matrix(
    frames: list[np.ndarray],
    rmap,
    camera,
    cfg: EDMConfig,
    *,
    video: str,
    bundle: str,
    seed: int = STRESS_SEED,
    matrix: tuple[dict, ...] = STRESS_MATRIX,
) -> dict:
    contract = camera_contract(camera)
    rows = []
    for spec in matrix:
        tracker = ProductionEDMTracker(rmap, camera, cfg)
        infos = []
        input_digests = []
        output_digests = []
        probe = None
        for frame in frames:
            require_frame_camera_contract(frame, camera)
            warped = apply_stress_transform(frame, spec, camera=camera)
            record = transform_record(frame, warped, spec, seed=seed)
            input_digests.append(record["input_sha256"])
            output_digests.append(record["output_sha256"])
            if probe is None:
                probe = {
                    "input_sha256": record["input_sha256"],
                    "output_sha256": record["output_sha256"],
                }
            infos.append(tracker.localize(warped))
        if camera_contract(camera) != contract:
            raise RuntimeError("stress replay mutated the camera contract")
        rows.append({
            "id": spec["id"],
            "ops": [dict(op) for op in spec.get("ops") or ()],
            "input_sha256": sequence_sha256(input_digests, seed),
            "output_sha256": sequence_sha256(output_digests, seed),
            "probe": probe or {"input_sha256": None, "output_sha256": None},
            "metrics": summarize_stress_frames(infos),
        })
    return build_stress_report(
        video=video, bundle=bundle, camera=camera, seed=seed, rows=rows,
    )


def _print_stress_report(report: dict) -> None:
    print("\n" + "=" * 68)
    print(report["note"])
    print(f"seed={report['seed']}  frames={report['frames']}  kind={report['robustness_kind']}")
    print(
        f"{'id':<28} {'accept':>7} {'TRACK':>6} {'WEAK':>5} {'LOST':>5} "
        f"{'lostRun':>7} {'inl':>6} {'reproj':>7} {'ms':>7} {'d_accept':>8}"
    )
    for row in report["transforms"]:
        states = row["states"]
        inl = row["inliers_median"]
        reproj = row["reproj_rms_median"]
        latency = row["latency_median_ms"]
        delta = row["delta_vs_baseline"]["accepted_rate"]
        print(
            f"{row['id']:<28} {row['accepted_rate']:7.3f} "
            f"{states['TRACK']:6d} {states['WEAK']:5d} {states['LOST']:5d} "
            f"{row['longest_lost']:7d} "
            f"{(f'{inl:.0f}' if inl is not None else '-'):>6} "
            f"{(f'{reproj:.3f}' if reproj is not None else '-'):>7} "
            f"{(f'{latency:.1f}' if latency is not None else '-'):>7} "
            f"{(f'{delta:+.3f}' if delta is not None else '-'):>8}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--local-topk", type=int, default=1)
    ap.add_argument("--boot-global-topk", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--stress-matrix",
        action="store_true",
        help="replay a fixed 2D appearance/image-plane transform matrix (not 3D parallax)",
    )
    ap.add_argument("--stress-seed", type=int, default=STRESS_SEED)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    pc = cfg["pnp_camera"]
    cam = Camera(model=pc["model"], width=pc["width"], height=pc["height"], params=list(pc["params"]))
    W, H = pc["width"], pc["height"]

    bundle = Path(
        args.bundle or (Path(args.config).parent / cfg["paths"]["bundle"])
    ).resolve()
    if not bundle.is_file():
        fallback = (ROOT / "outputs" / bundle.name).resolve()
        if not fallback.is_file():
            raise SystemExit(f"localization bundle not found: {bundle}")
        bundle = fallback
    digest = hashlib.sha256()
    with bundle.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    rmap = EDMRelocMap.load(bundle, expected_sha256=digest.hexdigest())
    print(f"map: {len(rmap.ref_names)} refs   camera: {pc['model']} {W}x{H} f={pc['params'][0]}")
    print(f"video: {Path(args.video).name}   local_topk={args.local_topk}")

    tracker_cfg = EDMConfig(local_topk=args.local_topk,
                            boot_global_topk=args.boot_global_topk)
    if args.stress_matrix:
        frames = load_query_frames(args.video, W, H, args.max_frames, args.stride)
        report = run_stress_matrix(
            frames,
            rmap,
            cam,
            tracker_cfg,
            video=args.video,
            bundle=bundle,
            seed=int(args.stress_seed),
        )
        _print_stress_report(report)
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2))
            print(f"\n-> {args.out}")
        return

    trk = ProductionEDMTracker(rmap, cam, tracker_cfg)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")

    states, lat, vpr, match, pnp, inl, ncorr, centers = [], [], [], [], [], [], [], []
    reference_sequences, rejections = Counter(), Counter()
    limited_jumps = 0
    i = n_seen = 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n_seen >= args.max_frames):
            break
        i += 1
        if (i - 1) % args.stride:
            continue
        n_seen += 1
        frame = _resize_to_camera(frame, W, H)
        info = trk.localize(frame)
        reference_sequences.update(name.split("/", 1)[0] for name in info.get("refs", []))
        if info.get("rejected"):
            rejections[str(info["rejected"])] += 1
        if info.get("limited_jump"):
            limited_jumps += 1
        states.append(info["state_out"])
        lat.append(info["total_ms"])
        vpr.append(info["vpr_ms"])
        match.append(info["match_ms"])
        pnp.append(info["pnp_ms"])
        if info.get("ok"):
            inl.append(info["inliers"])
            ncorr.append(info["n_corr"])
            centers.append((n_seen, np.asarray(info["center"], float)))
        if n_seen % 300 == 0:
            print(f"  {n_seen}  state={info['state_out']} inliers={info.get('inliers', 0)} "
                  f"{np.median(lat[-300:]):.0f}ms", flush=True)
    cap.release()
    wall = time.perf_counter() - t0

    n_ok = len(inl)
    print("\n" + "=" * 68)
    print(f"frames={n_seen}  localized={n_ok} ({100*n_ok/max(n_seen,1):.1f}%)  wall={wall:.0f}s")
    print(f"states: {dict(Counter(states))}")
    print(f"limited pose-center spikes: {limited_jumps}")
    if n_ok:
        print(f"correspondences median={np.median(ncorr):.0f}   PnP inliers median={np.median(inl):.0f}"
              f"  p05={np.percentile(inl,5):.0f}")
    print(f"latency ms (median): total={np.median(lat):.1f}  retrieval={np.median(vpr):.1f}  "
          f"match={np.median(match):.1f}  pnp={np.median(pnp):.1f}   -> {1000/max(np.median(lat),1e-6):.1f} FPS")

    steps = []
    for (a, ca), (b, cb) in zip(centers, centers[1:]):
        if b - a == 1:
            steps.append(float(np.linalg.norm(cb - ca)))
    if steps:
        s = np.array(steps)
        print(f"trajectory step (map-u, consecutive localized frames, n={len(s)}): "
              f"median={np.median(s):.4f} p95={np.percentile(s,95):.4f} max={s.max():.4f}")
        print(f"  steps > 10x median (pose jumps): {int((s > 10*np.median(s)).sum())}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "video": str(args.video), "bundle": str(bundle), "local_topk": args.local_topk,
            "frames": n_seen, "localized": n_ok, "rate": n_ok / max(n_seen, 1),
            "states": dict(Counter(states)),
            "inliers_median": float(np.median(inl)) if n_ok else None,
            "inliers_p05": float(np.percentile(inl, 5)) if n_ok else None,
            "correspondences_median": float(np.median(ncorr)) if n_ok else None,
            "reference_sequence_counts": dict(reference_sequences),
            "rejections": dict(rejections),
            "limited_jumps": limited_jumps,
            "latency_median_ms": {"total": float(np.median(lat)), "retrieval": float(np.median(vpr)),
                                  "match": float(np.median(match)), "pnp": float(np.median(pnp))},
            "step_median": float(np.median(steps)) if steps else None,
            "step_p95": float(np.percentile(steps, 95)) if steps else None,
            "step_max": float(np.max(steps)) if steps else None,
            "jumps_gt_10x_median": int((np.array(steps) > 10 * np.median(steps)).sum()) if steps else None,
        }, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
