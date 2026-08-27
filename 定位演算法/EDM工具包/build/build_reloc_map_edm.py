#!/usr/bin/env python3
"""Build an EDM relocalization bundle by fixed-pose re-triangulation.

The XFeat bundle cannot be reused: EDM is detector-free, so XFeat's keypoints and
64-D descriptors are dead weight to it. This rebuilds the 2D->3D layer natively from
EDM correspondences, on the SAME poses, so nothing about the map geometry moves.

HOW A DETECTOR-FREE MATCHER GETS CONSISTENT KEYPOINTS (the hard part):
  EDM emits different keypoints for every pair, so there is no per-image feature table
  to triangulate against. We recover one from EDM's own structure: every match keeps
  one side on the 1/8 coarse grid and the fine offset is clamped to +/-4px, so
      cell = round(kpt / 8)
  is an image-intrinsic identity that is stable across pairs (verified exact on this
  site: tests/smoke_edm_epipolar.py reports 100% round-trip).

  Every match is asymmetric: one side is a cell CENTRE, the other is that centre's
  subpixel correspondence. So a match is one precise observation of the landmark
  "image X, cell c centre" -- an exact, deterministic, image-intrinsic anchor.

  Subpixel positions are NOT merged across pairs. They must not be: EDM refines toward
  the PARTNER's cell centre, so a different partner legitimately lands on a different
  sub-location inside the 8px cell (measured median spread 2.07px, not a tail --
  tests/analyze_cell_spread.py). Averaging them blends distinct physical points.
  See build_keypoint_tables() for the anchor-centric table this implies.

Stages (each cached, so a later stage can be retuned without re-running EDM):
  A  covisibility pairs               (topology identical to the XFeat build)
  B  EDM over all pairs               -> edm_pairs.h5   (cells + conf per pair)
  C  per-image keypoint tables        -> feats-edm.h5
  D  pair matches as index arrays     -> matches-edm.h5
  E  hloc fixed-pose triangulation    -> COLMAP model
  F  pack bundle                      -> cell->xyz LUT + reference JPEGs + MegaLoc

The bundle is self-contained: it carries the reference images (EDM needs to SEE them at
flight time), so deployment no longer depends on an external image directory.
"""
from __future__ import annotations

import argparse
import gc
import shutil
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import pycolmap
import torch
from hloc import triangulation
from hloc.utils.io import names_to_pair

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
from edm_matcher import COARSE_STRIDE, EDM_H, EDM_W, GRID_H, GRID_W, EDMMatcher  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "validation"))
from edm_coverage_sidecar import (  # noqa: E402
    CoverageRecords,
    extract_coverage_records,
    merge_records,
    write_coverage_sidecar,
)

N_CELLS = GRID_W * GRID_H
KEYPOINT_IDENTITY = (
    "coarse cell = round(kpt/8); landmark position = anchor cell centre; "
    "refined partner observations kept separate"
)

DEF_WORK = Path(__file__).resolve().parent.parent / "outputs" / "river_edm_work"
DEF_OUT = Path(__file__).resolve().parent.parent / "outputs" / "river_site_reloc_map_edm.pt"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def restrict_reconstruction_to_active_refs(rec, active_names) -> list[str]:
    """Deregister posed images lacking EDM features from a triangulation copy."""
    active = set(active_names)
    removed = sorted(
        image.name
        for image in rec.images.values()
        if image.has_pose and image.name not in active
    )
    for name in removed:
        image = next(image for image in rec.images.values() if image.name == name)
        rec.deregister_frame(image.frame_id)
    if int(rec.num_reg_images()) != len(active):
        raise ValueError(
            f"active triangulation model mismatch: {rec.num_reg_images()} != {len(active)}"
        )
    return removed


def iter_anchor_batches(ref_names, batch_size: int):
    """Partition anchor references while preserving their canonical order."""
    names = list(ref_names)
    if batch_size <= 0:
        yield names
        return
    for start in range(0, len(names), batch_size):
        yield names[start : start + batch_size]


def pairs_for_anchor_batch(pairs, anchor_names):
    """Return every pair that can observe a landmark anchored in this batch."""
    anchors = set(anchor_names)
    return [pair for pair in pairs if pair[0] in anchors or pair[1] in anchors]


def anchor_batch_cache_name(
    pair_topk: int, batch_size: int, skip_geometric_verification: bool
) -> str:
    """Keep checkpoints from different geometry policies physically separate."""
    geometry = "skip_geometry" if skip_geometric_verification else "verified"
    return f"anchor_batches_top{pair_topk}_size{batch_size}_{geometry}"


def edm_pair_cache_name(pair_topk: int) -> str:
    """Bind the expensive matcher artifact to its covisibility breadth."""
    return f"edm_pairs_top{pair_topk}.h5"


def anchored_cell_count(xyz_by_cell) -> int:
    return int(np.isfinite(np.asarray(xyz_by_cell)[:, 0]).sum())


def zero_anchor_names(xyz_by_name) -> list[str]:
    return sorted(
        name for name, xyz in xyz_by_name.items() if anchored_cell_count(xyz) == 0
    )


def apply_zero_anchor_repairs(xyz_by_name, repairs) -> list[str]:
    """Replace empty LUTs only; never overwrite an already triangulated anchor."""
    repaired = []
    for name in zero_anchor_names(xyz_by_name):
        if name not in repairs or anchored_cell_count(repairs[name]) == 0:
            raise RuntimeError(f"zero-anchor repair produced no 3D cells for {name}")
        xyz_by_name[name] = repairs[name]
        repaired.append(name)
    return repaired


# ---------------------------------------------------------------- stage A
def pair_list_from_covis(ref_names, covis, pair_topk, active=None):
    active = list(active) if active is not None else list(ref_names)
    active_set = set(active)
    idx = {n: i for i, n in enumerate(ref_names)}
    seen, pairs = set(), []
    for n0 in active:
        for j in covis.get(n0, [])[:pair_topk]:
            j = int(j)
            if not (0 <= j < len(ref_names)):
                continue
            n1 = ref_names[j]
            if n1 not in active_set or n1 == n0:
                continue
            key = (min(idx[n0], idx[n1]), max(idx[n0], idx[n1]))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((n0, n1))
    return pairs


# ---------------------------------------------------------------- stage B
def run_edm_pairs(path: Path, pairs, image_root: Path, mconf_thr: float,
                  batch: int, overwrite: bool):
    if path.exists() and not overwrite:
        log(f"reuse EDM pairs {path}")
        return
    matcher = EDMMatcher(mconf_thr=mconf_thr)
    log(f"EDM over {len(pairs)} pairs (topk={matcher.topk}, batch={batch}) -> {path}")

    by_anchor: dict[str, list[str]] = {}
    for n0, n1 in pairs:
        by_anchor.setdefault(n0, []).append(n1)

    gray_cache: dict[str, np.ndarray] = {}

    def gray(name):
        if name not in gray_cache:
            if len(gray_cache) > 600:
                gray_cache.clear()
            gray_cache[name] = EDMMatcher.load_gray(image_root / name)
        return gray_cache[name]

    done, t0 = 0, time.perf_counter()
    tmp = path.with_suffix(".tmp.h5")
    with h5py.File(tmp, "w") as h5:
        for n0, partners in by_anchor.items():
            g0 = gray(n0)
            for s in range(0, len(partners), batch):
                chunk = partners[s:s + batch]
                res = matcher.match_one_to_many(g0, [gray(n) for n in chunk])
                for n1, r in zip(chunk, res):
                    k0, k1, mc = r["mkpts0"], r["mkpts1"], r["mconf"]
                    g = h5.create_group(names_to_pair(n0, n1))
                    g.create_dataset("cell0", data=EDMMatcher.cell_ids(k0).astype(np.int32))
                    g.create_dataset("cell1", data=EDMMatcher.cell_ids(k1).astype(np.int32))
                    g.create_dataset("kpt0", data=k0.astype(np.float32))
                    g.create_dataset("kpt1", data=k1.astype(np.float32))
                    g.create_dataset("conf", data=mc.astype(np.float32))
                    done += 1
                if done % 200 < batch:
                    r_ = done / max(time.perf_counter() - t0, 1e-9)
                    log(f"  pairs {done}/{len(pairs)} ({r_:.1f}/s)")
    tmp.rename(path)
    log(f"  EDM done: {done} pairs in {time.perf_counter()-t0:.0f}s")


# ---------------------------------------------------------------- stage C
def build_keypoint_tables(
    pairs_h5: Path,
    ref_names,
    pairs,
    scale_of: dict[str, float],
    *,
    anchor_names=None,
):
    """Anchor-centric keypoint tables. NO cross-pair merging of subpixel positions.

    WHY NOT MERGE (measured, tests/analyze_cell_spread.py):
      A cell's refined observations disagree by 2.07px at the median -- not a heavy tail,
      the whole distribution sits there. That is not noise: EDM's fine head regresses the
      correspondence of the PARTNER's cell centre, so a different partner legitimately
      refines to a different sub-location inside the 8px cell. Averaging them blends
      distinct physical points, and the earlier 1.5px "ambiguity" gate then threw away
      78% of all cells.

    THE ANCHOR SEMANTICS INSTEAD:
      Each match is asymmetric: one side is a cell CENTRE (exact, deterministic,
      image-intrinsic), the other is that centre's subpixel correspondence. So read every
      match as one precise observation of the landmark "image X, cell c centre":
        direction 01 -> landmark (n0, cell0), observed in n1 at the refined point
        direction 10 -> landmark (n1, cell1), observed in n0 at the refined point

      Keypoints per image are therefore:
        - the cell centres it anchors      (deduped by cell -> stable identity)
        - the refined points it contributes to other images' anchors (kept SEPARATE;
          each belongs to one anchor, so merging them would re-introduce the blur)

      COLMAP's track builder unions matches through the shared anchor keypoint, so a
      landmark seen by 10 partners becomes one 10-view track without needing any
      partner-to-partner match.
    """
    anchor_set = set(ref_names) if anchor_names is None else set(anchor_names)
    unknown_anchors = anchor_set - set(ref_names)
    if unknown_anchors:
        raise ValueError(f"anchor names absent from reference table: {sorted(unknown_anchors)}")
    grid_idx = {n: np.full(N_CELLS, -1, np.int32) for n in ref_names}
    kpts = {n: [] for n in ref_names}          # list of (x, y) in EDM px
    pair_idx: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def grid_kp(name, cell):
        """Index of this image's cell-centre keypoint, creating it on first use."""
        tbl = grid_idx[name]
        hit = tbl[cell]
        new = hit < 0
        if new.any():
            cells_new = np.unique(cell[new])
            start = len(kpts[name])
            gy, gx = np.divmod(cells_new, GRID_W)
            kpts[name].extend(np.stack([gx, gy], 1).astype(np.float64) * COARSE_STRIDE)
            tbl[cells_new] = np.arange(start, start + len(cells_new), dtype=np.int32)
        return tbl[cell]

    def refined_kp(name, pts):
        """Append refined points as fresh keypoints (one per anchor observation)."""
        start = len(kpts[name])
        kpts[name].extend(pts.astype(np.float64))
        return np.arange(start, start + len(pts), dtype=np.int32)

    n_anchor = n_obs = 0
    with h5py.File(pairs_h5, "r") as h5:
        for n0, n1 in pairs:
            g = h5[names_to_pair(n0, n1)]
            c0, c1 = g["cell0"][:], g["cell1"][:]
            k0, k1 = g["kpt0"][:], g["kpt1"][:]
            conf = g["conf"][:]
            r0 = EDMMatcher.is_refined(k0)      # True where n0 carries the subpixel offset
            d01 = ~r0                           # n0 on the grid -> landmark lives in n0

            # A landmark has exactly one anchor image. Partitioning by anchor therefore
            # partitions the 3D tracks without cutting any track across batches.
            keep = np.zeros(len(k0), dtype=bool)
            if n0 in anchor_set:
                keep |= d01
            if n1 in anchor_set:
                keep |= ~d01
            c0, c1 = c0[keep], c1[keep]
            k0, k1, conf = k0[keep], k1[keep], conf[keep]
            d01 = d01[keep]

            i0 = np.empty(len(k0), np.int32)
            i1 = np.empty(len(k0), np.int32)
            if d01.any():
                i0[d01] = grid_kp(n0, c0[d01])
                i1[d01] = refined_kp(n1, k1[d01])
            d10 = ~d01
            if d10.any():
                i1[d10] = grid_kp(n1, c1[d10])
                i0[d10] = refined_kp(n0, k0[d10])
            pair_idx[(n0, n1)] = (i0, i1, conf)
            n_obs += len(k0)

    tables = {}
    for name in ref_names:
        # EDM px -> that image's own camera px (the map may mix resolutions)
        arr = (np.asarray(kpts[name], np.float64) * scale_of[name]).astype(np.float32)
        n_anchor += int((grid_idx[name] >= 0).sum())
        tables[name] = {"keypoints": arr, "idx_of_cell": grid_idx[name]}

    n_kp = sum(len(t["keypoints"]) for t in tables.values())
    stats = {
        "keypoints_total": int(n_kp),
        "anchor_cells_total": int(n_anchor),
        "observations_total": int(n_obs),
        "keypoints_per_image": float(n_kp / max(len(ref_names), 1)),
        "anchor_cells_per_image": float(n_anchor / max(len(ref_names), 1)),
    }
    log(f"keypoint tables: {len(ref_names)} images, {n_kp} keypoints "
        f"({stats['keypoints_per_image']:.0f}/img), {n_anchor} anchor cells "
        f"({stats['anchor_cells_per_image']:.0f}/img), {n_obs} observations, 0 merged")
    return tables, pair_idx, stats


def write_feats_h5(path: Path, tables, ref_names, size_of: dict[str, tuple[int, int]]):
    if path.exists():
        path.unlink()
    with h5py.File(path, "w") as h5:
        for name in ref_names:
            g = h5.create_group(name)
            g.create_dataset("keypoints", data=tables[name]["keypoints"])
            g.create_dataset("image_size", data=np.array(size_of[name], np.float32))
    log(f"wrote {path}")


# ---------------------------------------------------------------- stage D
def write_matches_h5(path: Path, pairs, tables, pair_idx):
    """hloc match format. Every match survives: no endpoint was culled in stage C."""
    if path.exists():
        path.unlink()
    kept = 0
    with h5py.File(path, "w") as dst:
        for n0, n1 in pairs:
            i0, i1, conf = pair_idx[(n0, n1)]
            n_kp0 = len(tables[n0]["keypoints"])
            matches0 = np.full(n_kp0, -1, np.int32)
            scores0 = np.zeros(n_kp0, np.float32)
            order = np.argsort(conf)            # highest confidence wins a contested anchor
            matches0[i0[order]] = i1[order]
            scores0[i0[order]] = conf[order]
            kept += int((matches0 >= 0).sum())
            g = dst.create_group(names_to_pair(n0, n1))
            # mostly -1, so it compresses ~50x; hloc reads compressed datasets fine
            g.create_dataset("matches0", data=matches0, compression="gzip", compression_opts=1)
            g.create_dataset("matching_scores0", data=scores0, compression="gzip", compression_opts=1)
    log(f"wrote {path}: {kept} matches ({kept/max(len(pairs),1):.0f}/pair)")


# ---------------------------------------------------------------- stage E/F
def extract_anchor_xyz(tri, tables, anchor_names):
    """Extract cell->xyz LUTs for the landmarks owned by one anchor batch."""
    name_to_img = {im.name: im for im in tri.images.values()}
    result = {}

    for name in anchor_names:
        t = tables[name]
        n_kp = len(t["keypoints"])
        kp_xyz = np.full((n_kp, 3), np.nan, np.float32)
        img = name_to_img.get(name)
        if img is not None:
            for k, p2d in enumerate(img.points2D):
                if k < n_kp and p2d.has_point3D():
                    kp_xyz[k] = np.asarray(tri.points3D[p2d.point3D_id].xyz, np.float32)

        # O(1) runtime lookup: cell id -> xyz of that cell's ANCHOR keypoint (its centre),
        # which is exactly the point a direction-01 runtime match resolves to. NaN elsewhere.
        xyz_by_cell = np.full((N_CELLS, 3), np.nan, np.float32)
        anchored = t["idx_of_cell"] >= 0
        xyz_by_cell[anchored] = kp_xyz[t["idx_of_cell"][anchored]]
        result[name] = xyz_by_cell
    return result


def pack_bundle(
    xyz_by_name,
    ref_names,
    xb,
    image_root: Path,
    out: Path,
    meta_extra: dict,
):
    """Pack cell->xyz LUTs, reference JPEGs, and inherited tracking metadata."""
    refs, per_ref = {}, []
    for i, name in enumerate(ref_names, 1):
        xyz_by_cell = np.asarray(xyz_by_name[name], dtype=np.float32)
        if xyz_by_cell.shape != (N_CELLS, 3):
            raise ValueError(
                f"invalid xyz_by_cell shape for {name}: {xyz_by_cell.shape}"
            )

        gray = EDMMatcher.load_gray(image_root / name)
        ok, jpg = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise RuntimeError(f"jpeg encode failed for {name}")

        n3d = int(np.isfinite(xyz_by_cell[:, 0]).sum())
        per_ref.append(n3d)
        refs[name] = {"xyz_by_cell": xyz_by_cell, "image_jpg": np.frombuffer(jpg.tobytes(), np.uint8)}
        if i % 100 == 0 or i == len(ref_names):
            log(f"  packed {i}/{len(ref_names)} last_3d={n3d}")

    total = int(np.sum(per_ref))
    meta = {
        "feature": "edm",
        "matcher": "edm",
        "vpr": "MegaLoc",
        "bundle_vpr": "megaloc",
        "vpr_input": xb["meta"].get("vpr_input", 322),
        "edm_input_w": EDM_W, "edm_input_h": EDM_H,
        "edm_grid_w": GRID_W, "edm_grid_h": GRID_H,
        "keypoint_identity": KEYPOINT_IDENTITY,
        "xyz_source": "fixed-pose EDM re-triangulation on source COLMAP poses",
        "refs": len(ref_names),
        "total_3d_anchored_cells": total,
        "mean_3d_anchored_per_ref": float(np.mean(per_ref)) if per_ref else 0.0,
        "median_3d_anchored_per_ref": float(np.median(per_ref)) if per_ref else 0.0,
        "reference_images_embedded": True,
        **meta_extra,
    }
    # The active refs may be a SUBSET of the input bundle (e.g. a ref whose image could not
    # be recovered). ref_global/ref_centers/ref_yaws are row-aligned to the input ref_names
    # and covis stores integer indices INTO it, so both must be re-indexed to the new,
    # shorter ref_names -- otherwise every covisibility lookup silently points at the wrong
    # reference at flight time.
    old_names = list(xb["ref_names"])
    old_of = {n: i for i, n in enumerate(old_names)}
    keep = [old_of[n] for n in ref_names]
    new_of = {old_of[n]: i for i, n in enumerate(ref_names)}

    bundle = {
        "meta": meta,
        "ref_names": list(ref_names),
        "ref_global": np.asarray(xb["ref_global"], np.float32)[keep],   # MegaLoc unchanged
        "refs": refs,
    }
    for k in ("ref_centers", "ref_yaws", "ref_stability"):
        if k in xb and xb[k] is not None:
            bundle[k] = np.asarray(xb[k])[keep]
    if xb.get("covis"):
        bundle["covis"] = {n: [new_of[int(j)] for j in xb["covis"].get(n, [])
                               if int(j) in new_of] for n in ref_names}
    if len(ref_names) != len(old_names):
        meta["refs_dropped_from_input"] = len(old_names) - len(ref_names)
        log(f"  re-indexed {len(ref_names)}/{len(old_names)} refs "
            f"(ref_global, ref_centers, ref_yaws, covis)")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.pt")
    torch.save(bundle, tmp)
    tmp.rename(out)
    log("=" * 60)
    log(f"SAVED EDM reloc bundle -> {out}  ({out.stat().st_size/1e6:.0f} MB)")
    log(f"  refs={len(ref_names)} 3D-anchored cells={total} "
        f"mean/ref={meta['mean_3d_anchored_per_ref']:.0f} median/ref={meta['median_3d_anchored_per_ref']:.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument(
        "--in-bundle",
        required=True,
        help="seed bundle: MegaLoc + covis + tracking metadata are inherited from it",
    )
    ap.add_argument("--work-dir", default=str(DEF_WORK))
    ap.add_argument("--out", default=str(DEF_OUT))
    ap.add_argument(
        "--coverage-sidecar",
        help="optional exact EDM anchor observation sidecar (.npz) for offline coverage",
    )
    ap.add_argument("--pair-topk", type=int, default=20)
    ap.add_argument("--mconf-thr", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite-edm", action="store_true")
    ap.add_argument("--skip-geometric-verification", action="store_true")
    ap.add_argument(
        "--repair-zero-anchors-without-geometry",
        action="store_true",
        help=(
            "after verified batched triangulation, retry only empty anchor LUTs "
            "without independent two-view verification; fixed-pose triangulation "
            "still applies its reprojection and track checks"
        ),
    )
    ap.add_argument(
        "--anchor-batch-size",
        type=int,
        default=0,
        help=(
            "triangulate disjoint anchor-image batches sequentially; 0 keeps the "
            "legacy one-shot path"
        ),
    )
    args = ap.parse_args()
    model_path, image_root = Path(args.model), Path(args.image_root)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    log(f"load seed bundle (MegaLoc + covis) {args.in_bundle}")
    xb = torch.load(args.in_bundle, map_location="cpu", weights_only=False)
    ref_names = list(xb["ref_names"])
    if not xb.get("covis"):
        raise SystemExit("input bundle lacks covis metadata")

    rec = pycolmap.Reconstruction(str(model_path))
    # The map may mix source resolutions (720p / 1080p / 2.7K), so scale is PER IMAGE.
    size_of, scale_of = {}, {}
    for im in rec.images.values():
        c = rec.cameras[im.camera_id]
        size_of[im.name] = (c.width, c.height)
        scale_of[im.name] = c.width / EDM_W
    res = sorted({f"{w}x{h}" for w, h in size_of.values()})
    log(f"model: {rec.num_reg_images()} images, {rec.num_points3D()} pts, "
        f"{len(rec.cameras)} camera(s) {res}, EDM {EDM_W}x{EDM_H}")

    # Only refs whose image is actually in the model can be built; the rest are dropped and
    # the bundle is re-indexed accordingly in pack_bundle().
    active = [n for n in ref_names if n in size_of]
    if args.limit:
        active = active[:args.limit]
    if len(active) != len(ref_names):
        log(f"  {len(ref_names) - len(active)} bundle refs are not in the model -> dropped")

    # A
    pairs = pair_list_from_covis(ref_names, xb["covis"], args.pair_topk, active)
    pairs_txt = work / f"pairs-edm-covis-top{args.pair_topk}.txt"
    pairs_txt.write_text("".join(f"{a} {b}\n" for a, b in pairs))
    log(f"stage A: {len(pairs)} covis pairs -> {pairs_txt}")

    # B
    pairs_h5 = work / edm_pair_cache_name(args.pair_topk)
    run_edm_pairs(pairs_h5, pairs, image_root, args.mconf_thr, args.batch, args.overwrite_edm)

    opts = pycolmap.IncrementalPipelineOptions()
    opts.extract_colors = False
    coverage_chunks: list[CoverageRecords] = []

    if args.anchor_batch_size <= 0:
        # Legacy one-shot path, retained for smaller maps such as river_site.
        tables, pair_idx, kstats = build_keypoint_tables(
            pairs_h5, active, pairs, scale_of
        )
        feats_h5 = work / "feats-edm.h5"
        write_feats_h5(feats_h5, tables, active, size_of)
        matches_h5 = work / "matches-edm.h5"
        write_matches_h5(matches_h5, pairs, tables, pair_idx)

        tri_dir = work / "edm_model"
        active_model_dir = work / "active_reference_model"
        if active_model_dir.exists():
            shutil.rmtree(active_model_dir)
        active_model_dir.mkdir(parents=True)
        active_rec = pycolmap.Reconstruction(str(model_path))
        removed_from_triangulation = restrict_reconstruction_to_active_refs(
            active_rec, active
        )
        active_rec.write(str(active_model_dir))
        log(
            f"stage E input: {active_rec.num_reg_images()} active refs; "
            f"deregistered {len(removed_from_triangulation)} pose-only images"
        )
        log(f"stage E: fixed-pose triangulation -> {tri_dir}")
        tri = triangulation.main(
            tri_dir,
            active_model_dir,
            image_root,
            pairs_txt,
            feats_h5,
            matches_h5,
            skip_geometric_verification=args.skip_geometric_verification,
            mapper_options=opts,
        )
        log(
            f"  triangulated: images={tri.num_reg_images()} "
            f"points3D={tri.num_points3D()}"
        )
        xyz_by_name = extract_anchor_xyz(tri, tables, active)
        if args.coverage_sidecar:
            coverage_chunks.append(
                extract_coverage_records(tri, tables, active, active)
            )
        triangulation_points = int(tri.num_points3D())
        batch_meta = {}
    else:
        # EDM tracks are anchor-centric: each landmark belongs to exactly one image cell.
        # Splitting by anchor image therefore preserves every complete track while bounding
        # COLMAP's correspondence graph to a small sequential batch.
        batches = list(iter_anchor_batches(active, args.anchor_batch_size))
        batch_root = work / anchor_batch_cache_name(
            args.pair_topk,
            args.anchor_batch_size,
            args.skip_geometric_verification,
        )
        batch_root.mkdir(parents=True, exist_ok=True)
        pair_artifact = {
            "size": pairs_h5.stat().st_size,
            "mtime_ns": pairs_h5.stat().st_mtime_ns,
        }
        xyz_by_name = {}
        triangulation_points = 0
        total_keypoints = total_anchors = total_observations = 0

        for batch_index, anchor_batch in enumerate(batches):
            batch_dir = batch_root / f"batch_{batch_index:04d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = batch_dir / "anchor_xyz.pt"
            expected = {
                "anchor_names": list(anchor_batch),
                "pair_topk": args.pair_topk,
                "skip_geometric_verification": args.skip_geometric_verification,
                "source_model": str(model_path.resolve()),
                "pair_artifact": pair_artifact,
            }
            batch_coverage = None

            if checkpoint.exists():
                payload = torch.load(
                    checkpoint, map_location="cpu", weights_only=False
                )
                actual = {key: payload.get(key) for key in expected}
                if actual != expected:
                    raise RuntimeError(
                        f"stale anchor-batch checkpoint {checkpoint}: "
                        f"expected {expected}, got {actual}"
                    )
                batch_xyz = payload["xyz_by_name"]
                batch_stats = payload["keypoint_stats"]
                batch_points = int(payload["triangulation_points3D"])
                if args.coverage_sidecar:
                    raw_coverage = payload.get("coverage_records")
                    if raw_coverage is None:
                        raise RuntimeError(
                            f"anchor-batch checkpoint {checkpoint} predates exact "
                            "coverage tracks; rebuild it in a fresh work directory"
                        )
                    batch_coverage = CoverageRecords.from_payload(raw_coverage)
                log(
                    f"reuse anchor batch {batch_index + 1}/{len(batches)}: "
                    f"{len(anchor_batch)} anchors, {batch_points} points"
                )
            else:
                batch_pairs = pairs_for_anchor_batch(pairs, anchor_batch)
                participant_set = set(anchor_batch)
                for name0, name1 in batch_pairs:
                    participant_set.update((name0, name1))
                participants = [name for name in active if name in participant_set]
                log(
                    f"anchor batch {batch_index + 1}/{len(batches)}: "
                    f"{len(anchor_batch)} anchors, {len(participants)} participants, "
                    f"{len(batch_pairs)} pairs"
                )

                tables, pair_idx, batch_stats = build_keypoint_tables(
                    pairs_h5,
                    participants,
                    batch_pairs,
                    scale_of,
                    anchor_names=set(anchor_batch),
                )
                batch_pairs_txt = batch_dir / "pairs.txt"
                batch_pairs_txt.write_text(
                    "".join(f"{a} {b}\n" for a, b in batch_pairs),
                    encoding="utf-8",
                )
                feats_h5 = batch_dir / "feats-edm.h5"
                matches_h5 = batch_dir / "matches-edm.h5"
                write_feats_h5(feats_h5, tables, participants, size_of)
                write_matches_h5(matches_h5, batch_pairs, tables, pair_idx)

                active_model_dir = batch_dir / "active_reference_model"
                if active_model_dir.exists():
                    shutil.rmtree(active_model_dir)
                active_model_dir.mkdir(parents=True)
                active_rec = pycolmap.Reconstruction(str(model_path))
                removed = restrict_reconstruction_to_active_refs(
                    active_rec, participants
                )
                active_rec.write(str(active_model_dir))
                log(
                    f"  batch triangulation input: {active_rec.num_reg_images()} refs; "
                    f"deregistered {len(removed)} nonparticipants"
                )

                tri_dir = batch_dir / "edm_model"
                tri = triangulation.main(
                    tri_dir,
                    active_model_dir,
                    image_root,
                    batch_pairs_txt,
                    feats_h5,
                    matches_h5,
                    skip_geometric_verification=args.skip_geometric_verification,
                    mapper_options=opts,
                )
                batch_xyz = extract_anchor_xyz(tri, tables, anchor_batch)
                if args.coverage_sidecar:
                    batch_coverage = extract_coverage_records(
                        tri, tables, anchor_batch, active
                    )
                batch_points = int(tri.num_points3D())
                payload = {
                    **expected,
                    "xyz_by_name": batch_xyz,
                    "keypoint_stats": batch_stats,
                    "triangulation_points3D": batch_points,
                }
                if batch_coverage is not None:
                    payload["coverage_records"] = batch_coverage.as_payload()
                tmp_checkpoint = checkpoint.with_suffix(".tmp.pt")
                torch.save(payload, tmp_checkpoint)
                tmp_checkpoint.rename(checkpoint)
                log(
                    f"  completed batch {batch_index + 1}/{len(batches)}: "
                    f"images={tri.num_reg_images()} points3D={batch_points}"
                )
                del active_rec, pair_idx, tables, tri

            overlap = set(xyz_by_name) & set(batch_xyz)
            if overlap:
                raise RuntimeError(f"duplicate anchor ownership: {sorted(overlap)}")
            xyz_by_name.update(batch_xyz)
            if batch_coverage is not None:
                coverage_chunks.append(batch_coverage)
            triangulation_points += batch_points
            total_keypoints += int(batch_stats["keypoints_total"])
            total_anchors += int(batch_stats["anchor_cells_total"])
            total_observations += int(batch_stats["observations_total"])
            del batch_xyz
            gc.collect()

        if set(xyz_by_name) != set(active):
            missing = sorted(set(active) - set(xyz_by_name))
            extra = sorted(set(xyz_by_name) - set(active))
            raise RuntimeError(
                f"batched anchor coverage mismatch: missing={missing}, extra={extra}"
            )

        repaired_zero_anchors: list[str] = []
        repair_points = 0
        empty_anchors = zero_anchor_names(xyz_by_name)
        if empty_anchors and args.repair_zero_anchors_without_geometry:
            repair_dir = batch_root / "zero_anchor_repair_skip_geometry"
            repair_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = repair_dir / "anchor_xyz.pt"
            expected = {
                "anchor_names": empty_anchors,
                "pair_topk": args.pair_topk,
                "skip_geometric_verification": True,
                "source_model": str(model_path.resolve()),
                "pair_artifact": pair_artifact,
            }
            if checkpoint.exists():
                payload = torch.load(
                    checkpoint, map_location="cpu", weights_only=False
                )
                actual = {key: payload.get(key) for key in expected}
                if actual != expected:
                    raise RuntimeError(
                        f"stale zero-anchor repair checkpoint {checkpoint}: "
                        f"expected {expected}, got {actual}"
                    )
                repairs = payload["xyz_by_name"]
                repair_points = int(payload["triangulation_points3D"])
                repair_coverage = None
                if args.coverage_sidecar:
                    raw_coverage = payload.get("coverage_records")
                    if raw_coverage is None:
                        raise RuntimeError(
                            f"zero-anchor checkpoint {checkpoint} predates exact "
                            "coverage tracks; rebuild it in a fresh work directory"
                        )
                    repair_coverage = CoverageRecords.from_payload(raw_coverage)
                log(
                    f"reuse zero-anchor repair: {len(empty_anchors)} anchors, "
                    f"{repair_points} points"
                )
            else:
                repair_pairs = pairs_for_anchor_batch(pairs, empty_anchors)
                participant_set = set(empty_anchors)
                for name0, name1 in repair_pairs:
                    participant_set.update((name0, name1))
                participants = [
                    name for name in active if name in participant_set
                ]
                log(
                    f"zero-anchor repair: {len(empty_anchors)} anchors, "
                    f"{len(participants)} participants, {len(repair_pairs)} pairs; "
                    "skip independent two-view geometry"
                )
                tables, pair_idx, repair_stats = build_keypoint_tables(
                    pairs_h5,
                    participants,
                    repair_pairs,
                    scale_of,
                    anchor_names=set(empty_anchors),
                )
                repair_pairs_txt = repair_dir / "pairs.txt"
                repair_pairs_txt.write_text(
                    "".join(f"{a} {b}\n" for a, b in repair_pairs),
                    encoding="utf-8",
                )
                feats_h5 = repair_dir / "feats-edm.h5"
                matches_h5 = repair_dir / "matches-edm.h5"
                write_feats_h5(feats_h5, tables, participants, size_of)
                write_matches_h5(matches_h5, repair_pairs, tables, pair_idx)

                active_model_dir = repair_dir / "active_reference_model"
                if active_model_dir.exists():
                    shutil.rmtree(active_model_dir)
                active_model_dir.mkdir(parents=True)
                active_rec = pycolmap.Reconstruction(str(model_path))
                restrict_reconstruction_to_active_refs(active_rec, participants)
                active_rec.write(str(active_model_dir))
                tri = triangulation.main(
                    repair_dir / "edm_model",
                    active_model_dir,
                    image_root,
                    repair_pairs_txt,
                    feats_h5,
                    matches_h5,
                    skip_geometric_verification=True,
                    mapper_options=opts,
                )
                repairs = extract_anchor_xyz(tri, tables, empty_anchors)
                repair_coverage = None
                if args.coverage_sidecar:
                    repair_coverage = extract_coverage_records(
                        tri, tables, empty_anchors, active
                    )
                repair_points = int(tri.num_points3D())
                payload = {
                    **expected,
                    "xyz_by_name": repairs,
                    "keypoint_stats": repair_stats,
                    "triangulation_points3D": repair_points,
                }
                if repair_coverage is not None:
                    payload["coverage_records"] = repair_coverage.as_payload()
                tmp_checkpoint = checkpoint.with_suffix(".tmp.pt")
                torch.save(payload, tmp_checkpoint)
                tmp_checkpoint.rename(checkpoint)
                del active_rec, pair_idx, tables, tri

            repaired_zero_anchors = apply_zero_anchor_repairs(
                xyz_by_name, repairs
            )
            if repair_coverage is not None:
                coverage_chunks.append(repair_coverage)
            log(f"repaired zero anchors: {repaired_zero_anchors}")
            del repairs
            gc.collect()

        kstats = {
            "keypoints_total": total_keypoints,
            "anchor_cells_total": total_anchors,
            "observations_total": total_observations,
            "keypoints_per_image": total_keypoints / max(len(active), 1),
            "anchor_cells_per_image": total_anchors / max(len(active), 1),
        }
        batch_meta = {
            "triangulation_anchor_batch_size": args.anchor_batch_size,
            "triangulation_batches": len(batches),
            "zero_anchor_repairs_without_geometry": repaired_zero_anchors,
            "zero_anchor_repair_points3D": repair_points,
        }

    # F
    out_path = Path(args.out)
    pack_bundle(
        xyz_by_name,
        active,
        xb,
        image_root,
        out_path,
        {
            "triangulation_pair_topk": args.pair_topk,
            "triangulation_pairs": len(pairs),
            "triangulation_points3D": triangulation_points,
            "edm_mconf_thr": args.mconf_thr,
            "source_model": str(model_path),
            "image_root": str(image_root),
            "keypoint_stats": kstats,
            **batch_meta,
        },
    )
    if args.coverage_sidecar:
        coverage = merge_records(
            coverage_chunks,
            ref_count=len(active),
            cell_count=N_CELLS,
        )
        expected_anchors = sum(anchored_cell_count(xyz) for xyz in xyz_by_name.values())
        if len(coverage.anchor_ref_idx) != expected_anchors:
            raise RuntimeError(
                "coverage sidecar does not cover every finite production EDM anchor: "
                f"{len(coverage.anchor_ref_idx)} != {expected_anchors}"
            )
        sidecar = write_coverage_sidecar(
            args.coverage_sidecar,
            coverage,
            bundle_path=out_path,
            ref_names=active,
            cell_count=N_CELLS,
        )
        log(f"SAVED exact coverage observations -> {sidecar}")


if __name__ == "__main__":
    main()
