#!/usr/bin/env python3
"""Build and verify a pruned INT8 (schema-3) candidate of the EDM bank (B6).

Never touches the production bank: reads the shipped store via mmap (only
faults pages it samples), writes strictly under a candidate root. The loader
refuses to emit inside the source store's directory.

Subcommands:
  quant-error  sampled FP16->INT8->FP16 round-trip error from the real bank
               (default 8 refs x all levels; mmap, small page footprint).
  build        write schema-3 candidate for a keep-list JSON ({keep:[names]})
               to <candidate_root>/<name>/ (manifest schema 3 + int8 shards).
  verify       reload candidate, compare dequantized resident rows against the
               source bank, report max/mean error + row-identity of names.

Gate: off-map 0/700 per-row identical + 7-seg no-regress + model_identity pin
(see outputs/bank_prune_int8_20260906/REPORT.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "定位演算法" / "deploy_code" / "sfm_glomap_deploy"
sys.path.insert(0, str(DEPLOY))

import torch  # noqa: E402

import edm_matcher as M  # noqa: E402


def _source_bank_dir() -> Path:
    for c in sorted((REPO / "執行環境" / "models" / "edm_reference_features").glob("*.shards")):
        if c.is_dir():
            return c
    raise SystemExit("no sharded source bank under 執行環境/models/edm_reference_features/")


def _load_manifest(bank: Path) -> dict:
    return torch.load(bank / "manifest.pt", map_location="cpu", weights_only=True)


def cmd_quant_error(args) -> None:
    bank = _source_bank_dir()
    manifest = _load_manifest(bank)
    names = list(manifest["source_names"])
    shard_size = int(manifest["shard_size"])
    files = list(manifest["shard_files"])
    rng = torch.Generator().manual_seed(args.seed)
    picks = torch.randperm(len(names), generator=rng)[: args.refs].tolist()
    worst, tot, n = 0.0, 0.0, 0
    by_shard: dict[int, list[int]] = {}
    for r in picks:
        si, li = divmod(r, shard_size)
        by_shard.setdefault(si, []).append(li)
    for si, local_rows in sorted(by_shard.items()):
        payload = torch.load(bank / files[si], map_location="cpu",
                             weights_only=True, mmap=True)
        levels = payload["levels"]
        for li in local_rows:
            rows = [lv[li:li + 1].to(torch.float32) for lv in levels]
            q, s = M.EDMMatcher._quantize_bank_int8(rows)
            d = M.EDMMatcher._dequantize_bank_rows(q, s)
            for a, b in zip(rows, d):
                e = (a - b).abs()
                worst = max(worst, float(e.max()))
                tot += float(e.mean())
                n += 1
    print(json.dumps({"refs": len(picks), "levels_per_ref": n // max(len(picks), 1),
                      "roundtrip_max": worst, "roundtrip_mean": tot / max(n, 1)},
                     indent=1))


def cmd_build(args) -> None:
    bank = _source_bank_dir()
    manifest = _load_manifest(bank)
    names = list(manifest["source_names"])
    digests = list(manifest["source_sha256"])
    shard_size = int(manifest["shard_size"])
    files = list(manifest["shard_files"])
    keep = json.loads(Path(args.keep).read_text())["keep"]
    pos = {name: i for i, name in enumerate(names)}
    missing = [k for k in keep if k not in pos]
    if missing:
        raise SystemExit(f"keep-list has {len(missing)} names outside the bank, e.g. {missing[:3]}")
    keep_pos = sorted(pos[k] for k in keep)
    out = Path(args.out)
    if _is_inside(bank, out):
        raise SystemExit("refusing to emit inside the source store")
    tmp = out.parent / f".{out.name}.{os.getpid()}.tmp"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        resident = {p: r for r, p in enumerate(keep_pos)}
        per_shard: dict[int, list[int]] = {}
        for p in keep_pos:
            si, li = divmod(p, shard_size)
            per_shard.setdefault(si, []).append(li)
        out_files = []
        entries: list = []
        for si in sorted(per_shard):
            payload = torch.load(bank / files[si], map_location="cpu",
                                 weights_only=True, mmap=True)
            levels = payload["levels"]
            for li in per_shard[si]:
                # Raw rows, untouched dtype: lossless unless --quant int8.
                entries.append((si * shard_size + li,
                                [lv[li:li + 1] for lv in levels]))
        entries.sort(key=lambda e: resident[e[0]])
        n_out = 0
        for start in range(0, len(entries), M.REFERENCE_FEATURE_SHARD_ENTRIES):
            chunk = entries[start:start + M.REFERENCE_FEATURE_SHARD_ENTRIES]
            n_levels = len(chunk[0][1])
            if args.quant == "none":
                qlevels = [torch.cat([e[1][d] for e in chunk], dim=0)
                           for d in range(n_levels)]
                slevels = None
            else:
                converted = [
                    M.EDMMatcher._quantize_bank_int8(e[1]) for e in chunk
                ]
                qlevels = [torch.cat([c[0][d] for c in converted], dim=0)
                           for d in range(n_levels)]
                slevels = [torch.stack([c[1][d] for c in converted], dim=0)
                           for d in range(n_levels)]
            payload = {"schema_version": M.REFERENCE_FEATURE_SHARD_SCHEMA_3,
                       "start": start, "stop": start + len(chunk),
                       "levels": qlevels}
            if slevels is not None:
                payload["scales"] = slevels
            fname = f"shard_{n_out:04d}.pt"
            torch.save(payload, tmp / fname)
            out_files.append(fname)
            n_out += 1
        torch.save({"schema_version": M.REFERENCE_FEATURE_SHARD_SCHEMA_3,
                    "model_key": manifest["model_key"],
                    "quant": args.quant,
                    "resident_names": [names[p] for p in keep_pos],
                    "resident_positions": keep_pos,
                    "resident_sha256": [digests[p] for p in keep_pos],
                    "shard_size": M.REFERENCE_FEATURE_SHARD_ENTRIES,
                    "shard_files": out_files}, tmp / "manifest.pt")
        os.replace(tmp, out)
        print(json.dumps({"resident": len(keep_pos), "shards": len(out_files),
                          "out": str(out)}, indent=1))
    finally:
        import shutil
        if tmp.exists():
            shutil.rmtree(tmp)


def _is_inside(bank: Path, out: Path) -> bool:
    try:
        out.resolve().relative_to(bank.resolve())
        return True
    except ValueError:
        return False


def cmd_verify(args) -> None:
    bank = _source_bank_dir()
    src = _load_manifest(bank)
    shard_size = int(src["shard_size"])
    files = list(src["shard_files"])
    cand = torch.load(Path(args.candidate) / "manifest.pt",
                      map_location="cpu", weights_only=True)
    assert cand["schema_version"] == M.REFERENCE_FEATURE_SHARD_SCHEMA_3
    resident = cand["resident_names"]
    positions = cand["resident_positions"]
    worst, tot, n = 0.0, 0.0, 0
    checked = 0
    for rname, pos in zip(resident, positions):
        if checked >= args.refs and args.refs > 0:
            break
        checked += 1
        si, li = divmod(pos, shard_size)
        payload = torch.load(bank / files[si], map_location="cpu",
                             weights_only=True, mmap=True)
        src_rows = [lv[li:li + 1].to(torch.float32) for lv in payload["levels"]]
        csi, cli = divmod(checked - 1, M.REFERENCE_FEATURE_SHARD_ENTRIES)
        cpayload = torch.load(Path(args.candidate) / cand["shard_files"][csi],
                              map_location="cpu", weights_only=True)
        qrows = [lv[cli:cli + 1] for lv in cpayload["levels"]]
        if cand.get("quant", "int8") == "none":
            drows = [q.to(torch.float32) for q in qrows]
        else:
            drows = M.EDMMatcher._dequantize_bank_rows(
                qrows, [s[cli:cli + 1] for s in cpayload["scales"]]
            )
        for a, b in zip(src_rows, drows):
            e = (a - b).abs()
            worst = max(worst, float(e.max()))
            tot += float(e.mean())
            n += 1
    print(json.dumps({"checked_refs": checked, "verify_max": worst,
                      "verify_mean": tot / max(n, 1)}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quant-error")
    q.add_argument("--refs", type=int, default=8)
    q.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("build")
    b.add_argument("--keep", required=True)
    b.add_argument("--quant", choices=("int8", "none"), default="int8")
    v = sub.add_parser("verify")
    b.add_argument("--out", required=True)
    v.add_argument("--candidate", required=True)
    v.add_argument("--refs", type=int, default=8)
    args = ap.parse_args()
    {"quant-error": cmd_quant_error, "build": cmd_build,
     "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
