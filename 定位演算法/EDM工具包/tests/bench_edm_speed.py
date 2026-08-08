#!/usr/bin/env python3
"""How fast can EDM's forward actually go on this hardware?

The stream benchmark put EDM's TRACK-mode matching at 21.6ms for 2 references against the
XFeat route's 5.9ms, so EDM only becomes viable for tracking if the forward can be cut.
This measures the levers that do not change the map: fp16 autocast, channels_last, and
input resolution -- plus the acquisition cost (topk=10), which is where the XFeat route is
known to spend 440ms.

Match quality is reported alongside speed, because a fast forward that stops matching is
not a win.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy"))
from edm_matcher import EDMMatcher  # noqa: E402

def bench(matcher, q, refs, n_ref, amp: bool, iters: int = 30):
    imgs = refs[:n_ref]
    ctx = torch.autocast("cuda", dtype=torch.float16) if amp else torch.autocast("cuda", enabled=False)
    with ctx:
        for _ in range(5):
            r = matcher.match_many_to_one(imgs, q)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            r = matcher.match_many_to_one(imgs, q)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1000
    n_match = int(np.mean([len(x["mkpts0"]) for x in r]))
    return ms, n_match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--sequence", default="P1190119")
    args = parser.parse_args()
    sequence_dir = Path(args.image_root).expanduser().resolve() / args.sequence
    names = sorted(p.name for p in sequence_dir.glob("*.jpg"))[:12]
    if len(names) < 12:
        raise SystemExit(f"need at least 12 JPG images in {sequence_dir}, found {len(names)}")
    grays = [EDMMatcher.load_gray(sequence_dir / n) for n in names]
    q, refs = grays[0], grays[1:]

    m = EDMMatcher(mconf_thr=0.2)
    print(f"GPU {torch.cuda.get_device_name(0)}   EDM 1024x576 topk={m.topk}\n")
    print(f"{'config':<28} {'1 ref':>10} {'2 refs':>10} {'10 refs':>11}  {'matches/ref':>12}")

    for label, amp in (("fp32 (as benchmarked)", False), ("fp16 autocast", True)):
        row, nm = [], 0
        for k in (1, 2, 10):
            ms, nm = bench(m, q, refs, k, amp)
            row.append(ms)
        print(f"{label:<28} {row[0]:>9.1f}m {row[1]:>9.1f}m {row[2]:>10.1f}m  {nm:>12d}")

    # channels_last on top of fp16
    m.model = m.model.to(memory_format=torch.channels_last)
    row, nm = [], 0
    for k in (1, 2, 10):
        ms, nm = bench(m, q, refs, k, True)
        row.append(ms)
    print(f"{'fp16 + channels_last':<28} {row[0]:>9.1f}m {row[1]:>9.1f}m {row[2]:>10.1f}m  {nm:>12d}")

    print("\nreference points:")
    print("  XFeat route, same flight: TRACK match 5.9ms (mutual-NN), total 16.2ms")
    print("  XFeat route, acquisition: 439.8ms (LighterGlue x 30 refs)  <- frame 0 of the baseline run")


if __name__ == "__main__":
    main()
