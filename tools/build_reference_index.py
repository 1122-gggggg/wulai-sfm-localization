#!/usr/bin/env python3
"""Build a deterministic offline IVF reference index from checked inputs.

The command accepts only a JSON list of stable reference names and a NumPy
``float32`` descriptor matrix.  It performs no asset acquisition or network
access; malformed inputs and an existing output directory fail closed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = ROOT / "定位演算法" / "deploy_code" / "sfm_glomap_deploy"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from reference_index import build_reference_index  # noqa: E402


NORMALIZATION_TOLERANCE = 1e-3


def _load_names(path: Path) -> list[str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read names JSON {path}: {exc}") from exc
    if not isinstance(value, list):
        raise ValueError(f"names JSON must contain a list: {path}")
    if any(not isinstance(name, str) or not name for name in value):
        raise ValueError("names JSON must contain only non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError("names JSON must contain unique stable identifiers")
    return value


def _load_descriptors(path: Path) -> np.ndarray:
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"cannot read descriptor .npy {path}: {exc}") from exc
    if not isinstance(values, np.ndarray):
        close = getattr(values, "close", None)
        if callable(close):
            close()
        raise ValueError(f"descriptor input must be a single .npy array: {path}")
    if values.dtype != np.dtype(np.float32):
        raise ValueError("descriptor .npy must use float32 dtype")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("descriptor .npy must be a non-empty 2-D array")
    if not np.isfinite(values).all():
        raise ValueError("descriptor .npy must contain only finite values")
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError("descriptor .npy rows must have finite, non-zero norms")
    if not np.all(np.abs(norms - 1.0) <= NORMALIZATION_TOLERANCE):
        raise ValueError("descriptor .npy rows must be L2-normalized")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--descriptors",
        required=True,
        type=Path,
        help="normalized float32 descriptor matrix in .npy format",
    )
    parser.add_argument(
        "--names",
        required=True,
        type=Path,
        help="JSON file containing one unique stable name per descriptor row",
    )
    parser.add_argument("--output", required=True, type=Path, help="new index directory")
    parser.add_argument("--model-identity", required=True, help="immutable localizer model identity")
    parser.add_argument("--nlist", type=int, help="number of IVF lists; defaults to the library policy")
    parser.add_argument("--seed", type=int, default=0, help="non-negative deterministic k-means seed")
    parser.add_argument("--kmeans-iterations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-query-probes", type=int)
    parser.add_argument("--max-query-candidates", type=int)
    return parser


def _build(args: argparse.Namespace) -> int:
    descriptors = _load_descriptors(args.descriptors)
    names = _load_names(args.names)
    if len(names) != descriptors.shape[0]:
        raise ValueError(
            "names count does not match descriptor rows: "
            f"{len(names)} != {descriptors.shape[0]}"
        )
    output = build_reference_index(
        args.output,
        descriptors,
        names,
        model_identity=args.model_identity,
        nlist=args.nlist,
        seed=args.seed,
        kmeans_iterations=args.kmeans_iterations,
        batch_size=args.batch_size,
        max_query_probes=args.max_query_probes,
        max_query_candidates=args.max_query_candidates,
    )
    summary: dict[str, Any] = {
        "count": int(descriptors.shape[0]),
        "dimension": int(descriptors.shape[1]),
        "model_identity": args.model_identity,
        "output": str(output.resolve()),
        "seed": args.seed,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _build(args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"reference index build failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
