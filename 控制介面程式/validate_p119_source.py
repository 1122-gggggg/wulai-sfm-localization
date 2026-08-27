#!/usr/bin/env python3
"""Verify an explicitly provided P119 source and its accepted incomplete tail."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


DEFAULT_VIDEO = Path(__file__).resolve().parents[1] / "模擬器/測試影片/P1190119.MP4"
EXPECTED_SHA256 = "600bbf70227311cab079d77fcb896f97e6d3e55f6bc40b5bef01d74b65f7826c"
EXPECTED_DECLARED_FRAMES = 2935
EXPECTED_DECODED_FRAMES = 2934


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _frame_counts(path: Path) -> tuple[int | None, int | None]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        return None, None

    def parse(key: str) -> int | None:
        try:
            return int(streams[0].get(key))
        except (TypeError, ValueError):
            return None

    return parse("nb_frames"), parse("nb_read_frames")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--accept-known-incomplete", action="store_true")
    args = parser.parse_args()
    path = args.video.expanduser().resolve()
    result: dict[str, object] = {
        "video": str(path),
        "expected_sha256": EXPECTED_SHA256,
        "declared_frames_expected": EXPECTED_DECLARED_FRAMES,
        "decoded_frames_expected": EXPECTED_DECODED_FRAMES,
        "waiver_accepted": bool(args.accept_known_incomplete),
        "ok": False,
    }
    if not path.is_file():
        result["status"] = "MISSING"
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    actual_sha = _sha256(path)
    declared, decoded = _frame_counts(path)
    result.update(
        actual_sha256=actual_sha,
        declared_frames=declared,
        decoded_frames=decoded,
    )
    exact_known_source = (
        actual_sha == EXPECTED_SHA256
        and declared == EXPECTED_DECLARED_FRAMES
        and decoded == EXPECTED_DECODED_FRAMES
    )
    if not exact_known_source:
        result["status"] = "UNEXPECTED_SOURCE_OR_DECODE_RESULT"
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    result["status"] = "KNOWN_INCOMPLETE"
    result["ok"] = bool(args.accept_known_incomplete)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if args.accept_known_incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
