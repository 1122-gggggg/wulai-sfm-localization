#!/usr/bin/env python3
"""Create a hash-pinned mission selection from independent component manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "控制介面程式"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))

from mission_manifest import ManifestError  # noqa: E402
from mission_resolver import resolve_mission  # noqa: E402


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _reference(value: str, *, base: Path, label: str) -> dict[str, str]:
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(ROOT):
        raise ManifestError(f"{label} must stay inside the workspace: {path}")
    if not path.is_file() or path.is_symlink():
        raise ManifestError(f"{label} must be a regular non-symlink file: {path}")
    return {
        "path": os.path.relpath(path, base),
        "sha256": _digest(path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new selection JSON; not overwritten")
    parser.add_argument("--selection-id", required=True)
    parser.add_argument("--vehicle", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--map", dest="map_manifest", required=True)
    parser.add_argument("--localizer", required=True)
    parser.add_argument("--route")
    parser.add_argument("--calibration", action="append", default=[])
    parser.add_argument("--approval")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if not output.is_relative_to(ROOT):
        print("output must stay inside the workspace", file=sys.stderr)
        return 2
    if output.exists():
        print(f"refusing to overwrite existing selection: {output}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        document = {
            "schema": "sfm-mission-selection/v1",
            "selection_id": args.selection_id,
            "vehicle": _reference(args.vehicle, base=output.parent, label="vehicle"),
            "site": _reference(args.site, base=output.parent, label="site"),
            "map": _reference(args.map_manifest, base=output.parent, label="map"),
            "localizer": _reference(
                args.localizer,
                base=output.parent,
                label="localizer",
            ),
            "route": (
                None
                if args.route is None
                else _reference(args.route, base=output.parent, label="route")
            ),
            "calibrations": [
                _reference(value, base=output.parent, label="calibration")
                for value in args.calibration
            ],
        }
        if args.approval is not None:
            document["approval"] = _reference(
                args.approval,
                base=output.parent,
                label="approval",
            )
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        mission = resolve_mission(temporary, workspace_root=ROOT)
        if not mission.readiness.localization_ready:
            raise ManifestError(
                "selection is not localization-ready: "
                + "; ".join(mission.readiness.localization_errors)
            )
        os.replace(temporary, output)
        print(
            json.dumps(
                {
                    "selection": str(output),
                    "localization_ready": True,
                    "flight_ready": mission.readiness.flight_ready,
                    "flight_errors": list(mission.readiness.flight_errors),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (ManifestError, OSError, ValueError) as exc:
        print(f"cannot create mission selection: {exc}", file=sys.stderr)
        return 2
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
