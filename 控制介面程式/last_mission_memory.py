#!/usr/bin/env python3
"""Remember which mission selection (map) the operator opened last.

The record only names a selection file. Every launch still re-resolves and
re-validates it through the mission resolver, so a stale or edited selection is
rejected exactly like an explicitly passed one.

  last_mission_memory.py show   print the remembered selection path if it still
                                resolves; print nothing otherwise
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from mission_manifest import ManifestError
from mission_resolver import resolve_mission
from workspace_layout import workspace_from_file

WORKSPACE = workspace_from_file(__file__)
MEMORY_FILE = WORKSPACE.runtime / "last_mission_selection.json"
SCHEMA = "sfm-last-mission-selection/v1"


def remember_mission_selection(selection: str | os.PathLike[str], memory_file: Path = MEMORY_FILE) -> None:
    """Atomically record ``selection``; never raises on an unwritable location."""
    text = str(selection).strip()
    if not text:
        return
    payload = {"schema": SCHEMA, "mission_selection": str(Path(text).expanduser().resolve())}
    try:
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=memory_file.parent, prefix=".last_mission_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            os.replace(temp, memory_file)
        except BaseException:
            os.unlink(temp)
            raise
    except OSError as exc:
        print(f"[last-map] cannot save last map: {exc}", file=sys.stderr)


def load_remembered_selection(memory_file: Path = MEMORY_FILE) -> Path | None:
    """Return the remembered selection only if it still resolves, else None."""
    try:
        data = json.loads(memory_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        print(f"[last-map] ignoring unreadable memory file: {memory_file}", file=sys.stderr)
        return None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        print(f"[last-map] ignoring unrecognized memory file: {memory_file}", file=sys.stderr)
        return None
    selection = Path(str(data.get("mission_selection", "")))
    if not selection.is_file():
        print(f"[last-map] remembered selection is gone: {selection}", file=sys.stderr)
        return None
    try:
        resolve_mission(selection, workspace_root=WORKSPACE.root)
    except (ManifestError, OSError, ValueError) as exc:
        print(f"[last-map] remembered selection rejected ({exc}); using default", file=sys.stderr)
        return None
    return selection


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args != ["show"]:
        print(__doc__, file=sys.stderr)
        return 2
    selection = load_remembered_selection()
    if selection is not None:
        print(selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
