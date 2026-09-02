from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "plan_complementary_reference_bank.py"
SPEC = importlib.util.spec_from_file_location("complementary_reference_bank", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(seq: int, success: bool, refs: list[str], mode: str = "TRACK") -> dict:
    return {"display_seq": seq, "success": success, "refs": refs, "mode": mode}


def test_different_coordinate_frames_block_reference_union() -> None:
    plan = MODULE.build_plan(
        frame_a="map-a",
        rows_a=[_row(10, True, ["a.jpg"])],
        frame_b="map-b",
        rows_b=[_row(10, True, ["b.jpg"])],
        independent_holdout="P169.mp4",
    )

    assert plan["status"] == "BLOCKED_DIFFERENT_COORDINATE_FRAMES"
    assert plan["candidate_reference_names"] == []
    assert plan["deployment_authorized"] is False


def test_same_frame_selects_successful_refs_from_better_windows() -> None:
    plan = MODULE.build_plan(
        frame_a="shared-map",
        rows_a=[_row(10, True, ["a-good.jpg"]), _row(260, False, [])],
        frame_b="shared-map",
        rows_b=[_row(10, False, []), _row(260, True, ["b-good.jpg"])],
        independent_holdout="P169.mp4",
    )

    assert plan["status"] == "SHADOW_PLAN_READY"
    assert plan["candidate_reference_names"] == ["a-good.jpg", "b-good.jpg"]
    assert [row["preferred"] for row in plan["window_comparisons"]] == ["A", "B"]


def test_same_frame_without_reference_telemetry_stays_unready() -> None:
    plan = MODULE.build_plan(
        frame_a="shared-map",
        rows_a=[{"display_seq": 1, "success": True, "mode": "TRACK"}],
        frame_b="shared-map",
        rows_b=[{"display_seq": 1, "success": False, "mode": "LOST"}],
        independent_holdout="P169.mp4",
    )

    assert plan["status"] == "NEEDS_REFERENCE_TELEMETRY"
