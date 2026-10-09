from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

VALIDATION_DIR = Path(__file__).resolve().parents[3] / "定位演算法" / "validation"
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

import workmap_closure_experiment as closure  # noqa: E402
import workmap_sim3_experiment as experiment  # noqa: E402


def closure_records(scale):
    points = np.array([
        [x, y, 3 + 0.1 * (index % 3)]
        for index, (x, y) in enumerate(
            (x, y) for x in np.linspace(-0.6, 0.6, 10) for y in np.linspace(-0.3, 0.3, 10)
        )
    ])
    K = np.array([[720.0, 0, 480.0], [0, 720.0, 270.0], [0, 0, 1.0]])
    transform = experiment.Sim3(scale, np.eye(3), np.array([0.2, 0.1, 0.15]))
    records = []
    for index in range(12):
        center = np.array([index * 0.02, np.sin(index * 0.3) * 0.01, 0])
        local = np.column_stack([np.eye(3), -center])
        mapped = transform.apply_pose(local)
        main_xyz = transform.apply_points(points)
        main_pixels = (main_xyz @ mapped[:, :3].T + mapped[:, 3]) @ K.T
        local_pixels = (points @ local[:, :3].T + local[:, 3]) @ K.T
        records.append({
            "ordinal": index * 2, "local_pose": local.tolist(), "map_pose": mapped.tolist(),
            "local_xyz": points.tolist(), "local_xy": (local_pixels[:, :2] / local_pixels[:, 2:]).tolist(),
            "main_xyz": main_xyz.tolist(), "main_xy": (main_pixels[:, :2] / main_pixels[:, 2:]).tolist(),
            "local_inliers": 100, "main_inliers": 100, "K": K.tolist(),
            "local_point_ids": list(range(-100, 0)),
        })
    return records


def test_graph_preserves_nonunit_child_scale_and_frozen_main_geometry():
    records = closure_records(0.61)
    original = json.dumps(records)
    report = experiment.analyze(records)
    assert json.dumps(records) == original
    for name in ("fine_sim3_graph", "fine_sim3_graph_scale_prior"):
        candidate = report[name]
        np.testing.assert_allclose(candidate["solver"]["node_scales"], 0.61, atol=1e-8)
        assert candidate["heldout"]["center_max_u"] < 1e-8
        assert candidate["heldout"]["main_reprojection_p90_px"] < 1e-8


def test_rejected_initial_scale_cannot_be_rescued_by_unadmitted_graph():
    report = experiment.analyze(closure_records(0.36))
    assert report["status"] == "initial_sim3_rejected"
    assert report["training"]["rejection"] == "scale_bounds"
    assert "fine_sim3_graph" not in report
    assert "fine_sim3_graph_scale_prior" not in report


def test_independent_monocular_gauge_uses_its_existing_unbounded_scale_gate():
    records = closure_records(0.03)
    for record in records:
        record["submap_kind"] = "independent"
    original = json.dumps(records)
    report = experiment.analyze(records)
    assert report["training"]["rejection"] is None
    assert report["training"]["scale_bounds"] is None
    for name in ("fine_sim3_graph", "fine_sim3_graph_scale_prior"):
        candidate = report[name]
        np.testing.assert_allclose(candidate["solver"]["node_scales"], 0.03, atol=1e-8)
        assert candidate["heldout"]["main_reprojection_p90_px"] < 1e-8
    assert json.dumps(records) == original


def test_strict_pnp_rejects_conflicting_projections_even_when_loose_fit_passes():
    record = closure_records(0.61)[0]
    # Each pair has identical world coordinates but pixels four units apart.
    # No camera pose can fit both observations within one pixel.
    xyz = np.repeat(np.array(record["main_xyz"])[::2], 2, axis=0)
    xy = np.repeat(np.array(record["main_xy"])[::2], 2, axis=0)
    xy[::2, 0] += 2.0
    xy[1::2, 0] -= 2.0
    ids = np.arange(1, len(xy) + 1, dtype=np.int64)
    K = np.array(record["K"])
    loose = closure.solve_pnp(ids, xyz, xy, K, minimum=80, max_error_px=4.0)
    strict = closure.solve_pnp(ids, xyz, xy, K, minimum=80, max_error_px=1.0)
    assert loose["ok"]
    assert not strict["ok"]


def test_frozen_map_id_cannot_be_counted_as_independent_child_support():
    records = closure_records(0.61)
    records[0]["local_point_ids"][0] = 1
    with np.testing.assert_raises(ValueError):
        experiment.analyze(records)
