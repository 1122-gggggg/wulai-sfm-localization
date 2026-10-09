"""Sim(3) pose graph over the submaps."""
from __future__ import annotations

import math

import numpy as np
import pytest

from workmap_submap_graph import edge_residual, solve_submap_graph
from workmap_submaps import Sim3


def _sim(rotvec_deg, t, scale):
    return Sim3.from_vector([*np.radians(rotvec_deg), *t, math.log(scale)])


TRUTH = {
    1: _sim([0.5, -1.0, 2.0], [0.30, -0.10, 0.05], 1.08),
    2: _sim([-1.0, 0.5, -2.5], [-0.20, 0.25, 0.10], 0.93),
}
TRUTH[3] = TRUTH[2].compose(_sim([0.3, 0.2, -0.4], [0.05, 0.02, -0.03], 1.02))
SIGMA = [math.radians(0.5)] * 3 + [0.01] * 3 + [0.02]


def _noisy(sim, rng, rot_deg=0.1, t=0.002, log_scale=0.002):
    noise = np.concatenate(
        [np.radians(rng.normal(0, rot_deg, 3)), rng.normal(0, t, 3), rng.normal(0, log_scale, 1)]
    )
    return [float(v) for v in sim.to_vector() + noise]


def _edge(a, b, kind, z, sigma=SIGMA):
    return {"a": a, "b": b, "kind": kind, "z": z, "sigma": list(sigma)}


def _payload(edges, initial=None):
    return {
        "session_id": "s", "source_epoch": 0, "solver": "batch", "nodes": [0, 1, 2, 3],
        "initial": initial
        if initial is not None
        else {str(k): [0.0] * 7 for k in (1, 2, 3)},
        "edges": edges,
    }


def _error(result, node):
    solved, truth = Sim3.from_vector(result["nodes"][node]), TRUTH[node]
    delta = solved.compose(truth.inverse())
    return delta.rotation_deg(), float(np.linalg.norm(delta.translation)), abs(math.log(delta.scale))


def _check(result, nodes=(1, 2, 3)):
    assert result["ok"], result["reason"]
    assert result["nodes"][0] == [0.0] * 7
    for node in nodes:
        rotation, translation, log_scale = _error(result, node)
        assert rotation < 0.5 and translation < 0.01 and log_scale < 0.01, (node, rotation, translation, log_scale)


def test_edge_residual_vanishes_for_a_consistent_edge():
    z = TRUTH[2].inverse().compose(TRUTH[3])
    residual = edge_residual(TRUTH[2].to_vector(), TRUTH[3].to_vector(), z.to_vector(), SIGMA)
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_noisy_merge_and_sequential_edges_recover_the_submap_poses():
    rng = np.random.default_rng(5)
    edges = [
        _edge(0, 1, "merge", _noisy(TRUTH[1], rng)),
        _edge(0, 2, "merge", _noisy(TRUTH[2], rng)),
        _edge(2, 3, "sequential", _noisy(TRUTH[2].inverse().compose(TRUTH[3]), rng)),
    ]
    result = solve_submap_graph(_payload(edges))
    _check(result)
    assert result["cost_after"] <= result["cost_before"] and result["solver"] == "batch"


def test_one_outlier_overlap_edge_cannot_drag_a_well_supported_submap():
    rng = np.random.default_rng(6)
    edges = [
        _edge(0, 1, "merge", _noisy(TRUTH[1], rng)),
        _edge(0, 1, "overlap", _noisy(TRUTH[1], rng)),
        _edge(0, 1, "overlap", _noisy(TRUTH[1], rng)),
        _edge(0, 1, "overlap", [*np.zeros(3), 1.0, 0.0, 0.0, 0.0]),            # 1 u off
        _edge(0, 2, "merge", _noisy(TRUTH[2], rng)),
        _edge(2, 3, "sequential", _noisy(TRUTH[2].inverse().compose(TRUTH[3]), rng)),
    ]
    _check(solve_submap_graph(_payload(edges)))


def test_a_good_initial_guess_is_kept_when_the_edges_agree_with_it():
    rng = np.random.default_rng(7)
    edges = [_edge(0, k, "merge", _noisy(TRUTH[k], rng)) for k in (1, 2, 3)]
    initial = {str(k): [float(v) for v in TRUTH[k].to_vector()] for k in (1, 2, 3)}
    result = solve_submap_graph(_payload(edges, initial))
    _check(result)


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda p: p["edges"].append(_edge(0, 9, "merge", [0.0] * 7)), "malformed_graph"),
        (lambda p: p.update(solver="nonsense"), "unknown_solver"),
        (lambda p: p["edges"][0].update(sigma=[0.0] * 7), "malformed_graph"),
        (lambda p: p["initial"].pop("2"), "malformed_graph"),
    ],
)
def test_malformed_graphs_fail_with_a_named_reason(mutate, reason):
    rng = np.random.default_rng(8)
    payload = _payload([_edge(0, 1, "merge", _noisy(TRUTH[1], rng)), _edge(0, 2, "merge", _noisy(TRUTH[2], rng))])
    mutate(payload)
    result = solve_submap_graph(payload)
    assert not result["ok"] and result["reason"].startswith(reason)
