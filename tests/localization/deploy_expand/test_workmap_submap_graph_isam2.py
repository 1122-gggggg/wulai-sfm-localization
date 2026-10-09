"""iSAM2 submap graph: incremental edges agree with the batch solve (needs gtsam)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gtsam")

from test_workmap_submap_graph import TRUTH, _edge, _noisy, _payload  # noqa: E402
from workmap_submap_graph import (  # noqa: E402
    _ISAM2_CACHE,
    IncrementalSubmapGraph,
    parse_payload,
    solve_batch,
    solve_submap_graph,
)


def _edges():
    rng = np.random.default_rng(11)
    return [
        _edge(0, 1, "merge", _noisy(TRUTH[1], rng)),
        _edge(0, 2, "merge", _noisy(TRUTH[2], rng)),
        _edge(2, 3, "sequential", _noisy(TRUTH[2].inverse().compose(TRUTH[3]), rng)),
        _edge(0, 1, "overlap", _noisy(TRUTH[1], rng)),
        _edge(0, 3, "overlap", _noisy(TRUTH[3], rng)),
    ]


def test_each_added_edge_matches_the_batch_solution_with_one_update_cycle():
    edges = _edges()
    graph = IncrementalSubmapGraph()
    for count in range(1, len(edges) + 1):
        free, initial, parsed = parse_payload(_payload(edges[:count]))
        updates = graph.updates
        nodes, _before, _after = graph.solve(free, initial, parsed)
        assert graph.updates == updates + 1                      # one update cycle per new edge
        batch, _, _ = solve_batch(free, initial, parsed)
        for node, vector in batch.items():
            np.testing.assert_allclose(nodes[node], vector, atol=1e-3, err_msg=f"{count} edges, node {node}")
    updates = graph.updates
    graph.solve(free, initial, parsed)                           # nothing new: no update
    assert graph.updates == updates


def test_a_payload_that_does_not_extend_the_cached_graph_rebuilds_it():
    edges = _edges()
    first = solve_submap_graph({**_payload(edges[:3]), "solver": "isam2"})
    assert first["ok"] and first["solver"] == "isam2"
    (cached,) = _ISAM2_CACHE.values()
    again = solve_submap_graph({**_payload(edges[:4]), "solver": "isam2"})
    assert again["ok"] and _ISAM2_CACHE[next(iter(_ISAM2_CACHE))] is cached and cached.updates == 2
    changed = [dict(edges[0], sigma=[2 * v for v in edges[0]["sigma"]]), *edges[1:4]]
    rebuilt = solve_submap_graph({**_payload(changed), "solver": "isam2"})
    assert rebuilt["ok"] and next(iter(_ISAM2_CACHE.values())) is not cached
    batch = solve_submap_graph({**_payload(changed), "solver": "batch"})
    for node in (1, 2, 3):
        np.testing.assert_allclose(rebuilt["nodes"][node], batch["nodes"][node], atol=1e-3)


@pytest.mark.parametrize("outlier", [False, True])
def test_large_merge_corrections_and_outliers_match_componentwise_batch_loss(outlier):
    measurement = [0.02, -0.01, 0.03, 0.35, -0.2, 0.1, np.log(0.7)]
    edges = [
        _edge(0, 1, "sequential", [0.0] * 7, sigma=[0.2] * 7),
        _edge(0, 1, "merge", measurement),
    ]
    if outlier:
        edges.append(_edge(0, 1, "overlap", [0.02, -0.01, 0.03, 1.5, -0.2, 0.1, np.log(0.7)]))
    graph = IncrementalSubmapGraph()
    for count in range(1, len(edges) + 1):
        free, initial, parsed = parse_payload(_payload(edges[:count]))
        nodes, _, after = graph.solve(free, initial, parsed)
        batch, _, batch_cost = solve_batch(free, initial, parsed)
        np.testing.assert_allclose(nodes[1], batch[1], atol=1e-3)
        assert after == pytest.approx(batch_cost, abs=1e-4)
