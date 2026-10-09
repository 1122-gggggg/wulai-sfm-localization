"""Sim(3) pose graph over the submaps (runs in the BA process).

Nodes are submap IDs; node 0 is GlueMap and is fixed at the identity. A node value is
the Sim(3) ``S(v)`` that maps the submap's own frame into GlueMap; an edge holds the
measured ``Z_ab`` that maps b's frame into a's frame. Vectors are
``[rotvec(3), t(3), log s]``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from workmap_submaps import Sim3

_HUBER_SCALE = 1.0
_MAX_NFEV = 50
SOLVERS = ("batch", "isam2")


def edge_residual(va: Any, vb: Any, z_vec: Any, sigma: Any) -> np.ndarray:
    """``[rotvec, t, log s]`` of ``Z^-1 o S(va)^-1 o S(vb)``, divided by ``sigma`` (7,)."""
    delta = (
        Sim3.from_vector(z_vec)
        .inverse()
        .compose(Sim3.from_vector(va).inverse())
        .compose(Sim3.from_vector(vb))
    )
    return delta.to_vector() / np.asarray(sigma, dtype=np.float64).reshape(7)


def _failure(reason: str, solver: str) -> dict[str, Any]:
    return {
        "ok": False, "reason": reason, "nodes": {}, "cost_before": None, "cost_after": None,
        "solver_ms": 0.0, "solver": solver,
    }


def parse_payload(payload: Mapping[str, Any]) -> tuple[list[int], dict[int, np.ndarray], list[dict]]:
    """Validated ``(free node IDs, initial vectors by ID, edges)``; raises ``ValueError``."""
    nodes = [int(node) for node in payload["nodes"]]
    if 0 not in nodes or len(set(nodes)) != len(nodes):
        raise ValueError("nodes must be unique and include GlueMap node 0")
    initial = {
        int(key): np.asarray(value, dtype=np.float64).reshape(7)
        for key, value in payload["initial"].items()
    }
    initial[0] = np.zeros(7)
    edges = []
    for raw in payload["edges"]:
        edge = {
            "a": int(raw["a"]),
            "b": int(raw["b"]),
            "kind": str(raw["kind"]),
            "z": np.asarray(raw["z"], dtype=np.float64).reshape(7),
            "sigma": np.asarray(raw["sigma"], dtype=np.float64).reshape(7),
        }
        if edge["a"] not in nodes or edge["b"] not in nodes or edge["a"] == edge["b"]:
            raise ValueError("edge refers to an unknown node")
        if not (np.isfinite(edge["z"]).all() and (edge["sigma"] > 0).all()):
            raise ValueError("edge measurement must be finite with positive sigma")
        edges.append(edge)
    free = [node for node in nodes if node != 0]
    for node in free:
        if node not in initial or not np.isfinite(initial[node]).all():
            raise ValueError(f"node {node} has no finite initial value")
    return free, initial, edges


def _robust_cost(residual: np.ndarray) -> float:
    squared = residual**2
    scale = _HUBER_SCALE
    return float(
        0.5 * np.sum(np.where(squared <= scale**2, squared, 2.0 * scale * np.sqrt(squared) - scale**2))
    )


def solve_batch(
    free: list[int], initial: Mapping[int, np.ndarray], edges: list[dict]
) -> tuple[dict[int, np.ndarray], float, float]:
    from scipy.optimize import least_squares

    def values(x: np.ndarray) -> dict[int, np.ndarray]:
        out = {0: np.zeros(7)}
        for slot, node in enumerate(free):
            out[node] = x[7 * slot : 7 * slot + 7]
        return out

    def residuals(x: np.ndarray) -> np.ndarray:
        current = values(x)
        return np.concatenate(
            [edge_residual(current[e["a"]], current[e["b"]], e["z"], e["sigma"]) for e in edges]
        )

    x0 = np.concatenate([initial[node] for node in free]) if free else np.zeros(0)
    before = _robust_cost(residuals(x0))
    if not free or not edges:
        return {node: np.asarray(initial[node]) for node in [0, *free]}, before, before
    solution = least_squares(
        residuals, x0, method="trf", loss="huber", f_scale=_HUBER_SCALE, max_nfev=_MAX_NFEV
    )
    return values(solution.x), before, _robust_cost(residuals(solution.x))


_JACOBIAN_EPS = 1e-6
_ISAM2_STEP_TOL = 1e-5
_ISAM2_MAX_REFINEMENTS = 8
#: BA-process cache of incremental graphs by ``(session_id, source_epoch)``.
_ISAM2_CACHE: dict[tuple[str, int], "IncrementalSubmapGraph"] = {}


def _edge_signature(edge: Mapping[str, Any]) -> tuple:
    return (
        int(edge["a"]),
        int(edge["b"]),
        tuple(float(v) for v in edge["z"]),
        tuple(float(v) for v in edge["sigma"]),
    )


def _huber_weight(residual: float) -> float:
    magnitude = abs(residual)
    return 1.0 if magnitude <= _HUBER_SCALE else _HUBER_SCALE / magnitude


def _huber_loss(residual: float) -> float:
    magnitude = abs(residual)
    if magnitude <= _HUBER_SCALE:
        return 0.5 * residual * residual
    return _HUBER_SCALE * (magnitude - 0.5 * _HUBER_SCALE)


class IncrementalSubmapGraph:
    """iSAM2 over the submap Sim(3) vectors; one update cycle per new batch of edges."""

    def __init__(self) -> None:
        import gtsam  # ImportError -> solve_submap_graph reports gtsam_unavailable

        self._g = gtsam
        parameters = gtsam.ISAM2Params()
        parameters.relinearizeSkip = 1
        parameters.setRelinearizeThreshold(_ISAM2_STEP_TOL)
        parameters.setFactorization("QR")
        self._isam = gtsam.ISAM2(parameters)
        self._signatures: list[tuple] = []
        self._nodes: set[int] = set()
        self.updates = 0

    def key(self, node: int) -> int:
        return self._g.symbol("s", int(node))

    def accepts(self, edges: list[dict]) -> bool:
        """Whether the payload's edges extend exactly what this graph already holds."""
        held = len(self._signatures)
        return len(edges) >= held and [_edge_signature(e) for e in edges[:held]] == self._signatures

    def _factor(self, edge: Mapping[str, Any]):
        g = self._g
        key_a, key_b = self.key(edge["a"]), self.key(edge["b"])
        z, sigma = edge["z"], edge["sigma"]

        def error(this, values, jacobians):
            va, vb = np.asarray(values.atVector(key_a)), np.asarray(values.atVector(key_b))
            residual = edge_residual(va, vb, z, sigma)
            if jacobians is not None:
                for slot, (own, other) in enumerate(((va, vb), (vb, va))):
                    jacobian = np.zeros((7, 7))
                    for column in range(7):
                        step = np.zeros(7)
                        step[column] = _JACOBIAN_EPS
                        if slot == 0:
                            plus = edge_residual(own + step, other, z, sigma)
                            minus = edge_residual(own - step, other, z, sigma)
                        else:
                            plus = edge_residual(other, own + step, z, sigma)
                            minus = edge_residual(other, own - step, z, sigma)
                        jacobian[:, column] = (plus - minus) / (2.0 * _JACOBIAN_EPS)
                    jacobians[slot] = jacobian
            return residual

        # The bound Huber constructor only exposes block reweighting. Match
        # SciPy's componentwise loss, not Huber applied to the 7-vector norm.
        estimator = g.noiseModel.mEstimator.Custom.Create(
            _huber_weight, _huber_loss,
            g.noiseModel.mEstimator.Base.ReweightScheme.Scalar, "ScalarHuber",
        )
        model = g.noiseModel.Robust.Create(estimator, g.noiseModel.Unit.Create(7))
        return g.CustomFactor(model, [key_a, key_b], error)

    def solve(
        self, free: list[int], initial: Mapping[int, np.ndarray], edges: list[dict]
    ) -> tuple[dict[int, np.ndarray], float, float]:
        g = self._g
        new_edges = edges[len(self._signatures) :]
        graph, values = g.NonlinearFactorGraph(), g.Values()
        wanted = {0} | {e["a"] for e in new_edges} | {e["b"] for e in new_edges}
        for node in sorted(wanted - self._nodes):
            values.insert(self.key(node), np.asarray(initial[node], dtype=np.float64))
            if node == 0:
                graph.add(
                    g.PriorFactorVector(
                        self.key(0), np.zeros(7), g.noiseModel.Isotropic.Sigma(7, 1e-6)
                    )
                )
        for edge in new_edges:
            graph.add(self._factor(edge))
        self._nodes |= wanted
        if new_edges or values.size():
            self._isam.update(graph, values)
            estimate = self._isam.calculateEstimate()
            # Robust weights change after a large merge correction. A fixed
            # second update can publish an unconverged transform.
            for _ in range(_ISAM2_MAX_REFINEMENTS):
                previous = estimate
                self._isam.update()
                estimate = self._isam.calculateEstimate()
                change = max(
                    float(np.max(np.abs(
                        estimate.atVector(self.key(node)) - previous.atVector(self.key(node))
                    )))
                    for node in self._nodes
                )
                if change <= _ISAM2_STEP_TOL:
                    break
            self.updates += 1
        else:
            estimate = self._isam.calculateEstimate()
        self._signatures.extend(_edge_signature(e) for e in new_edges)
        nodes = {
            int(node): (
                np.asarray(estimate.atVector(self.key(node)))
                if node in self._nodes
                else np.asarray(initial[node], dtype=np.float64)
            )
            for node in [0, *free]
        }

        def cost(current: Mapping[int, np.ndarray]) -> float:
            if not edges:
                return 0.0
            return _robust_cost(
                np.concatenate(
                    [edge_residual(current[e["a"]], current[e["b"]], e["z"], e["sigma"]) for e in edges]
                )
            )

        return nodes, cost(initial), cost(nodes)


def solve_isam2(
    payload: Mapping[str, Any],
    free: list[int],
    initial: Mapping[int, np.ndarray],
    edges: list[dict],
) -> tuple[dict[int, np.ndarray], float, float]:
    """Extend the cached incremental graph, or rebuild it when the payload does not extend it."""
    key = (str(payload["session_id"]), int(payload["source_epoch"]))
    graph = _ISAM2_CACHE.get(key)
    if graph is None or not graph.accepts(edges):
        graph = IncrementalSubmapGraph()
        _ISAM2_CACHE.clear()
        _ISAM2_CACHE[key] = graph
    return graph.solve(free, initial, edges)


def solve_submap_graph(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Solve one submap graph; never raises, failures carry a named reason."""
    solver = str(payload.get("solver", "batch")) if isinstance(payload, Mapping) else "batch"
    started = time.perf_counter()
    try:
        if solver not in SOLVERS:
            return _failure("unknown_solver", solver)
        free, initial, edges = parse_payload(payload)
        if solver == "isam2":
            nodes, before, after = solve_isam2(payload, free, initial, edges)
        else:
            nodes, before, after = solve_batch(free, initial, edges)
    except ImportError:
        return _failure("gtsam_unavailable", solver)
    except (KeyError, TypeError, ValueError) as error:
        return _failure(f"malformed_graph:{type(error).__name__}", solver)
    if not all(np.isfinite(vector).all() for vector in nodes.values()):
        return _failure("nonfinite_solution", solver)
    return {
        "ok": True,
        "reason": "",
        "nodes": {int(node): [float(v) for v in vector] for node, vector in nodes.items()},
        "cost_before": before,
        "cost_after": after,
        "solver_ms": (time.perf_counter() - started) * 1000.0,
        "solver": solver,
    }
