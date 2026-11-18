#!/usr/bin/env python3
"""Train/evaluate bounded ANAFI residual PCMD in a pure offline reduced-order plant.

Training: train_anafi_wind.py --timesteps 300000 --seed 17 --n-envs 4 --out DIR
Frozen evaluation: train_anafi_wind.py --evaluate-only DIR/policy.zip --out NEW_DIR
Neither mode connects to an aircraft, starts Sphinx, or authorizes live flight.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
import uuid
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

TOOLS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
from anafi_wind_env import AnafiWindEnv, WindEnvConfig  # noqa: E402

SCENARIOS = ("calm", "steady", "gust", "reversal", "dropout", "stress")
SEEN_SCENARIOS = SCENARIOS[:-1]
METRIC_NAMES = (
    "duration_s", "horizontal_rmse_m", "horizontal_p95_m", "max_error_m",
    "max_ground_speed_mps", "pcmd_peak_pct", "residual_rms_pct",
    "saturation_fraction", "failure", "failure_reason", "recovery_time_s",
)
NUMERIC_METRICS = tuple(name for name in METRIC_NAMES if name not in {
    "failure", "failure_reason", "recovery_time_s",
})
MAX_SEED = 2**32 - 1
PPO_PARAMETERS = {
    "policy": "MlpPolicy", "learning_rate": 3e-4, "n_steps": 1024,
    "batch_size": 256, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95,
    "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5,
    "max_grad_norm": 0.5, "policy_kwargs": {"net_arch": [64, 64]},
    "device": "cpu", "verbose": 0,
}


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _emit(event: str, **fields: Any) -> None:
    print(_json_text({"event": event, **fields}), flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    """Record every actually imported repository Python dependency, not git state."""
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name is None:
            continue
        path = Path(name).resolve()
        if path.suffix == ".py" and path.is_relative_to(REPO_ROOT) and path.is_file():
            paths.add(path)
    return {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in sorted(paths)}


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in (
            "numpy", "gymnasium", "stable-baselines3", "torch",
        )},
    }


class TrainingProgress(BaseCallback):
    def __init__(self, requested_timesteps: int) -> None:
        super().__init__()
        self.requested_timesteps = requested_timesteps
        self.started = 0.0
        self.episodes = 0
        self.failures = 0

    def _on_training_start(self) -> None:
        self.started = time.monotonic()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            metrics = info.get("episode_metrics")
            if metrics is not None:
                self.episodes += 1
                self.failures += int(bool(metrics["failure"]))
        return True

    def _on_rollout_end(self) -> None:
        elapsed = time.monotonic() - self.started
        _emit(
            "training_progress", timesteps=self.num_timesteps,
            requested_timesteps=self.requested_timesteps,
            elapsed_s=round(elapsed, 3),
            timesteps_per_s=round(self.num_timesteps / max(elapsed, 1e-9), 1),
            completed_episodes=self.episodes, failed_episodes=self.failures,
        )


def _episode(
    config: WindEnvConfig, scenario: str, seed: int, model: PPO | None,
) -> dict[str, Any]:
    env = AnafiWindEnv(config)
    try:
        observation, _ = env.reset(seed=seed, options={"scenario": scenario})
        contract = env.contract()
        max_steps = int(contract["max_episode_steps"])
        if max_steps <= 0:
            raise ValueError("environment contract max_episode_steps must be positive")
        for step in range(1, max_steps + 1):
            if model is None:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if "episode_metrics" not in info:
                    raise RuntimeError("final environment info is missing episode_metrics")
                metrics = dict(info["episode_metrics"])
                missing = set(METRIC_NAMES) - metrics.keys()
                if missing:
                    raise RuntimeError(f"episode_metrics is missing {sorted(missing)}")
                if metrics.get("scenario") != scenario or metrics.get("seed") != seed:
                    raise RuntimeError("episode_metrics scenario/seed does not match reset")
                _json_text(metrics)  # Reject non-finite output instead of hiding it.
                metrics.update(
                    steps=step, terminated=bool(terminated), truncated=bool(truncated),
                    ended_before_horizon=step < max_steps,
                    configured_duration_s=float(config.episode_seconds),
                )
                return metrics
        raise RuntimeError(f"environment exceeded its finite horizon ({max_steps} steps)")
    finally:
        env.close()


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summarize(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    episodes = [row[arm] for row in rows]
    recovery = [float(item["recovery_time_s"]) for item in episodes
                if item["recovery_time_s"] is not None]
    recovery_status_counts: dict[str, int] = {}
    failure_reason_counts: dict[str, int] = {}
    for item in episodes:
        status = str(item.get("recovery_status", "observed" if item["recovery_time_s"] is not None else "null"))
        recovery_status_counts[status] = recovery_status_counts.get(status, 0) + 1
        if item["failure"]:
            reason = str(item["failure_reason"])
            failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1
    return {
        "episodes": len(episodes),
        "failures": sum(bool(item["failure"]) for item in episodes),
        "failure_reason_counts": failure_reason_counts,
        "early_ended_episodes": sum(item["ended_before_horizon"] for item in episodes),
        "mean": {name: _mean([float(item[name]) for item in episodes]) for name in NUMERIC_METRICS},
        "worst_max_error_m": max(float(item["max_error_m"]) for item in episodes),
        "worst_horizontal_p95_m": max(float(item["horizontal_p95_m"]) for item in episodes),
        "shortest_actual_duration_s": min(float(item["duration_s"]) for item in episodes),
        "recovery_observed_count": len(recovery),
        "recovery_null_count": len(episodes) - len(recovery),
        "recovery_status_counts": recovery_status_counts,
        "mean_recovery_time_s_observed_only": _mean(recovery),
    }


def _comparison(rows: list[dict[str, Any]], before: str = "baseline", after: str = "learned") -> dict[str, Any]:
    recoveries = [
        float(row[after]["recovery_time_s"]) - float(row[before]["recovery_time_s"])
        for row in rows
        if row[after]["recovery_time_s"] is not None and row[before]["recovery_time_s"] is not None
    ]
    return {
        before: _summarize(rows, before),
        after: _summarize(rows, after),
        f"mean_paired_delta_{after}_minus_{before}": {
            **{name: _mean([float(row[after][name]) - float(row[before][name]) for row in rows])
               for name in NUMERIC_METRICS},
            "recovery_time_s_observed_in_both_only": _mean(recoveries),
        },
        "recovery_pairs_observed_in_both": len(recoveries),
        "paired_failure_outcomes": {
            "both_failed": sum(row[before]["failure"] and row[after]["failure"] for row in rows),
            f"{before}_only_failed": sum(row[before]["failure"] and not row[after]["failure"] for row in rows),
            f"{after}_only_failed": sum(row[after]["failure"] and not row[before]["failure"] for row in rows),
            "neither_failed": sum(not row[before]["failure"] and not row[after]["failure"] for row in rows),
        },
    }


def _evaluate(
    model: PPO, config: WindEnvConfig, seeds: list[int], out: Path, reference: PPO | None = None,
) -> dict[str, Any]:
    rows = []
    arms = ("baseline", "learned") if reference is None else ("baseline", "reference", "learned")
    for scenario in SCENARIOS:
        for seed in seeds:
            baseline = _episode(config, scenario, seed, None)
            learned = _episode(config, scenario, seed, model)
            recovery_delta = None
            if baseline["recovery_time_s"] is not None and learned["recovery_time_s"] is not None:
                recovery_delta = float(learned["recovery_time_s"]) - float(baseline["recovery_time_s"])
            delta = {name: float(learned[name]) - float(baseline[name]) for name in NUMERIC_METRICS}
            delta["recovery_time_s"] = recovery_delta
            rows.append({
                "scenario": scenario, "seed": seed, "held_out_stress": scenario == "stress",
                "config": asdict(config), "baseline": baseline, "learned": learned,
                "delta_learned_minus_baseline": delta,
            })
            if reference is not None:
                rows[-1]["reference"] = _episode(config, scenario, seed, reference)
            _emit(
                "evaluation_pair", scenario=scenario, seed=seed,
                baseline_rmse_m=baseline["horizontal_rmse_m"],
                learned_rmse_m=learned["horizontal_rmse_m"],
                baseline_failure=baseline["failure"], learned_failure=learned["failure"],
                baseline_duration_s=baseline["duration_s"], learned_duration_s=learned["duration_s"],
            )
    report = {
        "schema_version": 1, "scope": "offline residual-ground-drift proxy, not atmospheric wind or aircraft certification",
        "comparison": "deterministic frozen PPO versus zero residual with identical config/scenario/seed",
        "aggregation": "equal-weight episode means; failed and short episodes included, never reset/replaced; maximum errors and actual durations retained",
        "recovery_aggregation": "null is not zero; observed-only means include counts/status; paired delta requires recovery observed in both arms",
        "evaluation_seeds": seeds, "config": asdict(config), "pairs": rows,
        "by_scenario": {scenario: _comparison([row for row in rows if row["scenario"] == scenario])
                        for scenario in SCENARIOS},
        "training_scenarios_aggregate": _comparison([row for row in rows if row["scenario"] in SEEN_SCENARIOS]),
        "held_out_stress_aggregate": _comparison([row for row in rows if row["scenario"] == "stress"]),
        "all_scenarios_aggregate": _comparison(rows),
        "live_flight_authorized": False,
    }
    if reference is not None:
        report["reference_comparison"] = {
            "scope": "previous frozen policy and candidate on the same CURRENT environment; not historical environment reproduction",
            "by_scenario": {scenario: _comparison(
                [row for row in rows if row["scenario"] == scenario], "reference", "learned",
            ) for scenario in SCENARIOS},
            "training_scenarios_aggregate": _comparison(
                [row for row in rows if row["scenario"] in SEEN_SCENARIOS], "reference", "learned",
            ),
            "held_out_stress_aggregate": _comparison(
                [row for row in rows if row["scenario"] == "stress"], "reference", "learned",
            ),
        }
    _write_json(out / "evaluation.json", report)
    fields = ["scenario", "seed", "held_out_stress", "config_json"]
    metric_fields = sorted(set().union(*(row[arm].keys() for row in rows for arm in arms)))
    fields += [f"{arm}.{name}" for arm in arms for name in metric_fields]
    fields += [f"delta_learned_minus_baseline.{name}" for name in (*NUMERIC_METRICS, "recovery_time_s")]
    with (out / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {name: row[name] for name in ("scenario", "seed", "held_out_stress")}
            flat["config_json"] = _json_text(row["config"])
            for arm in (*arms, "delta_learned_minus_baseline"):
                flat.update({f"{arm}.{name}": value if not isinstance(value, (dict, list, tuple)) else _json_text(value)
                             for name, value in row[arm].items()})
            writer.writerow(flat)
    _emit("evaluation_summary", training_scenarios=report["training_scenarios_aggregate"],
          held_out_stress=report["held_out_stress_aggregate"])
    return report


def _new_output(path: Path | None, mode: str) -> Path:
    if path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = REPO_ROOT / "outputs" / "anafi_rl" / f"{mode}-{stamp}-{uuid.uuid4().hex[:8]}"
    path = path.expanduser().resolve()
    if path.exists():
        raise ValueError(f"output already exists, refusing to overwrite: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _evaluation_seeds(explicit: list[int] | None, base_seed: int, n_envs: int) -> list[int]:
    training_seeds = set(range(base_seed, base_seed + n_envs))
    seeds = explicit if explicit is not None else [
        (base_seed + 1_000_000 + index) % (MAX_SEED + 1) for index in range(3)
    ]
    if len(set(seeds)) != len(seeds):
        raise ValueError("--eval-seeds must not contain duplicates")
    if any(seed < 0 or seed > MAX_SEED for seed in seeds):
        raise ValueError(f"evaluation seeds must be between 0 and {MAX_SEED}")
    if training_seeds.intersection(seeds):
        raise ValueError("evaluation seeds must be disjoint from all training initialization seeds")
    return seeds


def _contract(config: WindEnvConfig) -> dict[str, Any]:
    env = AnafiWindEnv(config)
    try:
        return json.loads(_json_text(env.contract()))
    finally:
        env.close()


def _load_reference(path: Path, contract: dict) -> tuple[PPO, dict]:
    path = path.expanduser().resolve()
    metadata_path = path.parent / "metadata.json"
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    if saved.get("mode") != "train" or saved.get("model_sha256") != _sha256(path):
        raise ValueError("reference model requires matching training metadata and model SHA256")
    old_contract = saved["environment_contract"]
    if old_contract["observation"] != contract["observation"]:
        raise ValueError("reference actor observation semantics differ from current environment")
    for field in ("shape", "dtype", "bounds", "order", "residual_cap_pct"):
        if old_contract["action"][field] != contract["action"][field]:
            raise ValueError(f"reference action field differs: {field}")
    model = PPO.load(path, device="cpu")
    if list(model.observation_space.shape) != contract["observation"]["shape"]:
        raise ValueError("reference policy observation shape does not match its metadata")
    return model, {
        "path": str(path), "model_sha256": _sha256(path),
        "metadata_sha256": _sha256(metadata_path),
        "training_contract": old_contract,
        "comparison_scope": "same current environment, prior weights frozen; historical source hash mismatch expected and recorded",
    }


def _run(args: argparse.Namespace) -> Path:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    mode = "evaluate" if args.evaluate_only is not None else "train"
    training_metadata = None
    if args.evaluate_only is not None:
        if args.timesteps is not None or args.n_envs is not None or args.seed is not None:
            raise ValueError("--evaluate-only uses saved training configuration; do not pass --timesteps, --n-envs, or --seed")
        model_path = args.evaluate_only.expanduser().resolve()
        if not model_path.is_file():
            raise ValueError(f"saved model does not exist: {model_path}")
        metadata_path = model_path.parent / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError(f"saved model requires its training metadata: {metadata_path}")
        training_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if training_metadata.get("mode") != "train":
            raise ValueError("model metadata must come from a training run")
        expected_hash = training_metadata.get("model_sha256")
        if expected_hash != _sha256(model_path):
            raise ValueError("saved model SHA256 does not match training metadata")
        config = WindEnvConfig(**training_metadata["environment_config"])
        base_seed = int(training_metadata["training"]["base_seed"])
        n_envs = int(training_metadata["training"]["n_envs"])
        contract = _contract(config)
        if contract != training_metadata["environment_contract"]:
            raise ValueError("current environment contract differs from saved model training contract")
        for name, expected in training_metadata["source_sha256"].items():
            if name == str(Path(__file__).resolve().relative_to(REPO_ROOT)):
                continue
            source = REPO_ROOT / name
            if not source.is_file() or _sha256(source) != expected:
                raise ValueError(f"training environment source changed: {name}")
    else:
        base_seed = 17 if args.seed is None else args.seed
        n_envs = 4 if args.n_envs is None else args.n_envs
        if n_envs <= 0:
            raise ValueError("--n-envs must be positive")
        if base_seed < 0 or base_seed + n_envs - 1 > MAX_SEED:
            raise ValueError(f"--seed and --n-envs must keep every initial Gym seed within 0..{MAX_SEED}")
        config = WindEnvConfig()
        contract = _contract(config)
    seeds = _evaluation_seeds(args.eval_seeds, base_seed, n_envs)
    requested = 300000 if args.timesteps is None else args.timesteps
    if requested <= 0:
        raise ValueError("--timesteps must be positive")
    reference, reference_metadata = (
        _load_reference(args.reference_model, contract)
        if args.reference_model is not None else (None, None)
    )
    out = _new_output(args.out, mode)
    metadata = {
        "schema_version": 1, "mode": mode, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "pure offline NumPy reduced-order plant; bounded roll/pitch residual to existing controller",
        "live_flight_authorized": False, "device": "cpu", "torch_threads": 1,
        "torch_interop_threads": 1, "versions": _versions(), "platform": platform.platform(),
        "environment_config": asdict(config), "environment_contract": contract,
        "ppo_parameters": PPO_PARAMETERS,
        "evaluation": {"scenarios": list(SCENARIOS), "held_out_scenarios": ["stress"],
                       "seeds": seeds, "deterministic_policy": True,
                       "baseline": "zero normalized residual action; unchanged controller/envelope",
                       "seed_separation": "disjoint from training initialization seeds; episode random streams are not replayed",
                       "null_recovery": "not observed or not applicable, as defined by environment contract and episode recovery_status; never replaced by zero"},
        "source_sha256": _source_hashes(),
    }
    if reference_metadata is not None:
        metadata["reference_model"] = reference_metadata
    if training_metadata is None:
        metadata["training"] = {
            "base_seed": base_seed, "n_envs": n_envs,
            "gym_initial_seeds": list(range(base_seed, base_seed + n_envs)),
            "subsequent_gym_resets": "seed=None advances each independently seeded environment RNG",
            "requested_timesteps": requested,
            "rollout_timesteps": n_envs * PPO_PARAMETERS["n_steps"],
            "expected_actual_timesteps": math.ceil(requested / (n_envs * PPO_PARAMETERS["n_steps"])) * n_envs * PPO_PARAMETERS["n_steps"],
            "timestep_rounding": "SB3 completes full rollouts; actual count may exceed request by less than one rollout",
            "scenarios": list(SEEN_SCENARIOS), "vectorization": "DummyVecEnv",
        }
        _write_json(out / "metadata.json", metadata)
        vector_env = DummyVecEnv([lambda: Monitor(AnafiWindEnv(config)) for _ in range(n_envs)])
        try:
            model = PPO(env=vector_env, seed=base_seed, **PPO_PARAMETERS)
            _emit("training_start", output=str(out), **metadata["training"])
            started = time.monotonic()
            model.learn(total_timesteps=requested, callback=TrainingProgress(requested), progress_bar=False)
            metadata["training"]["wall_training_time_s"] = time.monotonic() - started
            metadata["training"]["actual_timesteps"] = int(model.num_timesteps)
            model_path = out / "policy.zip"
            model.save(model_path)
        finally:
            vector_env.close()
        metadata["source_sha256"] = _source_hashes()
        metadata["model_path"] = "policy.zip"
        metadata["model_sha256"] = _sha256(model_path)
        _write_json(out / "metadata.json", metadata)
        del model
    else:
        metadata["training"] = training_metadata["training"]
        metadata["training_ppo_parameters"] = training_metadata["ppo_parameters"]
        metadata["training_source_sha256"] = training_metadata["source_sha256"]
        metadata["training_versions"] = training_metadata["versions"]
        metadata["source_model_path"] = str(model_path)
        metadata["source_training_metadata_path"] = str(metadata_path)
        metadata["model_sha256"] = _sha256(model_path)
    # Evaluation never uses the in-memory trained object: reload the persisted archive.
    model = PPO.load(model_path, device="cpu")
    metadata["model_reloaded_before_evaluation"] = True
    metadata["reloaded_model_timesteps"] = int(model.num_timesteps)
    if int(model.num_timesteps) != int(metadata["training"]["actual_timesteps"]):
        raise RuntimeError("reloaded model timestep count differs from training metadata")
    _write_json(out / "metadata.json", metadata)
    _emit("evaluation_start", output=str(out), model=str(model_path), seeds=seeds,
          scenarios=list(SCENARIOS))
    started = time.monotonic()
    report = _evaluate(model, config, seeds, out, reference)
    metadata["evaluation"]["wall_evaluation_time_s"] = time.monotonic() - started
    metadata["evaluation"]["paired_episodes"] = len(report["pairs"])
    metadata["evaluation"]["total_episodes"] = (3 if reference is not None else 2) * len(report["pairs"])
    metadata["evaluation"]["artifacts"] = ["evaluation.json", "evaluation.csv"]
    metadata["source_sha256"] = _source_hashes()
    _write_json(out / "metadata.json", metadata)
    _emit("complete", output=str(out), model=str(model_path),
          actual_training_timesteps=metadata["training"]["actual_timesteps"],
          live_flight_authorized=False)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timesteps", type=int, default=None, help="finite requested training steps (default 300000; rounded to full rollouts)")
    parser.add_argument("--seed", type=int, default=None, help="training RNG base seed (default 17)")
    parser.add_argument("--out", type=Path, help="new output directory; existing paths are refused")
    parser.add_argument("--eval-seeds", type=int, nargs="+", help="distinct explicit evaluation seeds (default base seed + 1000000, +1000001, +1000002)")
    parser.add_argument("--n-envs", type=int, default=None, help="offline CPU environments (default 4)")
    parser.add_argument("--evaluate-only", type=Path, metavar="MODEL", help="reload policy.zip with sibling metadata.json; no training")
    parser.add_argument("--reference-model", type=Path, metavar="MODEL", help="compare a previous frozen actor on the same current environment; matching observation/action semantics and model metadata required")
    args = parser.parse_args(argv)
    try:
        _run(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
