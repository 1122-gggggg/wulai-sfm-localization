from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_reference_index import main


def _inputs(tmp_path: Path, *, count: int = 12, dimension: int = 4) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260809)
    descriptors = rng.normal(size=(count, dimension)).astype(np.float32)
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptor_path = tmp_path / "descriptors.npy"
    np.save(descriptor_path, descriptors, allow_pickle=False)
    names_path = tmp_path / "names.json"
    names_path.write_text(
        json.dumps([f"reference-{index:04d}.jpg" for index in range(count)]),
        encoding="utf-8",
    )
    return descriptor_path, names_path


def _command(descriptors: Path, names: Path, output: Path) -> list[str]:
    return [
        "--descriptors",
        str(descriptors),
        "--names",
        str(names),
        "--output",
        str(output),
        "--model-identity",
        "megaloc:test:v1",
        "--nlist",
        "4",
        "--seed",
        "17",
        "--kmeans-iterations",
        "3",
        "--batch-size",
        "4",
        "--max-query-probes",
        "2",
        "--max-query-candidates",
        "8",
    ]


def test_cli_build_is_deterministic_and_publishes_index(tmp_path: Path) -> None:
    descriptors, names = _inputs(tmp_path)
    first = tmp_path / "index-one"
    second = tmp_path / "index-two"

    assert main(_command(descriptors, names, first)) == 0
    assert main(_command(descriptors, names, second)) == 0

    files = (
        "metadata.json",
        "centroids.npy",
        "postings_offsets.npy",
        "postings_indices.npy",
        "descriptors.npy",
        "names.json",
        "SHA256SUMS.json",
    )
    assert all((first / name).read_bytes() == (second / name).read_bytes() for name in files)
    metadata = json.loads((first / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["seed"] == 17
    assert metadata["model_identity"] == "megaloc:test:v1"
    assert metadata["count"] == 12
    assert metadata["dimension"] == 4


def test_cli_rejects_invalid_descriptor_contract(tmp_path: Path) -> None:
    descriptors, names = _inputs(tmp_path)
    invalid_cases = {
        "float64": np.ones((3, 2), dtype=np.float64),
        "nan": np.array([[np.nan, 0.0]], dtype=np.float32),
        "shape": np.ones(4, dtype=np.float32),
    }
    for label, invalid in invalid_cases.items():
        path = tmp_path / f"{label}.npy"
        np.save(path, invalid, allow_pickle=False)
        output = tmp_path / f"{label}-index"
        assert main(_command(path, names, output)) == 2
        assert not output.exists()


def test_cli_rejects_name_count_mismatch_and_existing_output(tmp_path: Path) -> None:
    descriptors, names = _inputs(tmp_path)
    names.write_text(json.dumps(["only-one-name"]), encoding="utf-8")
    output = tmp_path / "mismatch-index"
    assert main(_command(descriptors, names, output)) == 2
    assert not output.exists()

    descriptors, names = _inputs(tmp_path / "second")
    output.mkdir()
    assert main(_command(descriptors, names, output)) == 2
