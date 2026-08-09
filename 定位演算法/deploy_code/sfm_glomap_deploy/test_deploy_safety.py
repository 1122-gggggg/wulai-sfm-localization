from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_edm_onnx_flight as export_module  # noqa: E402
import production_localizer_factory as factory  # noqa: E402
import reloc_localizer_edm as edm_module  # noqa: E402
from reloc_localizer_edm import Camera, EDMLocalizer  # noqa: E402


def _jpeg_header(*, width: int = 1024, height: int = 576) -> np.ndarray:
    return np.frombuffer(
        b"\xff\xd8\xff\xc0\x00\x11\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00\xff\xd9",
        dtype=np.uint8,
    ).copy()


def _edm_bundle(*, image_size: int | None = None) -> dict:
    name = "ref.jpg"
    image = (
        _jpeg_header()
        if image_size is None
        else np.zeros(image_size, dtype=np.uint8)
    )
    return {
        "meta": {
            "feature": "edm",
            "edm_grid_w": 128,
            "edm_grid_h": 72,
            "edm_input_w": 1024,
            "edm_input_h": 576,
        },
        "ref_names": [name],
        "ref_global": np.ones(
            (1, edm_module.EDM_GLOBAL_DESCRIPTOR_DIM), dtype=np.float32
        ),
        "refs": {
            name: {
                "xyz_by_cell": np.full((128 * 72, 3), np.nan, dtype=np.float32),
                "image_jpg": image,
            },
        },
    }


def test_export_checkpoint_loader_uses_weights_only_and_requires_state_dict(
    tmp_path, monkeypatch,
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"fixture")
    calls = {}

    def fake_load(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return {"state_dict": OrderedDict({"weight": torch.ones(1)})}

    monkeypatch.setattr(export_module.torch, "load", fake_load)

    state = export_module._load_checkpoint_state_dict(checkpoint)

    assert list(state) == ["weight"]
    assert calls == {
        "path": str(checkpoint),
        "kwargs": {"map_location": "cpu", "weights_only": True},
    }


@pytest.mark.parametrize(
    "checkpoint",
    [None, {}, {"state_dict": []}, {"state_dict": {"weight": object()}}],
)
def test_export_checkpoint_loader_rejects_unexpected_payload(
    checkpoint, monkeypatch, tmp_path,
) -> None:
    path = tmp_path / "fixture.ckpt"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(export_module.torch, "load", lambda *_args, **_kwargs: checkpoint)

    with pytest.raises(ValueError, match="state_dict"):
        export_module._load_checkpoint_state_dict(path)


def test_export_checkpoint_loader_rejects_oversized_file_before_torch_load(
    tmp_path, monkeypatch,
) -> None:
    checkpoint = tmp_path / "oversized.ckpt"
    checkpoint.write_bytes(b"xx")
    monkeypatch.setattr(export_module, "EDM_MAX_CHECKPOINT_BYTES", 1)
    monkeypatch.setattr(
        export_module.torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("oversized checkpoint must not be loaded"),
    )

    with pytest.raises(ValueError, match="checkpoint size"):
        export_module._load_checkpoint_state_dict(checkpoint)


def test_export_checkpoint_loader_rejects_symlink(tmp_path) -> None:
    target = tmp_path / "target.ckpt"
    target.write_bytes(b"fixture")
    checkpoint = tmp_path / "link.ckpt"
    checkpoint.symlink_to(target)

    with pytest.raises(ValueError, match="non-symlink"):
        export_module._load_checkpoint_state_dict(checkpoint)


def test_edm_bundle_rejects_100k_references_before_entry_materialization() -> None:
    count = 100_000
    bundle = _edm_bundle()
    bundle["ref_names"] = [f"ref-{index}" for index in range(count)]
    bundle["ref_global"] = np.ones((count, 1), dtype=np.float32)
    bundle["refs"] = {}

    with pytest.raises(ValueError, match="reference count"):
        edm_module._validate_edm_bundle_schema(bundle)


def test_edm_bundle_rejects_encoded_image_before_decode(monkeypatch) -> None:
    bundle = _edm_bundle(image_size=2)
    monkeypatch.setattr(edm_module, "EDM_MAX_IMAGE_ENCODED_BYTES", 1)

    with pytest.raises(ValueError, match="encoded image"):
        edm_module._validate_edm_bundle_schema(bundle)


def test_edm_bundle_rejects_projected_decode_bytes_before_decode(monkeypatch) -> None:
    bundle = _edm_bundle()
    monkeypatch.setattr(edm_module, "EDM_MAX_TOTAL_DECODED_IMAGE_BYTES", 1)

    with pytest.raises(ValueError, match="decoded image"):
        edm_module._validate_edm_bundle_schema(bundle)


def test_edm_bundle_rejects_xyz_memory_before_decode(monkeypatch) -> None:
    bundle = _edm_bundle()
    monkeypatch.setattr(edm_module, "EDM_MAX_XYZ_BYTES", 1)

    with pytest.raises(ValueError, match="XYZ"):
        edm_module._validate_edm_bundle_schema(bundle)


def test_edm_bundle_rejects_oversized_file_before_hash_or_load(
    tmp_path, monkeypatch,
) -> None:
    artifact = tmp_path / "oversized.pt"
    artifact.write_bytes(b"xx")
    monkeypatch.setattr(edm_module, "EDM_MAX_BUNDLE_FILE_BYTES", 1)
    monkeypatch.setattr(
        edm_module,
        "verify_sha256",
        lambda *_args, **_kwargs: pytest.fail("oversized file must be rejected first"),
    )
    monkeypatch.setattr(
        edm_module.torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("oversized file must not be loaded"),
    )

    with pytest.raises(ValueError, match="bundle size"):
        edm_module.EDMRelocMap.load(artifact, expected_sha256="0" * 64)


def test_edm_bundle_rejects_jpeg_dimensions_before_decode() -> None:
    bundle = _edm_bundle()
    bundle["refs"]["ref.jpg"]["image_jpg"] = _jpeg_header(
        width=65535,
        height=65535,
    )

    with pytest.raises(ValueError, match="JPEG dimensions"):
        edm_module._validate_edm_bundle_schema(bundle)


def test_edm_bundle_rejects_global_descriptor_dimension() -> None:
    bundle = _edm_bundle()
    bundle["ref_global"] = np.ones((1, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="ref_global"):
        edm_module._validate_edm_bundle_schema(bundle)


def test_edm_bundle_rejects_total_encoded_image_budget(monkeypatch) -> None:
    bundle = _edm_bundle()
    monkeypatch.setattr(edm_module, "EDM_MAX_TOTAL_ENCODED_IMAGE_BYTES", 1)

    with pytest.raises(ValueError, match="encoded image bytes"):
        edm_module._validate_edm_bundle_schema(bundle)


def test_edm_bundle_rejects_covis_edge_budget(monkeypatch) -> None:
    bundle = _edm_bundle()
    bundle["ref_names"] = ["a.jpg", "b.jpg"]
    bundle["ref_global"] = np.ones(
        (2, edm_module.EDM_GLOBAL_DESCRIPTOR_DIM), dtype=np.float32
    )
    entry = bundle["refs"].pop("ref.jpg")
    bundle["refs"] = {
        "a.jpg": entry,
        "b.jpg": {
            "xyz_by_cell": entry["xyz_by_cell"].copy(),
            "image_jpg": entry["image_jpg"].copy(),
        },
    }
    bundle["covis"] = {"a.jpg": [1], "b.jpg": [0]}
    monkeypatch.setattr(edm_module, "EDM_MAX_COVIS_EDGES", 1)

    with pytest.raises(ValueError, match="covis edge count"):
        edm_module._validate_edm_bundle_schema(bundle)


def test_edm_scales_query_points_on_x_and_y_axes_independently() -> None:
    name = "ref.jpg"
    xyz = np.full((128 * 72, 3), np.nan, dtype=np.float32)
    xyz[1 * 128 + 1] = [1.0, 2.0, 3.0]
    matcher = type(
        "Matcher",
        (),
        {
            "match_many_to_one": lambda _self, _images, _query: [{
                "mkpts0": np.array([[8.0, 8.0]], dtype=np.float32),
                "mkpts1": np.array([[100.0, 200.0]], dtype=np.float32),
                "mconf": np.array([0.9], dtype=np.float32),
            }],
        },
    )()
    reloc_map = type(
        "Map",
        (),
        {"ref_names": [name], "images": {name: np.zeros((576, 1024), np.uint8)},
         "xyz_by_cell": {name: xyz}},
    )()

    localizer = EDMLocalizer(
        reloc_map,
        Camera("PINHOLE", 1280, 800, [900.0, 900.0, 640.0, 400.0]),
        matcher=matcher,
    )

    points2d, _points3d, _confidence, count = localizer.correspondences_for_sources(
        np.zeros((800, 1280), dtype=np.uint8),
        [reloc_map.images[name]],
        [xyz],
    )[0]

    assert count == 1
    np.testing.assert_allclose(points2d, [[125.0, 200.0 * 800.0 / 576.0]])


@pytest.mark.parametrize(
    "meta",
    [
        {"bundle_vpr": "xfeat"},
        {"vpr": "other-vpr"},
        {"bundle_vpr": "megaloc", "vpr": "xfeat"},
    ],
)
def test_xfeat_factory_rejects_incompatible_vpr_metadata(meta) -> None:
    with pytest.raises(ValueError, match="MegaLoc"):
        factory._validate_xfeat_vpr_metadata(meta)


def test_xfeat_factory_accepts_declared_megaloc_family() -> None:
    assert factory._validate_xfeat_vpr_metadata(
        {"bundle_vpr": "megaloc", "vpr": "MegaLoc-8448"}
    ) == "megaloc"
