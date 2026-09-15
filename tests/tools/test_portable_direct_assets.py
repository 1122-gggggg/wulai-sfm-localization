import hashlib
import json
from pathlib import Path

import pytest

import export_simulator_package as exporter


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _site(root: Path):
    site = root / "site"
    (site / "images/view").mkdir(parents=True)
    (site / "depth").mkdir()
    image = site / "images/view/frame.jpg"
    image.write_bytes(b"reference image")
    depth = site / "depth/frame.npz"
    depth.write_bytes(b"reference depth")
    model = site / "model.bin"
    model.write_bytes(b"model")
    frames = site / "keyframes.jsonl"
    frames.write_text(
        json.dumps({"output_name": "view/frame.jpg", "image_sha256": _hash(image)}) + "\n"
    )
    bundle = site / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "schema": "direct-localization-bundle/v1",
                "keyframes_manifest": frames.name,
                "keyframes_images_root": "images",
                "reference_depth_dir": "depth",
                "files": [
                    {"path": p.name, "sha256": _hash(p), "size_bytes": p.stat().st_size}
                    for p in (frames, model)
                ],
            }
        )
    )
    profile = site / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "site_id": "fixture",
                "localizer": "direct",
                "assets": {"localization_bundle": bundle.name},
                "asset_sha256": {"localization_bundle": _hash(bundle)},
            }
        )
    )
    return profile, image, depth, model


def test_portable_direct_bundle_includes_models_reference_images_and_depths(tmp_path):
    root = tmp_path / "source"
    profile, image, depth, model = _site(root)
    destination = tmp_path / "package"
    copied = exporter.copy_site_bundle(root, destination, [profile])
    for path in (image, depth, model):
        relative = path.relative_to(root)
        assert str(relative) in copied
        assert (destination / relative).read_bytes() == path.read_bytes()


@pytest.mark.parametrize("tamper", ["content", "symlink"])
def test_portable_direct_image_integrity_is_fail_closed(tmp_path, tamper):
    root = tmp_path / "source"
    profile, image, _, _ = _site(root)
    if tamper == "content":
        image.write_bytes(b"wrong image")
    else:
        image.unlink()
        outside = tmp_path / "outside.jpg"
        outside.write_bytes(b"reference image")
        image.symlink_to(outside)
    with pytest.raises(exporter.ArtifactResolutionError):
        exporter.collect_site_bundle(root, [profile])
