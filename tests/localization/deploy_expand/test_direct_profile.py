from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

import production_localizer_factory
from direct_profile import DirectOnlineWorkmap, DirectProfileError, load_direct_profile


def _valid_profile() -> dict:
    return {
        "schema": "direct-deployment-profile/v1",
        "name": "workmap-fixture",
        "map_scale": 1.0,
        "intrinsics": {
            "schema_version": 1,
            "image_width": 640,
            "image_height": 480,
            "K": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            "images_are_undistorted": True,
        },
        "reloc": {
            "top_k": 4,
            "lift_distance_px": 3.0,
            "period_s": 0.5,
            "min_points": 8,
            "reference_bank": "fixture",
            "min_reference_occupied_bins": 1,
            "bank_occupancy_percentile": 50,
            "descriptor_batch_size": 4,
            "batch_refs": True,
            "reference_cache_size": 2,
            "frozen_pnp_thresholds": {
                "strong_inliers": 8,
                "minimum_inlier_ratio": 0.2,
                "minimum_hull_coverage": 0.1,
                "minimum_occupancy_4x4": 2,
                "minimum_positive_depth_ratio": 0.9,
                "maximum_reprojection_p90_px": 4.0,
            },
        },
        "fast_loop": {
            "resolution": [640, 480],
            "tracker": "klt",
            "track_cap": 100,
            "reseed_min_points": 20,
            "topup": True,
            "klt": {"fb_max_px": 1.0, "win": 21, "levels": 3},
            "pnp": {
                "max_error_px": 3.0,
                "min_trials": 10,
                "max_trials": 100,
                "confidence": 0.99,
                "refine_iters": 5,
                "min_points": 8,
                "min_inliers": 8,
            },
        },
        "vo": {
            "enabled": True,
            "keyframe_stride": 5,
            "batch_lag": 2,
            "min_live": 8,
            "detect_cap": 100,
            "min_distance": 8,
            "min_parallax_deg": 1.0,
            "max_reproj_px": 2.0,
            "refine": False,
            "window_ba": False,
        },
        "dead_reckon": {"enabled": False, "max_frames": 10},
    }


def _load(tmp_path: Path, raw: dict):
    path = tmp_path / "direct_profile.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_direct_profile(path)


def test_legacy_profile_defaults_to_disabled_online_workmap(tmp_path: Path) -> None:
    profile = _load(tmp_path, _valid_profile())

    assert profile.online_workmap == DirectOnlineWorkmap(enabled=False)
    with pytest.raises(FrozenInstanceError):
        profile.online_workmap.enabled = True


def test_enabled_online_workmap_accepts_only_the_frozen_minimal_shape(tmp_path: Path) -> None:
    raw = _valid_profile()
    raw["online_workmap"] = {"enabled": True}

    profile = _load(tmp_path, raw)

    assert profile.online_workmap.enabled is True
    assert profile.online_workmap.optimizer == "local_ba"


def test_online_workmap_accepts_explicit_incremental_optimizer(tmp_path: Path) -> None:
    raw = _valid_profile()
    raw["online_workmap"] = {"enabled": True, "optimizer": "isam2", "retain_valid_tracks": True}
    profile = _load(tmp_path, raw).online_workmap
    assert profile.optimizer == "isam2"
    assert profile.retain_valid_tracks is True


@pytest.mark.parametrize("optimizer", [None, 1, "unknown", [], {}])
def test_online_workmap_rejects_unknown_optimizer(tmp_path: Path, optimizer) -> None:
    raw = _valid_profile()
    raw["online_workmap"] = {"enabled": True, "optimizer": optimizer}
    with pytest.raises(DirectProfileError, match="optimizer"):
        _load(tmp_path, raw)


@pytest.mark.parametrize(
    "section",
    [
        None,
        [],
        {},
        {"enabled": 1},
        {"enabled": "true"},
        {"enabled": True, "retain_valid_tracks": 1},
        {"enabled": False, "session_dir": "elsewhere"},
    ],
)
def test_online_workmap_rejects_invalid_shapes(tmp_path: Path, section) -> None:
    raw = _valid_profile()
    raw["online_workmap"] = section

    with pytest.raises(DirectProfileError):
        _load(tmp_path, raw)


@pytest.mark.parametrize(
    ("vo_key", "vo_value", "message"),
    [
        ("enabled", False, "requires vo.enabled=true"),
        ("refine", True, "vo.refine=false"),
        ("window_ba", True, "vo.refine=false"),
    ],
)
def test_enabled_online_workmap_requires_vo_mapping_contract(
    tmp_path: Path, vo_key: str, vo_value, message: str
) -> None:
    raw = _valid_profile()
    raw["online_workmap"] = {"enabled": True}
    raw["vo"][vo_key] = vo_value

    with pytest.raises(DirectProfileError, match=message):
        _load(tmp_path, raw)


def _patch_fake_direct_runtime(monkeypatch, tmp_path: Path):
    release = tmp_path / "base-release"
    release.mkdir()
    cache = tmp_path / "cache"
    assets = SimpleNamespace(root=release)
    profile = SimpleNamespace(
        name="fixture",
        reloc=SimpleNamespace(top_k=4),
        online_workmap=SimpleNamespace(enabled=True),
    )

    class FakeAssets:
        @staticmethod
        def load(*_args, **_kwargs):
            return assets

    monkeypatch.setattr(
        production_localizer_factory,
        "_import_direct_runtime",
        lambda: (
            FakeAssets,
            lambda *_args, **_kwargs: profile,
            lambda *_args, **_kwargs: object(),
            None,
            lambda: SimpleNamespace(
                edm_repo=Path("/edm"),
                edm_checkpoint=Path("/edm/checkpoint"),
                boq_weights=Path("/boq"),
            ),
            lambda: cache,
        ),
    )
    monkeypatch.setattr(production_localizer_factory, "expected_sha256", lambda *_args: None)
    monkeypatch.setattr(production_localizer_factory, "Camera", lambda *args: args)
    return release


def test_factory_rejects_session_directory_inside_frozen_release(
    tmp_path: Path, monkeypatch
) -> None:
    release = _patch_fake_direct_runtime(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="outside the base release"):
        production_localizer_factory._build_direct_localizer(
            bundle=tmp_path / "bundle.json",
            frame_source=lambda: None,
            camera_tuple=("PINHOLE", 640, 480, [500.0, 500.0, 320.0, 240.0]),
            production_profile=tmp_path / "profile.json",
            workmap_session_dir=release / "workmap-session",
            workmap_eval_stage="baseline",
        )


def test_factory_requires_trusted_profile_sha_for_full_workmap(tmp_path: Path, monkeypatch) -> None:
    _patch_fake_direct_runtime(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="trusted SHA-256"):
        production_localizer_factory._build_direct_localizer(
            bundle=tmp_path / "bundle.json",
            frame_source=lambda: None,
            camera_tuple=("PINHOLE", 640, 480, [500.0, 500.0, 320.0, 240.0]),
            production_profile=tmp_path / "profile.json",
            workmap_eval_stage="full",
        )


def test_adapter_baseline_keeps_the_frozen_tracker_for_enabled_profile(
    tmp_path: Path, monkeypatch
) -> None:
    import direct_localizer_adapter

    class FakeTracker:
        def __init__(self, *_args):
            self.map_frame = None

    monkeypatch.setenv("SFM_EVALUATION_ONLY", "1")
    monkeypatch.setattr(direct_localizer_adapter, "TwoRateTracker", FakeTracker)
    profile = SimpleNamespace(online_workmap=SimpleNamespace(enabled=True))

    tracker = direct_localizer_adapter.DirectTrackerAdapter(
        object(),
        profile,
        object(),
        workmap_session_dir=tmp_path / "session",
        workmap_eval_stage="baseline",
    )

    assert isinstance(tracker.trk, FakeTracker)
    assert tracker.workmap_eval_stage == "baseline"
    assert tracker.workmap_session_id is None


def test_adapter_rejects_nonfull_stage_outside_evaluation_mode(tmp_path: Path, monkeypatch) -> None:
    import direct_localizer_adapter

    monkeypatch.delenv("SFM_EVALUATION_ONLY", raising=False)
    profile = SimpleNamespace(online_workmap=SimpleNamespace(enabled=True))

    with pytest.raises(RuntimeError, match="SFM_EVALUATION_ONLY=1"):
        direct_localizer_adapter.DirectTrackerAdapter(
            object(),
            profile,
            object(),
            workmap_session_dir=tmp_path / "session",
            workmap_eval_stage="ba",
        )


def test_adapter_full_stage_requires_enabled_profile(tmp_path: Path) -> None:
    import direct_localizer_adapter

    profile = SimpleNamespace(online_workmap=SimpleNamespace(enabled=False))

    with pytest.raises(ValueError, match="requires online_workmap.enabled=true"):
        direct_localizer_adapter.DirectTrackerAdapter(
            object(),
            profile,
            object(),
            workmap_session_dir=tmp_path / "session",
            workmap_eval_stage="full",
        )


def test_session_ids_are_stable_and_distinct_for_repeat_directories(
    tmp_path: Path,
) -> None:
    from direct_localizer_adapter import _workmap_session_id_for_dir

    first = tmp_path / "mode" / "sequence" / "repeat-1" / "session"
    second = tmp_path / "mode" / "sequence" / "repeat-2" / "session"

    first_id = _workmap_session_id_for_dir(first)

    assert first_id == _workmap_session_id_for_dir(first)
    assert first_id != _workmap_session_id_for_dir(second)


def test_workmap_expansion_defaults_and_enabled_flags(tmp_path: Path) -> None:
    raw = _valid_profile()
    defaults = _load(tmp_path, raw).online_workmap.expansion
    assert not any(
        (defaults.frozen_region_gate, defaults.spatiotemporal_query, defaults.depth_seeds)
    )
    raw["online_workmap"] = {
        "enabled": True,
        "expansion": dict.fromkeys(
            ("frozen_region_gate", "spatiotemporal_query", "depth_seeds"), True
        ),
    }
    flags = _load(tmp_path, raw).online_workmap.expansion
    assert flags.frozen_region_gate and flags.spatiotemporal_query and flags.depth_seeds


@pytest.mark.parametrize(
    ("enabled", "changes", "remove"),
    [
        (True, {}, "depth_seeds"),
        (True, {"unknown": False}, None),
        (True, {"depth_seeds": 1}, None),
        (False, {}, None),
    ],
)
def test_workmap_expansion_rejects_invalid_flags(tmp_path, enabled, changes, remove) -> None:
    flags = dict.fromkeys(("frozen_region_gate", "spatiotemporal_query", "depth_seeds"), True)
    flags.update(changes)
    if remove:
        flags.pop(remove)
    raw = _valid_profile()
    raw["online_workmap"] = {"enabled": enabled, "expansion": flags}
    with pytest.raises(DirectProfileError):
        _load(tmp_path, raw)


def test_submap_graph_solver_defaults_to_batch_and_accepts_isam2(tmp_path: Path) -> None:
    raw = _valid_profile()
    raw["online_workmap"] = {"enabled": True}
    assert _load(tmp_path, raw).online_workmap.submap_graph_solver == "batch"
    raw["online_workmap"] = {"enabled": True, "submap_graph_solver": "isam2"}
    assert _load(tmp_path, raw).online_workmap.submap_graph_solver == "isam2"


@pytest.mark.parametrize("solver", [None, 1, "gtsam", "", [], {}])
def test_submap_graph_solver_rejects_other_values(tmp_path: Path, solver) -> None:
    raw = _valid_profile()
    raw["online_workmap"] = {"enabled": True, "submap_graph_solver": solver}
    with pytest.raises(DirectProfileError, match="submap_graph_solver"):
        _load(tmp_path, raw)
