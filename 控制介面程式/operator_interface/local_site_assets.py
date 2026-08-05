"""Local-filesystem implementations of the replaceable site asset ports."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable

from site_asset_interfaces import (
    AssetCheck,
    ImportedAsset,
    ImportedSite,
    ValidatedSitePackage,
)

_CONTROL_ROOT = Path(__file__).resolve().parents[1]
if str(_CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(_CONTROL_ROOT))

from site_profile import QueryCamera, SiteProfile, load_site_profile


_REQUIRED_ASSETS = (
    ("map_ply", "點雲地圖"),
    ("localization_bundle", "EDM 定位 bundle"),
    ("localizer_profile", "EDM runtime profile"),
    ("map_reference_poses", "參考影像位姿"),
)


def _json(path: Path) -> dict:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {value}")
            ),
        )
    except OSError as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return raw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _is_inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
    except ValueError:
        return False
    return True


def _validate_ply(path: Path) -> None:
    try:
        with path.open("rb") as stream:
            header = stream.read(64 * 1024)
    except OSError as exc:
        raise ValueError(f"cannot read PLY {path}: {exc}") from exc
    end = header.find(b"end_header")
    if not header.startswith(b"ply\n") or end < 0:
        raise ValueError(f"invalid PLY header: {path}")
    try:
        lines = header[: end + len(b"end_header")].decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"PLY header must be ASCII: {path}") from exc
    formats = {"format ascii 1.0", "format binary_little_endian 1.0"}
    if not any(line.strip() in formats for line in lines):
        raise ValueError(f"PLY must be ASCII or binary little-endian 1.0: {path}")
    vertex_lines = [line.split() for line in lines if line.startswith("element vertex ")]
    if len(vertex_lines) != 1 or int(vertex_lines[0][2]) <= 0:
        raise ValueError(f"PLY must contain at least one vertex: {path}")
    properties = {
        line.split()[-1]
        for line in lines
        if line.startswith("property ") and len(line.split()) >= 3
    }
    missing = sorted({"x", "y", "z", "red", "green", "blue"} - properties)
    if missing:
        raise ValueError(f"PLY is missing UI vertex properties {missing}: {path}")


def _camera_dict(camera: QueryCamera) -> dict:
    return {
        "model": camera.model,
        "width": camera.width,
        "height": camera.height,
        "params": list(camera.params),
    }


def _camera_tuple(raw: object, label: str) -> tuple[str, int, int, tuple[float, ...]]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} camera must be an object")
    try:
        model = str(raw["model"]).strip().upper()
        width = raw["width"]
        height = raw["height"]
        params = raw["params"]
    except KeyError as exc:
        raise ValueError(f"{label} camera is missing {exc.args[0]}") from exc
    if (
        not model
        or isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
        or not isinstance(params, list)
        or not params
    ):
        raise ValueError(f"invalid {label} camera")
    try:
        values = tuple(float(value) for value in params)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label} camera params") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid {label} camera params")
    return model, width, height, values


def _validate_reference_poses(
    path: Path,
    query_camera: QueryCamera,
    bundle_names: Iterable[str],
) -> int:
    raw = _json(path)
    if _camera_tuple(raw.get("camera"), "reference poses") != (
        query_camera.model,
        query_camera.width,
        query_camera.height,
        query_camera.params,
    ):
        raise ValueError("reference poses camera does not match site query_camera")
    poses = raw.get("poses")
    if not isinstance(poses, dict) or not poses:
        raise ValueError("reference poses must contain a non-empty poses object")
    names = tuple(bundle_names)
    if set(poses) != set(names):
        raise ValueError("reference image names do not exactly match the EDM bundle")
    for name, pose in poses.items():
        if not isinstance(pose, dict):
            raise ValueError(f"invalid reference pose: {name}")
        rotation = pose.get("R")
        translation = pose.get("t")
        values = []
        if (
            not isinstance(rotation, list)
            or len(rotation) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in rotation)
            or not isinstance(translation, list)
            or len(translation) != 3
        ):
            raise ValueError(f"invalid R/t shape in reference pose: {name}")
        for row in rotation:
            values.extend(row)
        values.extend(translation)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError(f"non-finite R/t in reference pose: {name}")
    return len(poses)


def inspect_edm_bundle(path: Path) -> tuple[str, ...]:
    """Use the production EDM loader so import and runtime accept the same artifact."""
    deploy = (
        Path(__file__).resolve().parents[2]
        / "定位演算法"
        / "deploy_code"
        / "sfm_glomap_deploy"
    )
    if str(deploy) not in sys.path:
        sys.path.insert(0, str(deploy))
    try:
        from reloc_localizer_edm import EDMRelocMap

        bundle = EDMRelocMap.load(path)
    except Exception as exc:
        raise ValueError(f"invalid EDM localization bundle {path}: {exc}") from exc
    return tuple(bundle.ref_names)


def load_edm_runtime_profile(path: Path) -> dict:
    deploy = (
        Path(__file__).resolve().parents[2]
        / "定位演算法"
        / "deploy_code"
        / "sfm_glomap_deploy"
    )
    if str(deploy) not in sys.path:
        sys.path.insert(0, str(deploy))
    try:
        from edm_profile import load_edm_production_profile

        return load_edm_production_profile(path)
    except Exception as exc:
        raise ValueError(f"invalid EDM runtime profile {path}: {exc}") from exc


class LocalSitePackageProvider:
    def __init__(
        self,
        managed_root: str | Path,
        *,
        bundle_inspector: Callable[[Path], tuple[str, ...]] = inspect_edm_bundle,
        edm_profile_loader: Callable[[Path], dict] = load_edm_runtime_profile,
    ):
        self.managed_root = Path(managed_root).expanduser().resolve()
        self.bundle_inspector = bundle_inspector
        self.edm_profile_loader = edm_profile_loader

    def validate_folder(self, folder: str | Path) -> ValidatedSitePackage:
        root = Path(folder).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"selected site package is not a folder: {root}")
        manifest = root / "site_profile.json"
        if not manifest.is_file():
            raise ValueError("selected folder must contain site_profile.json")
        profile = load_site_profile(manifest)
        if profile.schema_version != 2:
            raise ValueError("imported site package must use schema_version 2")
        if profile.coordinate_frame is None:
            raise ValueError("site_profile.json must declare coordinate_frame")
        if profile.query_camera is None:
            raise ValueError("site_profile.json must declare query_camera")
        separate_assets = {
            "route_json": profile.route_json,
            "poles_json": profile.poles_json,
            "megaloc_cache": profile.megaloc_cache,
            "track_landmarks": profile.track_landmarks,
            "localizer_deploy_dir": profile.localizer_deploy_dir,
        }
        configured_separately = sorted(
            key for key, value in separate_assets.items() if value is not None
        )
        if configured_separately:
            raise ValueError(
                "base site package must not embed separately managed assets: "
                + ", ".join(configured_separately)
            )
        checks = []
        for key, label in _REQUIRED_ASSETS:
            asset = getattr(profile, key)
            expected = getattr(profile.asset_sha256, key)
            if asset is None:
                raise ValueError(f"site package is missing required {key}")
            if not _is_inside(asset, root):
                raise ValueError(f"{key} must stay inside the selected folder")
            if expected is None:
                raise ValueError(f"site_profile.json is missing asset_sha256.{key}")
            actual = _sha256(asset)
            if actual != expected:
                raise ValueError(
                    f"{key} SHA-256 mismatch: expected {expected}, got {actual}"
                )
            checks.append(AssetCheck(key, label, asset, actual))
        _validate_ply(profile.map_ply)
        self.edm_profile_loader(profile.localizer_profile)
        names = self.bundle_inspector(profile.localization_bundle)
        reference_count = _validate_reference_poses(
            profile.map_reference_poses, profile.query_camera, names
        )
        return ValidatedSitePackage(
            folder=root,
            site_id=profile.site_id,
            display_name=profile.display_name,
            coordinate_frame_id=profile.coordinate_frame.id,
            reference_count=reference_count,
            checks=tuple(checks),
        )

    def import_folder(self, folder: str | Path) -> ImportedSite:
        report = self.validate_folder(folder)
        source_profile = load_site_profile(report.folder / "site_profile.json")
        destination = self.managed_root / report.site_id
        existing = destination / "site_profile.json"
        if existing.is_file():
            loaded = load_site_profile(existing)
            expected = {
                check.key: check.sha256 for check in report.checks
            }
            actual = {
                key: getattr(loaded.asset_sha256, key) for key in expected
            }
            contents = {
                key: _sha256(getattr(loaded, key)) for key in expected
            }
            if actual != expected or contents != expected:
                raise ValueError(
                    f"managed site {report.site_id!r} already exists with different assets"
                )
            return ImportedSite(existing, report.site_id, already_present=True)
        self.managed_root.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{report.site_id}-", dir=self.managed_root)
        )
        try:
            copies = {
                "map_ply": (source_profile.map_ply, Path("map/map.ply")),
                "localization_bundle": (
                    source_profile.localization_bundle,
                    Path("localization/localization_bundle.pt"),
                ),
                "localizer_profile": (
                    source_profile.localizer_profile,
                    Path("localization/edm_runtime_profile.json"),
                ),
                "map_reference_poses": (
                    source_profile.map_reference_poses,
                    Path("localization/reference_poses.json"),
                ),
            }
            for source, relative in copies.values():
                assert source is not None
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            for key, (_source, relative) in copies.items():
                expected = getattr(source_profile.asset_sha256, key)
                actual = _sha256(staging / relative)
                if actual != expected:
                    raise ValueError(
                        f"{key} changed while it was being imported; retry the package"
                    )
            coordinate = source_profile.coordinate_frame
            camera = source_profile.query_camera
            assert coordinate is not None and camera is not None
            profile_json = {
                "schema_version": 2,
                "site_id": report.site_id,
                "display_name": report.display_name,
                "localizer": "edm",
                "localizer_deploy_dir": None,
                "localizer_profile": "localization/edm_runtime_profile.json",
                "map_reference_poses": "localization/reference_poses.json",
                "query_camera": _camera_dict(camera),
                "coordinate_frame": {
                    "id": coordinate.id,
                    "convention": coordinate.convention,
                    "horizontal_axes": list(coordinate.horizontal_axes),
                    "up_axis": coordinate.up_axis,
                    "handedness": coordinate.handedness,
                    "units": coordinate.units,
                },
                "asset_sha256": {
                    key: getattr(source_profile.asset_sha256, key)
                    for key, _label in _REQUIRED_ASSETS
                },
                "hardware_approval": None,
                "flight": {
                    "approved": False,
                    "coordinate_frame_id": coordinate.id,
                    "route_clearance_approved": False,
                    "approval_note": "Imported for ground localization; flight approval required.",
                    "controller": None,
                },
                "assets": {
                    "map_ply": "map/map.ply",
                    "route_json": None,
                    "localization_bundle": "localization/localization_bundle.pt",
                    "megaloc_cache": None,
                    "track_landmarks": None,
                    "poles_json": None,
                },
            }
            (staging / "site_profile.json").write_text(
                json.dumps(profile_json, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            load_site_profile(staging / "site_profile.json")
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return ImportedSite(destination / "site_profile.json", report.site_id)


class _LocalSupplementProvider:
    profile_key = ""
    digest_key = ""
    relative_path = Path()

    def __init__(self, managed_root: str | Path):
        self.managed_root = Path(managed_root).expanduser().resolve()

    def _load_managed_profile(self, profile_path: str | Path) -> tuple[Path, SiteProfile, dict]:
        path = Path(profile_path).expanduser().resolve()
        if not _is_inside(path, self.managed_root):
            raise ValueError("import a site package before adding route or target files")
        profile = load_site_profile(path)
        if path != self.managed_root / profile.site_id / "site_profile.json":
            raise ValueError("route and target imports require a managed site profile")
        return path, profile, _json(path)

    def _commit(
        self, source: Path, profile_path: Path, profile: SiteProfile, raw: dict
    ) -> ImportedAsset:
        target = profile_path.parent / self.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_asset = target.with_name(f".{target.name}.tmp")
        shutil.copy2(source, temp_asset)
        os.replace(temp_asset, target)
        raw["assets"][self.profile_key] = self.relative_path.as_posix()
        raw.setdefault("asset_sha256", {})[self.digest_key] = _sha256(target)
        flight = raw.setdefault("flight", {})
        flight["approved"] = False
        flight["route_clearance_approved"] = False
        flight["approval_note"] = "Imported asset changed; flight re-approval required."
        temp_profile = profile_path.with_name(".site_profile.json.tmp")
        temp_profile.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        load_site_profile(temp_profile)
        os.replace(temp_profile, profile_path)
        return ImportedAsset(profile_path, target)


def _finite_vec3(raw: object, label: str) -> tuple[float, float, float]:
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{label} must be a three-number list")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in raw
    ):
        raise ValueError(f"{label} must contain finite numbers")
    return tuple(float(value) for value in raw)


class LocalRouteProvider(_LocalSupplementProvider):
    profile_key = "route_json"
    digest_key = "route_json"
    relative_path = Path("routes/flight_route.json")

    def import_file(self, source: str | Path, profile_path: str | Path) -> ImportedAsset:
        path, profile, profile_raw = self._load_managed_profile(profile_path)
        route_path = Path(source).expanduser().resolve()
        route = _json(route_path)
        frame_id = profile.coordinate_frame.id if profile.coordinate_frame else None
        expected = {
            "schema": "sfm-flight-route/v1",
            "site_id": profile.site_id,
            "coordinate_frame_id": frame_id,
            "units": "map",
            "purpose": "flight",
            "closed": False,
        }
        for key, value in expected.items():
            if route.get(key) != value:
                raise ValueError(f"route {key} must be {value!r}")
        if route.get("frame") not in {"glomap", "aligned"}:
            raise ValueError("route frame must be 'glomap' or 'aligned'")
        waypoints = route.get("waypoints")
        if not isinstance(waypoints, list) or len(waypoints) < 2:
            raise ValueError("route needs at least two waypoints")
        parsed = [_finite_vec3(value, f"waypoint[{index}]") for index, value in enumerate(waypoints)]
        if all(a == b for a, b in zip(parsed, parsed[1:])):
            raise ValueError("route must contain a non-zero segment")
        return self._commit(route_path, path, profile, profile_raw)


class LocalTargetProvider(_LocalSupplementProvider):
    profile_key = "poles_json"
    digest_key = "poles_json"
    relative_path = Path("targets/inspection_targets.json")

    def import_file(self, source: str | Path, profile_path: str | Path) -> ImportedAsset:
        path, profile, profile_raw = self._load_managed_profile(profile_path)
        target_path = Path(source).expanduser().resolve()
        targets = _json(target_path)
        frame_id = profile.coordinate_frame.id if profile.coordinate_frame else None
        expected = {
            "schema": "sfm-inspection-targets/v1",
            "site_id": profile.site_id,
            "coordinate_frame_id": frame_id,
            "frame": "aligned",
            "units": "map",
        }
        for key, value in expected.items():
            if targets.get(key) != value:
                raise ValueError(f"inspection targets {key} must be {value!r}")
        poles = targets.get("poles")
        if not isinstance(poles, list) or not poles:
            raise ValueError("inspection targets must contain at least one pole")
        for index, pole in enumerate(poles):
            if not isinstance(pole, dict):
                raise ValueError(f"pole[{index}] must be an object")
            if "center" in pole:
                _finite_vec3(pole["center"], f"pole[{index}].center")
            else:
                _finite_vec3(pole.get("base"), f"pole[{index}].base")
                _finite_vec3(pole.get("top"), f"pole[{index}].top")
        return self._commit(target_path, path, profile, profile_raw)
