"""Local-filesystem implementations of the replaceable site asset ports."""
from __future__ import annotations

import hashlib
import json
import math
import os
import fcntl
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from collections.abc import Callable, Iterable
from pathlib import Path

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
    # Dropping this on import made every managed site fall back to the legacy
    # "GLOMAP -Y is up" guess, which is 22.51 deg wrong on target_site_v1.
    ("map_align", "重力對齊"),
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


def discover_ply_files(folder: str | Path) -> list[Path]:
    """Find regular PLY files below a selected folder in stable relative-path order."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"selected map folder is not a directory: {root}")
    found = [
        path.resolve()
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() == ".ply"
    ]
    return sorted(found, key=lambda path: path.relative_to(root).as_posix().casefold())


#: Filename patterns for the assets a site package is expected to carry. Order is
#: the order the operator sees them in. Discovery is a CONVENIENCE: it proposes
#: candidates, it never decides -- the caller confirms before anything is hashed
#: into a site_profile.json.
SITE_ASSET_PATTERNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("map_ply", "地圖點雲", ("*.ply",)),
    ("map_reference_poses", "參考影格姿態", ("*ref_poses*.json",)),
    ("localization_bundle", "定位 bundle", ("*reloc_map*.pt", "*bundle*.pt")),
    ("map_align", "重力對齊", ("T_align_gravity.json",)),
    ("route_json", "航線", ("flight_path.json", "*route*.json")),
)

#: PLY files the route editor itself writes. They live beside the map but are
#: previews of a route, never the map, so proposing one would be a trap.
_ROUTE_PREVIEW_PLY = ("flight_path.ply", "route.ply", "poles.ply", "wires.ply")


@contextmanager
def _site_asset_lock(path: Path):
    """Serialize commits for one site across threads and processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _site_asset_lock_path(root: Path, site_id: str) -> Path:
    return root / f".{site_id}.asset.lock"


def _new_temp_path(parent: Path, prefix: str, suffix: str = ".tmp") -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=parent)
    os.close(descriptor)
    return Path(name)


def _remove_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        # Cleanup must not hide the transaction's original failure.
        pass


class DiscoveredAsset:
    """One asset slot: what was proposed, and everything else that matched."""

    __slots__ = ("key", "label", "path", "candidates")

    def __init__(self, key: str, label: str, candidates: list[Path]):
        self.key = key
        self.label = label
        self.candidates = tuple(candidates)
        self.path = candidates[0] if len(candidates) == 1 else None

    @property
    def found(self) -> bool:
        return bool(self.candidates)

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1

    def __repr__(self) -> str:
        return f"DiscoveredAsset({self.key!r}, n={len(self.candidates)})"


class DiscoveredSitePackage:
    """Result of scanning a folder the operator picked."""

    __slots__ = ("folder", "assets")

    def __init__(self, folder: Path, assets: list[DiscoveredAsset]):
        self.folder = folder
        self.assets = tuple(assets)

    def asset(self, key: str) -> DiscoveredAsset | None:
        for item in self.assets:
            if item.key == key:
                return item
        return None

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(item.key for item in self.assets if not item.found)

    @property
    def ambiguous(self) -> tuple[str, ...]:
        return tuple(item.key for item in self.assets if item.ambiguous)

    @property
    def has_route(self) -> bool:
        route = self.asset("route_json")
        return bool(route is not None and route.found)

    def summary_lines(self) -> list[str]:
        lines = []
        for item in self.assets:
            if not item.found:
                lines.append(f"✗ {item.label}：未找到")
            elif item.ambiguous:
                lines.append(f"? {item.label}：{len(item.candidates)} 個候選，需選擇")
            else:
                lines.append(f"✓ {item.label}：{item.path.name}")
        return lines


class AvailableSitePackage:
    """A folder under the sites root that already carries a site_profile.json."""

    __slots__ = ("folder", "site_id", "display_name", "error")

    def __init__(self, folder: Path, site_id: str, display_name: str,
                 error: str | None = None):
        self.folder = folder
        self.site_id = site_id
        self.display_name = display_name
        self.error = error

    @property
    def label(self) -> str:
        base = f"{self.display_name} ({self.site_id})" if self.site_id else self.folder.name
        return f"{base} ⚠ {self.error}" if self.error else base

    def __repr__(self) -> str:
        return f"AvailableSitePackage({self.site_id!r}, error={self.error!r})"


def list_site_packages(root: str | Path) -> list[AvailableSitePackage]:
    """Every importable site folder directly under `root`, in display order.

    Unreadable manifests are LISTED with their error rather than hidden: a site the
    operator expects to see and cannot must say why, not silently disappear.
    """
    base = Path(root).expanduser()
    if not base.is_dir():
        return []
    found: list[AvailableSitePackage] = []
    #: site_id -> index in `found`. Importing copies a package to
    #: managed_root/<site_id>, and managed_root IS this directory, so the source
    #: folder and its imported copy are both here and describe the same field.
    #: Listing both offered the operator two buttons that do the same thing, and
    #: one of them (the copy) cannot be imported at all.
    by_site_id: dict[str, int] = {}
    for folder in sorted(base.iterdir(), key=lambda p: p.name.casefold()):
        manifest = folder / "site_profile.json"
        if not folder.is_dir() or not manifest.is_file():
            continue
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            site_id = str(raw.get("site_id") or "")
            display = str(raw.get("display_name") or folder.name)
            candidate = AvailableSitePackage(folder.resolve(), site_id, display)
            existing = by_site_id.get(site_id) if site_id else None
            if existing is None:
                if site_id:
                    by_site_id[site_id] = len(found)
                found.append(candidate)
            elif (found[existing].folder.name != site_id
                    and folder.name != site_id):
                # Neither is the managed copy: two independent packages claim the
                # same site_id. Dropping one silently contradicts this function's
                # own promise that nothing disappears without saying why.
                found[existing] = AvailableSitePackage(
                    found[existing].folder, site_id, display,
                    error=f"site_id 與 {folder.name} 重複，請確認要用哪一份",
                )
            elif found[existing].folder.name == site_id and folder.name != site_id:
                # Keep the SOURCE package, not the imported copy at
                # managed_root/<site_id>. The copy re-points its assets at
                # localization/localization_bundle.pt, which carries no trusted
                # digest, so validate_folder refuses it -- listing it gave the
                # operator a button that could not import.
                found[existing] = candidate
        except Exception as exc:
            found.append(AvailableSitePackage(
                folder.resolve(), "", folder.name, error=f"讀取失敗：{exc}"))
    return found


def list_site_routes(folder: str | Path) -> list[Path]:
    """Every route JSON under a site folder, newest first.

    Drafts land in route_drafts/ with a timestamped name and authored routes in
    routes/, so a field accumulates several. Which one the aircraft flies is the
    operator's choice, not the first one found.
    """
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        return []
    found = [
        path.resolve()
        for path in root.rglob("*.json")
        if path.is_file()
        and not path.is_symlink()
        and (
            "route" in path.name.casefold()
            or "flight_path" in path.name.casefold()
            or "route_drafts" in path.parts
            or "routes" in path.parts
        )
        and path.name != "site_profile.json"
    ]
    return sorted(found, key=lambda path: (-path.stat().st_mtime, path.name))


class AvailableSiteRoute:
    """A route file under a site folder, described by what is actually in it."""

    __slots__ = ("path", "waypoints", "schema", "purpose", "align_source")

    def __init__(self, path: Path, waypoints: int, schema: str,
                 purpose: str, align_source: str):
        self.path = path
        self.waypoints = waypoints
        self.schema = schema
        self.purpose = purpose
        self.align_source = align_source

    @property
    def flight_ready(self) -> bool:
        """Whether import_file's exact-equality keys can possibly be satisfied."""
        return self.schema == "sfm-flight-route/v1" and self.purpose == "flight"

    @property
    def label(self) -> str:
        marks = []
        if not self.flight_ready:
            marks.append("僅供顯示")
        if self.align_source == "legacy":
            # Authored against the assumed -Y up rather than the measured one, so
            # every waypoint carries the site's tilt. Worth saying on the button.
            marks.append("舊對齊")
        suffix = f"（{'、'.join(marks)}）" if marks else ""
        return f"{self.path.name} · {self.waypoints} 點{suffix}"


def describe_site_routes(folder: str | Path) -> list[AvailableSiteRoute]:
    """list_site_routes narrowed to files that really are routes, newest first.

    list_site_routes matches on path shape alone, so a site's routes/ directory
    also yields things like T_align.json and wires.json. Offering those as routes
    puts a button in front of the operator that can only ever fail validation.
    """
    described: list[AvailableSiteRoute] = []
    for path in list_site_routes(folder):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        waypoints = raw.get("waypoints")
        if not isinstance(waypoints, list) or len(waypoints) < 2:
            continue
        described.append(
            AvailableSiteRoute(
                path,
                len(waypoints),
                str(raw.get("schema") or ""),
                str(raw.get("purpose") or ""),
                str(raw.get("align_source") or ""),
            )
        )
    return described


def discover_site_package(folder: str | Path) -> DiscoveredSitePackage:
    """Propose which file fills each asset slot for a folder the operator picked.

    Deliberately does NOT write a site_profile.json or hash anything: those pin
    what the aircraft will localize against, so they stay a confirmed action.
    """
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"selected site folder is not a directory: {root}")
    files = [
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    ]
    claimed: set[Path] = set()
    assets: list[DiscoveredAsset] = []
    for key, label, patterns in SITE_ASSET_PATTERNS:
        matched: list[Path] = []
        for path in files:
            if path in claimed:
                continue
            name = path.name.casefold()
            if key == "map_ply" and name in _ROUTE_PREVIEW_PLY:
                continue
            if any(path.match(pattern) for pattern in patterns):
                matched.append(path)
        matched.sort(key=lambda path: path.relative_to(root).as_posix().casefold())
        # A file fills at most one slot, so a later, looser pattern cannot re-claim
        # something an earlier, more specific one already took.
        claimed.update(matched)
        assets.append(DiscoveredAsset(key, label, matched))
    return DiscoveredSitePackage(root, assets)


def match_managed_site_profile_for_map(
    source: str | Path, managed_root: str | Path
) -> SiteProfile | None:
    """Return the one imported site whose declared map digest matches this PLY."""
    map_path = Path(source).expanduser().resolve()
    if map_path.suffix.lower() != ".ply" or not map_path.is_file():
        raise ValueError(f"selected map must be an existing .ply file: {map_path}")
    _validate_ply(map_path)
    digest = _sha256(map_path)
    root = Path(managed_root).expanduser().resolve()
    matches: list[SiteProfile] = []
    for profile_path in sorted(root.glob("*/site_profile.json")):
        try:
            profile = load_site_profile(profile_path)
        except ValueError:
            continue
        if (
            profile.coordinate_frame is not None
            and profile.asset_sha256.map_ply == digest
        ):
            matches.append(profile)
    if len(matches) > 1:
        names = ", ".join(profile.site_id for profile in matches)
        raise ValueError(f"PLY 同時匹配多個已匯入場域，無法自動綁定：{names}")
    return matches[0] if matches else None


def site_pack_root_for_profile(
    profile_path: str | Path | None, packages_root: str | Path
) -> Path | None:
    """The site folder a profile's assets live in, or None when it is elsewhere.

    A system profile under site_profiles/ reaches its pack by relative path, so
    its own parent is site_profiles/ and not the site. The map_ply is what
    actually locates the pack -- deriving from the profile's own path is what put
    editor drafts and the route picker in a directory holding no routes.
    """
    if profile_path is None:
        return None
    root = Path(packages_root).expanduser().resolve()
    try:
        profile = load_site_profile(profile_path)
        relative = profile.map_ply.resolve().relative_to(root)
        site_root = root / relative.parts[0]
        managed_path = (site_root / "site_profile.json").resolve()
        if Path(profile.source).resolve() == managed_path:
            return site_root
        managed = load_site_profile(managed_path)
        if (
            profile.site_id != managed.site_id
            or profile.coordinate_frame != managed.coordinate_frame
            or profile.asset_sha256.map_ply != managed.asset_sha256.map_ply
        ):
            return None
    except (IndexError, OSError, ValueError):
        return None
    return site_root


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


#: Tolerance on the scaled intrinsics, in pixels of the QUERY image. Maps are
#: built from high-resolution stills and flown from a 720p stream, so the two
#: cameras are the same lens at different resolutions -- but only if every
#: parameter scales by the single ratio the resolutions imply. This is an
#: equivalence test, not a relaxation: a genuinely different lens fails it.
_CAMERA_SCALE_TOLERANCE_PX = 0.05

#: How many LEADING params of each COLMAP model are measured in PIXELS: the focal
#: length(s) followed by the principal point. Every parameter after them is a
#: dimensionless distortion coefficient.
_CAMERA_PIXEL_PARAM_COUNT = {
    "SIMPLE_PINHOLE": 3,            # f, cx, cy
    "PINHOLE": 4,                   # fx, fy, cx, cy
    "SIMPLE_RADIAL": 3,             # f, cx, cy, k
    "RADIAL": 3,                    # f, cx, cy, k1, k2
    "SIMPLE_RADIAL_FISHEYE": 3,     # f, cx, cy, k
    "RADIAL_FISHEYE": 3,            # f, cx, cy, k1, k2
    "FOV": 4,                       # fx, fy, cx, cy, omega
    "OPENCV": 4,                    # fx, fy, cx, cy, k1, k2, p1, p2
    "OPENCV_FISHEYE": 4,            # fx, fy, cx, cy, k1..k4
    "FULL_OPENCV": 4,               # fx, fy, cx, cy, k1, k2, p1, p2, k3..k6
    "THIN_PRISM_FISHEYE": 4,        # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
}

#: Distortion coefficients do NOT scale with resolution, and a PIXEL tolerance
#: says nothing about them either way: 0.05 is enormous in k1 units, so a real
#: distortion mismatch passed, while multiplying a genuine k1=0.2 by a 3840->1280
#: ratio moved it 0.133 and REJECTED a same-lens package. They come from one
#: calibration record, so compare them unscaled to a relative tolerance that only
#: absorbs formatting/rounding.
_CAMERA_DISTORTION_RTOL = 1e-3
_CAMERA_DISTORTION_ATOL = 1e-6


def _camera_comparison_scale(
    reference: tuple[str, int, int, tuple[float, ...]],
    query: tuple[str, int, int, tuple[float, ...]],
) -> tuple[float, int, bool] | None:
    ref_model, ref_w, ref_h, ref_params = reference
    _, q_w, q_h, _ = query
    if min(ref_w, ref_h, q_w, q_h) <= 0:
        return None
    same_resolution = (ref_w, ref_h) == (q_w, q_h)
    pixel_count = _CAMERA_PIXEL_PARAM_COUNT.get(ref_model)
    unknown_model = pixel_count is None or pixel_count > len(ref_params)
    if unknown_model:
        # Unknown model: which params carry pixels is unknown, so no scaling can
        # be justified. Equal resolutions are still comparable term by term -- but
        # under the STRICTER of the two tolerances for every parameter, because
        # applying the 0.05 PIXEL tolerance to an unidentified coefficient is the
        # exact mistake this function was fixed to stop making.
        if not same_resolution:
            return None
        pixel_count = 0
    if same_resolution:
        return 1.0, pixel_count, unknown_model
    scale_x = q_w / ref_w
    scale_y = q_h / ref_h
    # A different aspect ratio means the image was cropped, not scaled, and the
    # principal point no longer maps by a single factor.
    if abs(scale_x - scale_y) > 1e-6:
        return None
    return scale_x, pixel_count, unknown_model


def _camera_parameters_match(
    reference: tuple[str, int, int, tuple[float, ...]],
    query: tuple[str, int, int, tuple[float, ...]],
    scale: float,
    pixel_count: int,
    unknown_model: bool,
) -> bool:
    ref_params = reference[3]
    query_params = query[3]
    for index, (raw_ref, raw_query) in enumerate(zip(ref_params, query_params)):
        a, b = float(raw_ref), float(raw_query)
        if index < pixel_count:
            if abs(a * scale - b) > _CAMERA_SCALE_TOLERANCE_PX:
                return False
            continue
        unscaled = max(
            _CAMERA_DISTORTION_ATOL,
            _CAMERA_DISTORTION_RTOL * max(abs(a), abs(b)),
        )
        if unknown_model:
            unscaled = min(unscaled, _CAMERA_SCALE_TOLERANCE_PX)
        if abs(a - b) > unscaled:
            return False
    return True


def _camera_is_same_lens(reference, query) -> bool:
    """True when `reference` is `query` captured at a different resolution."""
    ref_model, ref_w, ref_h, ref_params = reference
    q_model, q_w, q_h, q_params = query
    if ref_model != q_model or len(ref_params) != len(q_params):
        return False
    comparison = _camera_comparison_scale(reference, query)
    if comparison is None:
        return False
    scale, pixel_count, unknown_model = comparison
    return _camera_parameters_match(
        reference, query, scale, pixel_count, unknown_model
    )


def _validate_reference_poses(
    path: Path,
    query_camera: QueryCamera,
    bundle_names: Iterable[str],
) -> int:
    raw = _json(path)
    reference = _camera_tuple(raw.get("camera"), "reference poses")
    query = (
        query_camera.model,
        query_camera.width,
        query_camera.height,
        query_camera.params,
    )
    if not _camera_is_same_lens(reference, query):
        raise ValueError(
            "reference poses camera is not the site query_camera at another "
            f"resolution: reference={reference[1]}x{reference[2]} params={reference[3]}, "
            f"query={query[1]}x{query[2]} params={query[3]}"
        )
    poses = raw.get("poses")
    if not isinstance(poses, dict) or not poses:
        raise ValueError("reference poses must contain a non-empty poses object")
    names = tuple(bundle_names)
    # Every bundle reference MUST have a pose: that is the direction that matters,
    # because a reference the localizer can match against but cannot place is a
    # pose computed from nothing. Extra poses are harmless -- bundle builds drop
    # frames on quality, so the poses file is legitimately a superset (urai ships
    # 1390 poses for 1383 bundle references).
    orphans = sorted(set(names) - set(poses))
    if orphans:
        raise ValueError(
            f"{len(orphans)} EDM bundle reference(s) have no pose, first: {orphans[0]}"
        )
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


_DEPLOY_RESULT_PREFIX = "SFM_ASSET_RESULT:"


def _run_deploy_inspector(path: Path, expression: str) -> object:
    """Run production asset code without mutating the UI process import path."""
    deploy = (
        Path(__file__).resolve().parents[2]
        / "定位演算法"
        / "deploy_code"
        / "sfm_glomap_deploy"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-B", "-c", expression, str(path.resolve())],
            cwd=deploy,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            capture_output=True,
            timeout=120.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"EDM asset inspector failed for {path}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise ValueError(
            f"EDM asset inspector rejected {path} (exit {completed.returncode}): {detail}"
        )
    payload = next(
        (
            line[len(_DEPLOY_RESULT_PREFIX):]
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(_DEPLOY_RESULT_PREFIX)
        ),
        None,
    )
    if payload is None:
        raise ValueError(f"EDM asset inspector returned no result for {path}")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"EDM asset inspector returned invalid JSON for {path}") from exc


def inspect_edm_bundle(path: Path) -> tuple[str, ...]:
    """Use the production EDM loader so import and runtime accept the same artifact."""
    expression = (
        "import json,sys; from reloc_localizer_edm import EDMRelocMap; "
        "bundle=EDMRelocMap.load(sys.argv[1]); "
        f"print({_DEPLOY_RESULT_PREFIX!r}+json.dumps(bundle.ref_names))"
    )
    try:
        names = _run_deploy_inspector(path, expression)
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError("production loader returned invalid reference names")
        return tuple(names)
    except Exception as exc:
        raise ValueError(f"invalid EDM localization bundle {path}: {exc}") from exc


def load_edm_runtime_profile(path: Path) -> dict:
    expression = (
        "import json,sys; from edm_profile import load_edm_production_profile; "
        "profile=load_edm_production_profile(sys.argv[1]); "
        f"print({_DEPLOY_RESULT_PREFIX!r}+json.dumps(profile))"
    )
    try:
        profile = _run_deploy_inspector(path, expression)
        if not isinstance(profile, dict):
            raise ValueError("production loader returned a non-object profile")
        return profile
    except Exception as exc:
        raise ValueError(f"invalid EDM runtime profile {path}: {exc}") from exc


def _validate_site_package_profile(profile: SiteProfile) -> None:
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


def _validate_site_package_assets(
    profile: SiteProfile, root: Path
) -> list[AssetCheck]:
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
    return checks


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
        _validate_site_package_profile(profile)
        checks = _validate_site_package_assets(profile, root)
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
        self.managed_root.mkdir(parents=True, exist_ok=True)
        lock_path = _site_asset_lock_path(self.managed_root, report.site_id)
        with _site_asset_lock(lock_path):
            return self._import_folder_locked(report, source_profile)

    def _import_folder_locked(
        self, report: ValidatedSitePackage, source_profile: SiteProfile
    ) -> ImportedSite:
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
                "map_align": (
                    source_profile.map_align,
                    Path("localization/T_align_gravity.json"),
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
                "map_align": "localization/T_align_gravity.json",
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
            site_root = site_pack_root_for_profile(path, self.managed_root)
            managed_profile = (
                None if site_root is None else (site_root / "site_profile.json").resolve()
            )
            if managed_profile is None or not managed_profile.is_file():
                raise ValueError(
                    "目前場域尚未對應到已匯入的場域資料夾；請先匯入場域資料"
                )
            path = managed_profile
        profile = load_site_profile(path)
        if (
            path.name != "site_profile.json"
            or path.parent.parent != self.managed_root
        ):
            raise ValueError("route and target imports require a managed site profile")
        return path, profile, _json(path)

    def _commit(
        self, source: Path, profile_path: Path, profile: SiteProfile, raw: dict
    ) -> ImportedAsset:
        target = profile_path.parent / self.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_asset = _new_temp_path(target.parent, f".{target.name}.")
        try:
            shutil.copy2(source, temp_asset)
        except Exception:
            _remove_file(temp_asset)
            raise

        lock_path = _site_asset_lock_path(self.managed_root, profile.site_id)
        backup_asset = None
        temp_profile = None
        target_replaced = False
        try:
            with _site_asset_lock(lock_path):
                current_raw = _json(profile_path)
                if target.exists() and not target.is_file():
                    raise ValueError(f"managed asset target is not a file: {target}")
                if target.is_file():
                    backup_asset = _new_temp_path(target.parent, f".{target.name}.", ".bak")
                    shutil.copy2(target, backup_asset)

                os.replace(temp_asset, target)
                temp_asset = None
                target_replaced = True

                current_raw["assets"][self.profile_key] = self.relative_path.as_posix()
                current_raw.setdefault("asset_sha256", {})[self.digest_key] = _sha256(target)
                flight = current_raw.setdefault("flight", {})
                flight["approved"] = False
                flight["route_clearance_approved"] = False
                flight["approval_note"] = (
                    "Imported asset changed; flight re-approval required."
                )
                temp_profile = _new_temp_path(profile_path.parent, ".site_profile.json.")
                temp_profile.write_text(
                    json.dumps(current_raw, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                load_site_profile(temp_profile)
                os.replace(temp_profile, profile_path)
                temp_profile = None
        except Exception:
            if target_replaced:
                try:
                    if backup_asset is None:
                        target.unlink()
                    else:
                        os.replace(backup_asset, target)
                        backup_asset = None
                except Exception as rollback_error:
                    raise RuntimeError(
                        f"asset transaction rollback failed for {target}"
                    ) from rollback_error
            raise
        finally:
            _remove_file(temp_asset)
            _remove_file(temp_profile)
            _remove_file(backup_asset)
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
