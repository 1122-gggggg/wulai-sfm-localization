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
import time
import uuid
from contextlib import contextmanager
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

from map_point_io import DENSE_DISPLAY_PLY_NAME
from site_asset_interfaces import (
    AssetCheck,
    ImportedAsset,
    ImportedSite,
    ValidatedSitePackage,
)

_CONTROL_ROOT = Path(__file__).resolve().parents[1]
if str(_CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(_CONTROL_ROOT))
_FLIGHT_CONTROL = _CONTROL_ROOT.parent / "定位演算法" / "flight_control"
if str(_FLIGHT_CONTROL) not in sys.path:
    sys.path.insert(0, str(_FLIGHT_CONTROL))

from route_domain import reject_json_constant, validate_flight_route_fields

from mission_manifest import sha256_file
from site_profile import (
    ROUTE_EDITOR_AUTO_APPROVAL_NOTE,
    QueryCamera,
    SiteProfile,
    load_site_profile,
)
from workspace_layout import workspace_from_file


_REQUIRED_ASSETS = (
    ("map_ply", "點雲地圖"),
    ("localization_bundle", "定位 bundle"),
    ("localizer_profile", "定位 runtime profile"),
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
    try:
        return sha256_file(path)
    except OSError as exc:
        raise ValueError(f"cannot hash {path}: {exc}") from exc


def _json_bytes(raw: dict) -> bytes:
    return (json.dumps(raw, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _referenced_path(owner: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = owner.parent / candidate
    return candidate.resolve()


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
        # The editor loads the bound dense cloud itself once map.ply matches.
        and path.name != DENSE_DISPLAY_PLY_NAME
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

#: PLY files the route editor itself writes, plus its dense display cloud. They
#: live beside the map but are never the map, so proposing one would be a trap.
_ROUTE_PREVIEW_PLY = (
    "flight_path.ply", "route.ply", "poles.ply", "wires.ply", DENSE_DISPLAY_PLY_NAME,
)


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
    """Every formal route JSON under a site folder, newest first.

    Legacy route_drafts/ files are intentionally excluded: the current workflow
    has only formal add/edit operations. Which formal route the aircraft flies is
    the operator's choice, not the first one found.
    """
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        return []
    found = [
        path.resolve()
        for path in root.rglob("*.json")
        if path.is_file()
        and not path.is_symlink()
        and "route_drafts" not in path.parts
        and (
            "route" in path.name.casefold()
            or "flight_path" in path.name.casefold()
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


@dataclass(frozen=True)
class SiteFolderFile:
    """One required file of a candidate site folder, as found on disk."""

    label: str
    path: Path | None
    size_bytes: int | None
    problem: str | None = None


@dataclass(frozen=True)
class SiteFolderPreview:
    """A cheap pre-import look at a folder: no hashing, no bundle or PLY parsing.

    It only decides whether offering 匯入 is worthwhile;
    LocalSitePackageProvider.validate_folder still verifies every digest, the
    bundle and the reference poses before anything is copied.
    """

    folder: Path
    display_name: str = ""
    site_id: str = ""
    coordinate_frame_id: str = ""
    camera: str = ""
    files: tuple[SiteFolderFile, ...] = ()
    error: str | None = None

    @property
    def ready(self) -> bool:
        return (
            self.error is None
            and bool(self.files)
            and all(item.problem is None for item in self.files)
        )


def _preview_file(label: str, path: Path | None, root: Path) -> SiteFolderFile:
    if path is None:
        return SiteFolderFile(label, None, None, "site_profile.json 未指定")
    if not _is_inside(path, root):
        return SiteFolderFile(label, path, None, "位於所選資料夾外")
    try:
        size = path.stat().st_size if path.is_file() else None
    except OSError:
        size = None
    if size is None:
        return SiteFolderFile(label, path, None, "找不到檔案")
    return SiteFolderFile(label, path, size)


def preview_site_folder(folder: str | Path) -> SiteFolderPreview:
    """Report what a mapping-output folder declares and which required files exist."""
    root = Path(folder).expanduser().resolve()
    manifest = root / "site_profile.json"
    if not manifest.is_file():
        return SiteFolderPreview(root, error="此資料夾沒有 site_profile.json")
    manifest_file = SiteFolderFile("場域 manifest", manifest, manifest.stat().st_size)
    try:
        profile = load_site_profile(manifest, validate_files=False)
    except Exception as exc:
        return SiteFolderPreview(
            root, files=(manifest_file,), error=f"site_profile.json 無法讀取：{exc}",
        )
    try:
        _validate_site_package_profile(profile)
    except ValueError as exc:
        error = str(exc)
    else:
        error = None
    camera = profile.query_camera
    return SiteFolderPreview(
        root,
        display_name=profile.display_name,
        site_id=profile.site_id,
        coordinate_frame_id=(
            profile.coordinate_frame.id if profile.coordinate_frame is not None else ""
        ),
        camera="" if camera is None else f"{camera.model} {camera.width}×{camera.height}",
        files=(manifest_file, *(
            _preview_file(label, getattr(profile, key), root)
            for key, label in _REQUIRED_ASSETS
        )),
        error=error,
    )


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
        try:
            relative = profile.map_ply.resolve().relative_to(root)
            site_root = root / relative.parts[0]
        except ValueError:
            site_root = root / profile.site_id
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


def _validate_bound_route(route: dict, profile: SiteProfile) -> None:
    if not isinstance(route, dict):
        raise ValueError("route JSON must be an object")
    frame_id = profile.coordinate_frame.id if profile.coordinate_frame else None
    if frame_id is None:
        raise ValueError("route approval requires a bound map coordinate frame")
    if (
        "map_ply_sha256" in route
        and route["map_ply_sha256"] != profile.asset_sha256.map_ply
    ):
        raise ValueError("route authored map SHA-256 does not match the selected map")
    validate_flight_route_fields(
        route,
        expected_site_id=profile.site_id,
        expected_coordinate_frame_id=frame_id,
    )


def _resolve_selected_route(route_path: str | Path) -> Path:
    candidate = Path(route_path).expanduser().absolute()
    if any(part.is_symlink() for part in (candidate, *candidate.parents)):
        raise ValueError("selected route must not use symlinks")
    return candidate.resolve()


def _route_receipt_profile(
    selected: Path, profile: SiteProfile, map_digest: str | None
) -> SiteProfile:
    """Return the profile whose authored-route receipts govern the route."""
    for parent in selected.parents:
        owner = parent / "site_profile.json"
        if not owner.is_file():
            continue
        if owner.resolve() == profile.source:
            break
        bound = load_site_profile(owner)
        if (
            bound.site_id != profile.site_id
            or bound.coordinate_frame != profile.coordinate_frame
            or bound.asset_sha256.map_ply != map_digest
            or _sha256(bound.map_ply) != map_digest
        ):
            raise ValueError("selected route approval belongs to a different map")
        return bound
    return profile


def _authored_route_receipt_approved(
    selected: Path,
    profile: SiteProfile,
    receipt_profile: SiteProfile,
    digest: str,
    map_digest: str | None,
) -> bool:
    receipts = () if receipt_profile.flight is None else receipt_profile.flight.authored_routes
    for receipt in receipts:
        receipt_path = (receipt_profile.source.parent / receipt.path).resolve()
        if selected != receipt_path:
            continue
        if (
            not _is_inside(selected, receipt_profile.source.parent)
            or receipt.sha256 != digest
            or receipt.map_sha256 != map_digest
            or receipt.coordinate_frame_id != profile.coordinate_frame.id
        ):
            raise ValueError("authored route approval SHA-256 or map identity mismatch")
        return True
    return False


def _pinned_route_approved(
    selected: Path, profile: SiteProfile, digest: str
) -> bool:
    # Existing pinned profiles remain eligible only for their exact route.
    # A route's self-reported source marker never establishes provenance.
    flight = profile.flight
    if selected != profile.route_json:
        return False
    if not (flight.approved and flight.route_clearance_approved):
        return False
    if profile.asset_sha256.route_json != digest:
        raise ValueError("selected route SHA-256 mismatch")
    return True


def profile_for_selected_route(
    profile_path: str | Path, route_path: str | Path
) -> SiteProfile:
    """Verify a selected route's persisted approval without approving a preview.

    An unapproved valid route remains viewable, but gets a locked flight overlay.
    Receipts never grant controller, mission, hardware, or localization readiness.
    """
    profile = load_site_profile(profile_path)
    selected = _resolve_selected_route(route_path)
    route = _json(selected)
    _validate_bound_route(route, profile)
    map_digest = _sha256(profile.map_ply)
    if map_digest != profile.asset_sha256.map_ply:
        raise ValueError("selected route map SHA-256 mismatch")
    digest = _sha256(selected)
    flight = profile.flight
    receipt_profile = _route_receipt_profile(selected, profile, map_digest)
    if flight is not None:
        approved = _authored_route_receipt_approved(
            selected, profile, receipt_profile, digest, map_digest
        ) or _pinned_route_approved(selected, profile, digest)
        flight = replace(
            flight,
            approved=approved,
            route_clearance_approved=approved,
            approval_note=(
                ROUTE_EDITOR_AUTO_APPROVAL_NOTE if approved
                else "Selected route has no verified AUTO approval."
            ),
        )
    return replace(
        profile,
        route_json=selected,
        asset_sha256=replace(profile.asset_sha256, route_json=digest),
        flight=flight,
    )


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
            f"{len(orphans)} localization bundle reference(s) have no pose, "
            f"first: {orphans[0]}"
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


_DEPLOY_CODE_ROOT = Path(__file__).resolve().parents[2] / "定位演算法" / "deploy_code"


def _run_deploy_inspector(
    path: Path,
    expression: str,
    arguments: tuple[str, ...] = (),
    *,
    deploy_package: str = "sfm_glomap_deploy",
) -> object:
    """Run production asset code without mutating the UI process import path."""
    deploy = _DEPLOY_CODE_ROOT / deploy_package
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                expression,
                str(path.resolve()),
                *arguments,
            ],
            cwd=deploy,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            capture_output=True,
            timeout=120.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"deploy asset inspector failed for {path}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise ValueError(
            f"deploy asset inspector rejected {path} (exit {completed.returncode}): {detail}"
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
        raise ValueError(f"deploy asset inspector returned no result for {path}")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"deploy asset inspector returned invalid JSON for {path}"
        ) from exc



def inspect_direct_bundle(
    path: Path, expected_sha256: str | None = None
) -> tuple[str, ...]:
    """Structurally validate a direct-localization-bundle/v1 with its own loader."""
    expression = (
        "import json,sys; from direct_paths import vendor_sys_path; vendor_sys_path(); "
        "from direct_map import DirectMapAssets; "
        "assets=DirectMapAssets.load(sys.argv[1], expected_sha256=sys.argv[2] or None); "
        f"print({_DEPLOY_RESULT_PREFIX!r}+json.dumps(list(assets.ref_names)))"
    )
    try:
        names = _run_deploy_inspector(
            path,
            expression,
            (expected_sha256 or "",),
            deploy_package="sfm_direct_deploy",
        )
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError("production loader returned invalid reference names")
        return tuple(names)
    except Exception as exc:
        raise ValueError(f"invalid direct localization bundle {path}: {exc}") from exc


def load_direct_runtime_profile(path: Path) -> dict:
    """Validate a direct-deployment-profile/v1 through the production loader."""
    expression = (
        "import json,sys; from direct_paths import vendor_sys_path; vendor_sys_path(); "
        "from direct_profile import load_direct_profile; "
        "profile=load_direct_profile(sys.argv[1]); "
        f"print({_DEPLOY_RESULT_PREFIX!r}+json.dumps("
        "{'name': str(profile.name), 'map_scale': float(profile.map_scale), "
        "'reloc_top_k': int(profile.reloc.top_k)}))"
    )
    try:
        profile = _run_deploy_inspector(
            path, expression, deploy_package="sfm_direct_deploy"
        )
        if not isinstance(profile, dict):
            raise ValueError("production loader returned a non-object profile")
        return profile
    except Exception as exc:
        raise ValueError(f"invalid direct runtime profile {path}: {exc}") from exc


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
    }
    if profile.localizer != "direct":
        # The direct runtime lives outside every site folder (sfm_direct_deploy),
        # so its deploy dir is part of the package contract rather than a
        # separately managed asset the importer must reject.
        separate_assets["localizer_deploy_dir"] = profile.localizer_deploy_dir
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
        bundle_inspector: Callable[[Path, str], tuple[str, ...]] = inspect_direct_bundle,
        profile_loader: Callable[[Path], dict] = load_direct_runtime_profile,
    ):
        self.managed_root = Path(managed_root).expanduser().resolve()
        self.bundle_inspector = bundle_inspector
        self.profile_loader = profile_loader

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
        self.profile_loader(profile.localizer_profile)
        bundle_sha256 = profile.asset_sha256.localization_bundle
        assert bundle_sha256 is not None
        names = self.bundle_inspector(profile.localization_bundle, bundle_sha256)
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
            if source_profile.localizer != "direct":
                raise ValueError(
                    "only direct site packages can be imported; "
                    f"got localizer {source_profile.localizer!r}"
                )
            copies = {
                "map_ply": (source_profile.map_ply, Path("map/map.ply")),
                "localization_bundle": (
                    source_profile.localization_bundle,
                    Path("localization/direct_bundle.json"),
                ),
                "localizer_profile": (
                    source_profile.localizer_profile,
                    Path("localization/direct_localizer_profile.json"),
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
            deploy_dir = source_profile.localizer_deploy_dir
            if deploy_dir is None:
                raise ValueError("direct site package must declare localizer_deploy_dir")
            profile_json = {
                "schema_version": 2,
                "site_id": report.site_id,
                "display_name": report.display_name,
                "localizer": "direct",
                "localizer_deploy_dir": os.path.relpath(deploy_dir.resolve(), destination),
                "localizer_profile": "localization/direct_localizer_profile.json",
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
                    "localization_bundle": "localization/direct_bundle.json",
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


def _authored_route_receipt(path: str, sha256: str, current: SiteProfile) -> dict:
    return {
        "path": path,
        "sha256": sha256,
        "map_sha256": current.asset_sha256.map_ply,
        "coordinate_frame_id": current.coordinate_frame.id,
    }


@dataclass
class _AssetCommitState:
    """Mutable bookkeeping for one managed-asset commit transaction."""

    temp_asset: Path | None
    backup_asset: Path | None = None
    temp_profile: Path | None = None
    linked_backups: dict[Path, Path] = field(default_factory=dict)
    linked_temps: dict[Path, Path] = field(default_factory=dict)
    linked_replaced: list[Path] = field(default_factory=list)
    target_replaced: bool = False
    rollback_error: Exception | None = None

    def cleanup(self) -> None:
        _remove_file(self.temp_asset)
        _remove_file(self.temp_profile)
        for temporary in self.linked_temps.values():
            _remove_file(temporary)
        if self.rollback_error is None:
            _remove_file(self.backup_asset)
            for backup in self.linked_backups.values():
                _remove_file(backup)


class _LocalSupplementProvider:
    profile_key = ""
    digest_key = ""
    relative_path = Path()

    def __init__(self, managed_root: str | Path):
        self.managed_root = Path(managed_root).expanduser().resolve()

    def _linked_manifest_updates(
        self,
        target: Path,
        digest: str,
        previous_target: Path | None = None,
    ) -> dict[Path, bytes]:
        return {}

    def _adoption_site_root(self, requested: Path, source: SiteProfile) -> Path:
        site_root = site_pack_root_for_profile(requested, self.managed_root)
        if site_root is not None:
            return site_root
        try:
            relative = source.map_ply.relative_to(self.managed_root)
            return self.managed_root / relative.parts[0]
        except ValueError:
            return self.managed_root / source.site_id

    @staticmethod
    def _adoption_manifest(requested: Path, site_root: Path) -> dict:
        raw = _json(requested)
        # Adoption keeps the deployed contract, including its controller
        # and independent hardware references, without granting approval.
        for key, value in raw["assets"].items():
            asset = _referenced_path(requested, value)
            if asset is not None:
                raw["assets"][key] = os.path.relpath(asset, site_root)
        for key in ("localizer_deploy_dir", "localizer_profile", "map_reference_poses", "map_align"):
            asset = _referenced_path(requested, raw.get(key))
            if asset is not None:
                raw[key] = os.path.relpath(asset, site_root)
        for section, keys in _NESTED_MANIFEST_REFERENCES:
            fields = raw.get(section)
            if not isinstance(fields, dict):
                continue
            for key in keys:
                asset = _referenced_path(requested, fields.get(key))
                if asset is not None:
                    fields[key] = os.path.relpath(asset, site_root)
        if isinstance(raw.get("flight"), dict):
            raw["flight"].pop("authored_routes", None)
        return raw

    @classmethod
    def _write_adopted_profile(cls, requested: Path, site_root: Path, path: Path) -> None:
        site_root.mkdir(parents=True, exist_ok=True)
        raw = cls._adoption_manifest(requested, site_root)
        temporary = _new_temp_path(site_root, ".site_profile.json.")
        try:
            temporary.write_bytes(_json_bytes(raw))
            load_site_profile(temporary)
            _sync_path(temporary)
            os.replace(temporary, path)
            _sync_path(site_root)
        finally:
            _remove_file(temporary)

    @staticmethod
    def _require_adoptable_source(source: SiteProfile) -> None:
        if source.schema_version != 2 or source.coordinate_frame is None:
            raise ValueError("managed site adoption requires a schema v2 bound coordinate frame")
        if (
            source.asset_sha256.map_ply is None
            or _sha256(source.map_ply) != source.asset_sha256.map_ply
        ):
            raise ValueError("managed site adoption map SHA-256 mismatch")

    @staticmethod
    def _require_adopted_identity(profile: SiteProfile, source: SiteProfile) -> None:
        if (
            profile.site_id != source.site_id
            or profile.coordinate_frame != source.coordinate_frame
            or profile.asset_sha256.map_ply != source.asset_sha256.map_ply
            or _sha256(profile.map_ply) != source.asset_sha256.map_ply
        ):
            raise ValueError("managed site adoption map/coordinate identity mismatch")

    def _load_managed_profile(self, profile_path: str | Path) -> tuple[Path, SiteProfile, dict]:
        requested = Path(profile_path).expanduser().resolve()
        source = load_site_profile(requested)
        if requested.name == "site_profile.json" and requested.parent.parent == self.managed_root:
            return requested, source, _json(requested)
        self._require_adoptable_source(source)
        site_root = self._adoption_site_root(requested, source)
        path = site_root / "site_profile.json"
        self.managed_root.mkdir(parents=True, exist_ok=True)
        with _site_asset_lock(_site_asset_lock_path(self.managed_root, source.site_id)):
            if not path.is_file():
                self._write_adopted_profile(requested, site_root, path)
        profile = load_site_profile(path)
        self._require_adopted_identity(profile, source)
        raw = _json(path)
        source_flight = _json(requested).get("flight")
        if isinstance(source_flight, dict) and source_flight.get("controller") is not None:
            raw.setdefault("flight", {})["controller"] = source_flight["controller"]
        return path, profile, raw

    def _stage_linked_manifests(
        self,
        target: Path,
        digest: str,
        previous_target: Path | None,
        backups: dict[Path, Path],
        temps: dict[Path, Path],
    ) -> None:
        for linked_path, payload in self._linked_manifest_updates(
            target,
            digest,
            previous_target,
        ).items():
            if not linked_path.is_file() or linked_path.is_symlink():
                raise ValueError(
                    f"linked mission manifest is not a regular file: {linked_path}"
                )
            backup = _new_temp_path(
                linked_path.parent, f".{linked_path.name}.", ".bak"
            )
            shutil.copy2(linked_path, backup)
            backups[linked_path] = backup
            temp = _new_temp_path(linked_path.parent, f".{linked_path.name}.")
            temp.write_bytes(payload)
            temps[linked_path] = temp

    @staticmethod
    def _replace_linked_manifests(
        temps: dict[Path, Path], replaced: list[Path]
    ) -> None:
        for linked_path, temp in list(temps.items()):
            os.replace(temp, linked_path)
            temps.pop(linked_path, None)
            replaced.append(linked_path)

    @staticmethod
    def _rollback_asset_transaction(
        target: Path,
        backup_asset: Path | None,
        target_replaced: bool,
        linked_backups: dict[Path, Path],
        linked_replaced: list[Path],
    ) -> Exception | None:
        rollback_error = None
        for linked_path in reversed(linked_replaced):
            backup = linked_backups.get(linked_path)
            if backup is None:
                continue
            try:
                os.replace(backup, linked_path)
                linked_backups.pop(linked_path, None)
            except Exception as exc:
                rollback_error = rollback_error or exc
        if target_replaced:
            try:
                if backup_asset is None:
                    target.unlink()
                else:
                    os.replace(backup_asset, target)
            except Exception as exc:
                rollback_error = rollback_error or exc
        return rollback_error

    def _require_route_save_identity(
        self, temp_asset: Path, profile: SiteProfile, current: SiteProfile
    ) -> None:
        if (
            current.coordinate_frame != profile.coordinate_frame
            or current.asset_sha256.map_ply != profile.asset_sha256.map_ply
        ):
            raise ValueError("route save map identity changed during transaction")
        _validate_bound_route(_json(temp_asset), current)
        if _sha256(current.map_ply) != current.asset_sha256.map_ply:
            raise ValueError("route save map SHA-256 mismatch")

    def _previous_asset_target(self, profile_path: Path, current_raw: dict) -> Path:
        previous_value = current_raw.get("assets", {}).get(self.profile_key)
        if not previous_value:
            return (profile_path.parent / self.relative_path).resolve()
        return (profile_path.parent / str(previous_value)).resolve()

    @staticmethod
    def _commit_flight(current_raw: dict, raw: dict) -> dict:
        flight = current_raw.get("flight")
        if not isinstance(flight, dict):
            flight = current_raw["flight"] = {}
        source_flight = raw.get("flight")
        if flight.get("controller") is None and isinstance(source_flight, dict):
            flight["controller"] = source_flight.get("controller")
        return flight

    @staticmethod
    def _legacy_editor_receipt(
        flight: dict,
        current: SiteProfile,
        profile_path: Path,
        previous_target: Path,
        receipts: list,
    ) -> dict | None:
        # Migrate only the previously hash-pinned editor approval,
        # never a route's untrusted self-reported source marker.
        old_sha = current.asset_sha256.route_json
        if not (
            previous_target.is_file()
            and _is_inside(previous_target, profile_path.parent / "routes")
            and flight.get("approval_note") == ROUTE_EDITOR_AUTO_APPROVAL_NOTE
            and flight.get("approved") is True
            and flight.get("route_clearance_approved") is True
            and old_sha == _sha256(previous_target)
        ):
            return None
        previous_relative = previous_target.relative_to(profile_path.parent).as_posix()
        if any(item["path"] == previous_relative for item in receipts):
            return None
        return _authored_route_receipt(previous_relative, old_sha, current)

    def _update_route_receipts(
        self,
        flight: dict,
        current: SiteProfile,
        profile_path: Path,
        previous_target: Path,
        selected_relative_path: Path,
        digest: str,
        approve_for_auto: bool,
    ) -> None:
        flight.setdefault("coordinate_frame_id", current.coordinate_frame.id)
        receipts = list(flight.get("authored_routes", []))
        legacy = self._legacy_editor_receipt(
            flight, current, profile_path, previous_target, receipts
        )
        if legacy is not None:
            receipts.append(legacy)
        receipts = [
            item for item in receipts
            if item["path"] != selected_relative_path.as_posix()
        ]
        if approve_for_auto:
            receipts.append(
                _authored_route_receipt(selected_relative_path.as_posix(), digest, current)
            )
        flight["authored_routes"] = receipts

    @staticmethod
    def _replace_asset_target(
        state: _AssetCommitState, target: Path, digest: str
    ) -> None:
        if target.exists() and not target.is_file():
            raise ValueError(f"managed asset target is not a file: {target}")
        if target.is_file():
            state.backup_asset = _new_temp_path(target.parent, f".{target.name}.", ".bak")
            shutil.copy2(target, state.backup_asset)
        os.replace(state.temp_asset, target)
        state.temp_asset = None
        state.target_replaced = True
        if _sha256(target) != digest:
            raise ValueError("saved asset SHA-256 mismatch")

    def _publish_profile(
        self,
        state: _AssetCommitState,
        target: Path,
        digest: str,
        previous_target: Path,
        profile_path: Path,
        current_raw: dict,
    ) -> None:
        state.temp_profile = _new_temp_path(profile_path.parent, ".site_profile.json.")
        state.temp_profile.write_bytes(_json_bytes(current_raw))
        load_site_profile(state.temp_profile)
        _sync_path(state.temp_profile)
        self._stage_linked_manifests(
            target, digest, previous_target, state.linked_backups, state.linked_temps,
        )
        for temporary in state.linked_temps.values():
            _sync_path(temporary)
        profile_backup = _new_temp_path(profile_path.parent, ".site_profile.json.", ".bak")
        shutil.copy2(profile_path, profile_backup)
        state.linked_backups[profile_path] = profile_backup
        self._replace_linked_manifests(state.linked_temps, state.linked_replaced)
        os.replace(state.temp_profile, profile_path)
        state.temp_profile = None
        state.linked_replaced.append(profile_path)

    def _commit_locked(
        self,
        state: _AssetCommitState,
        source_profile: SiteProfile,
        raw: dict,
        profile_path: Path,
        target: Path,
        selected_relative_path: Path,
        digest: str,
        approve_for_auto: bool,
    ) -> None:
        current = load_site_profile(profile_path)
        current_raw = _json(profile_path)
        is_route = self.profile_key == "route_json"
        if is_route:
            self._require_route_save_identity(state.temp_asset, source_profile, current)
        previous_target = self._previous_asset_target(profile_path, current_raw)
        flight = self._commit_flight(current_raw, raw)
        if is_route:
            self._update_route_receipts(
                flight, current, profile_path, previous_target,
                selected_relative_path, digest, approve_for_auto,
            )
        self._replace_asset_target(state, target, digest)
        current_raw["assets"][self.profile_key] = selected_relative_path.as_posix()
        current_raw.setdefault("asset_sha256", {})[self.digest_key] = digest
        flight["approved"] = bool(approve_for_auto)
        flight["route_clearance_approved"] = bool(approve_for_auto)
        flight["approval_note"] = (
            ROUTE_EDITOR_AUTO_APPROVAL_NOTE if approve_for_auto
            else "Imported asset changed; flight re-approval required."
        )
        self._publish_profile(
            state, target, digest, previous_target, profile_path, current_raw
        )
        if approve_for_auto:
            verified = profile_for_selected_route(profile_path, target)
            if verified.flight is None or not verified.flight.approved:
                raise ValueError("saved route approval did not verify")
        for folder in {target.parent, *(path.parent for path in state.linked_replaced)}:
            _sync_path(folder)

    def _abort_commit(self, state: _AssetCommitState, target: Path) -> None:
        state.rollback_error = self._rollback_asset_transaction(
            target,
            state.backup_asset,
            state.target_replaced,
            state.linked_backups,
            state.linked_replaced,
        )
        if state.rollback_error is not None:
            raise RuntimeError(
                f"asset transaction rollback failed for {target}; recovery backups retained"
            ) from state.rollback_error

    def _commit(
        self,
        source: Path,
        profile_path: Path,
        profile: SiteProfile,
        raw: dict,
        *,
        approve_for_auto: bool = False,
        relative_path: Path | None = None,
        expected_source_sha256: str | None = None,
    ) -> ImportedAsset:
        selected_relative_path = relative_path or self.relative_path
        target = (profile_path.parent / selected_relative_path).resolve()
        if not _is_inside(target, profile_path.parent):
            raise ValueError("managed asset target must stay inside the site folder")
        target.parent.mkdir(parents=True, exist_ok=True)
        state = _AssetCommitState(
            temp_asset=_new_temp_path(target.parent, f".{target.name}.")
        )
        lock_path = _site_asset_lock_path(self.managed_root, profile.site_id)
        try:
            shutil.copy2(source, state.temp_asset)
            digest = _sha256(state.temp_asset)
            if expected_source_sha256 is not None and digest != expected_source_sha256:
                raise ValueError("route changed while saving; approval not committed")
            _sync_path(state.temp_asset)
            with _site_asset_lock(lock_path):
                try:
                    self._commit_locked(
                        state, profile, raw, profile_path, target,
                        selected_relative_path, digest, approve_for_auto,
                    )
                except Exception:
                    self._abort_commit(state, target)
                    raise
        finally:
            state.cleanup()
        return ImportedAsset(profile_path, target, approved_for_auto=bool(approve_for_auto))


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


_NESTED_MANIFEST_REFERENCES = (
    ("pose_chain", ("site_alignment", "camera_body_extrinsic")),
    ("hardware_approval", ("receipt", "signature", "trust_store")),
)


def _route_deletion_candidates(profile_path: Path, requested_profile: Path) -> set[Path]:
    workspace = workspace_from_file(profile_path).root
    control = workspace / "控制介面程式"
    candidates = set(profile_path.parent.rglob("*.json"))
    candidates.update((control / "site_profiles").glob("*.json"))
    candidates.update((control / "mission_components" / "routes").glob("*.json"))
    candidates.update((control / "mission_selections").glob("*.json"))
    candidates.update({profile_path, requested_profile})
    return candidates


def _load_scanned_manifest(
    candidate: Path, profile_path: Path, requested_profile: Path
) -> dict | None:
    try:
        return _json(candidate)
    except ValueError:
        if candidate in {profile_path, requested_profile}:
            raise
        # Other site assets may be arrays, or incomplete editor output.
        # They are not manifests owned by this transaction.
        return None


def _manifest_asset_references(candidate: Path, raw: dict, assets: dict) -> dict[str, Path | None]:
    references = {key: _referenced_path(candidate, value) for key, value in assets.items()}
    for key in ("localizer_profile", "map_reference_poses", "map_align"):
        references[key] = _referenced_path(candidate, raw.get(key))
    for section, keys in _NESTED_MANIFEST_REFERENCES:
        nested = raw.get(section)
        if isinstance(nested, dict):
            for key in keys:
                references[f"{section}.{key}"] = _referenced_path(candidate, nested.get(key))
    return references


def _detach_route_from_manifest(raw: dict, assets: dict) -> None:
    assets["route_json"] = None
    digests = raw.get("asset_sha256")
    if isinstance(digests, dict):
        digests.pop("route_json", None)
    flight = raw.get("flight")
    if flight is None:
        flight = raw["flight"] = {}
    flight["approved"] = False
    flight["route_clearance_approved"] = False
    flight["approval_note"] = "Selected route deleted; flight re-approval required."


def _detached_manifest_bytes(candidate: Path, raw: dict, assets: dict, target: Path) -> bytes | None:
    # A map/target/runtime asset must never become deletable merely
    # because it also happens to contain a waypoints array.
    references = _manifest_asset_references(candidate, raw, assets)
    if any(path == target and key != "route_json" for key, path in references.items()):
        raise ValueError("selected route is also a non-route site asset")
    changed = references.get("route_json") == target
    if changed:
        _detach_route_from_manifest(raw, assets)
    flight = raw.get("flight")
    if isinstance(flight, dict) and isinstance(flight.get("authored_routes"), list):
        receipts = flight["authored_routes"]
        remaining = [
            item for item in receipts
            if _referenced_path(candidate, item["path"]) != target
        ]
        if remaining != receipts:
            flight["authored_routes"] = remaining
            changed = True
    return _json_bytes(raw) if changed else None


def _package_references_route(candidate: Path, raw: dict, target: Path) -> bool:
    route_ref = raw.get("route")
    return (
        isinstance(route_ref, dict)
        and _referenced_path(candidate, route_ref.get("path")) == target
    )


def _selection_detach_updates(selections: dict[Path, dict], removals: set[Path]) -> dict[Path, bytes]:
    updates: dict[Path, bytes] = {}
    removed_components = {path.resolve() for path in removals}
    for selection_path, raw in selections.items():
        route_ref = raw.get("route")
        if (
            isinstance(route_ref, dict)
            and _referenced_path(selection_path, route_ref.get("path")) in removed_components
        ):
            raw["route"] = None
            raw["approval"] = None
            updates[selection_path] = _json_bytes(raw)
    return updates


def _unlink_linked_manifests(removals: set[Path], replaced: list[Path]) -> None:
    for linked_path in sorted(removals):
        linked_path.unlink()
        replaced.append(linked_path)


def _cleanup_route_deletion(
    temps: dict[Path, Path],
    backup_asset: Path | None,
    backups: dict[Path, Path],
    rollback_error: BaseException | None,
) -> None:
    for temp in temps.values():
        _remove_file(temp)
    if rollback_error is None:
        _remove_file(backup_asset)
        for backup in backups.values():
            _remove_file(backup)


class LocalRouteProvider(_LocalSupplementProvider):
    profile_key = "route_json"
    digest_key = "route_json"
    relative_path = Path("routes/flight_route.json")

    def _deletion_manifest_changes(
        self, target: Path, profile_path: Path, requested_profile: Path
    ) -> tuple[dict[Path, bytes], set[Path]]:
        """Detach optional references; remove only directly linked mandatory packages."""
        candidates = _route_deletion_candidates(profile_path, requested_profile)
        updates: dict[Path, bytes] = {}
        removals: set[Path] = set()
        selections: dict[Path, dict] = {}
        for candidate in sorted(candidates):
            if candidate == target:
                continue
            raw = _load_scanned_manifest(candidate, profile_path, requested_profile)
            if raw is None:
                continue
            assets = raw.get("assets")
            if isinstance(assets, dict) and "schema_version" in raw:
                payload = _detached_manifest_bytes(candidate, raw, assets, target)
                if payload is not None:
                    updates[candidate] = payload
            elif raw.get("schema") == "sfm-route-package/v1":
                if _package_references_route(candidate, raw, target):
                    # route-package/v1 requires a real route ArtifactRef. It
                    # cannot represent an empty route; do not leave an invalid
                    # package or retarget it to an unchosen replacement.
                    removals.add(candidate)
            elif raw.get("schema") == "sfm-mission-selection/v1":
                selections[candidate] = raw
        updates.update(_selection_detach_updates(selections, removals))
        return updates, removals

    def _resolve_deletion_target(self, source: str | Path, path: Path) -> Path:
        candidate = Path(source).expanduser().absolute()
        target = candidate.resolve()
        if not _is_inside(target, path.parent):
            raise ValueError("deleted route must stay inside the active site folder")
        # Check the lexical path before resolving: resolve() hides symlinks to
        # both files and directories, including links pointing back into site.
        for part in (candidate, *candidate.parents):
            if part.is_symlink():
                raise ValueError("deleted route must not use symlinks")
        return target

    def _plan_route_deletion(
        self, path: Path, target: Path, requested_profile: Path
    ) -> tuple[dict[Path, bytes], set[Path]]:
        if target not in {route.path for route in describe_site_routes(path.parent)}:
            raise ValueError("selected file is not a listed site route")
        route_raw = _json(target)
        if "assets" in route_raw or route_raw.get("schema") in {
            "sfm-route-package/v1", "sfm-mission-selection/v1",
        }:
            raise ValueError("selected route is a manifest, not a route asset")
        return self._deletion_manifest_changes(target, path, requested_profile)

    def _stage_route_deletion_manifests(
        self,
        updates: dict[Path, bytes],
        removals: set[Path],
        backups: dict[Path, Path],
        temps: dict[Path, Path],
    ) -> None:
        for linked_path in sorted(updates.keys() | removals):
            if not linked_path.is_file() or any(
                part.is_symlink()
                for part in (linked_path, *linked_path.parents)
            ):
                raise ValueError(
                    f"linked mission manifest is not a regular file: {linked_path}"
                )
            backup = _new_temp_path(
                linked_path.parent, f".{linked_path.name}.", ".bak"
            )
            backups[linked_path] = backup
            shutil.copy2(linked_path, backup)
            if linked_path in updates:
                temp = _new_temp_path(linked_path.parent, f".{linked_path.name}.")
                temps[linked_path] = temp
                temp.write_bytes(updates[linked_path])
                raw = json.loads(updates[linked_path])
                if "schema_version" in raw:
                    load_site_profile(temp)

    def delete_file(
        self, source: str | Path, profile_path: str | Path
    ) -> ImportedAsset:
        """Delete a listed route and detach its references as one locked transaction."""
        requested_profile = Path(profile_path).expanduser().resolve()
        path, profile, _raw = self._load_managed_profile(requested_profile)
        target = self._resolve_deletion_target(source, path)

        backups: dict[Path, Path] = {}
        temps: dict[Path, Path] = {}
        replaced: list[Path] = []
        backup_asset = None
        target_deleted = False
        rollback_error = None
        lock_path = _site_asset_lock_path(self.managed_root, profile.site_id)
        try:
            with _site_asset_lock(lock_path):
                try:
                    profile = load_site_profile(path)
                    updates, removals = self._plan_route_deletion(path, target, requested_profile)
                    self._stage_route_deletion_manifests(updates, removals, backups, temps)
                    backup_asset = _new_temp_path(
                        target.parent, f".{target.name}.", ".bak"
                    )
                    shutil.copy2(target, backup_asset)
                    self._replace_linked_manifests(temps, replaced)
                    _unlink_linked_manifests(removals, replaced)
                    target.unlink()
                    target_deleted = True
                    profile = load_site_profile(path)
                except Exception:
                    rollback_error = self._rollback_asset_transaction(
                        target, backup_asset, target_deleted, backups, replaced
                    )
                    if rollback_error is not None:
                        raise RuntimeError(
                            f"route deletion rollback failed for {target}; recovery backups retained"
                        ) from rollback_error
                    raise
        finally:
            _cleanup_route_deletion(temps, backup_asset, backups, rollback_error)
        return ImportedAsset(
            path, target,
            approved_for_auto=bool(profile.flight and profile.flight.approved),
        )

    def _linked_manifest_updates(
        self,
        target: Path,
        digest: str,
        previous_target: Path | None = None,
    ) -> dict[Path, bytes]:
        if self.managed_root.name != "場域" or self.managed_root.parent.name != "地圖檔":
            return {}
        workspace = self.managed_root.parent.parent
        route_components = workspace / "控制介面程式" / "mission_components" / "routes"
        selections = workspace / "控制介面程式" / "mission_selections"
        if not route_components.is_dir():
            return {}

        updates: dict[Path, bytes] = {}
        component_digests: dict[Path, str] = {}
        for component_path in sorted(route_components.glob("*.json")):
            component = _json(component_path)
            route_ref = component.get("route")
            if not isinstance(route_ref, dict):
                continue
            referenced = _referenced_path(component_path, route_ref.get("path"))
            if referenced not in {target, previous_target}:
                continue
            route_ref["path"] = Path(
                os.path.relpath(target, component_path.parent)
            ).as_posix()
            route_ref["sha256"] = digest
            route_document = _json(target)
            component["frame"] = {
                "kind": "map",
                "id": route_document["coordinate_frame_id"],
            }
            component["site_id"] = route_document["site_id"]
            payload = _json_bytes(component)
            updates[component_path] = payload
            component_digests[component_path.resolve()] = hashlib.sha256(payload).hexdigest()

        for selection_path in sorted(selections.glob("*.json")) if selections.is_dir() else ():
            selection = _json(selection_path)
            route_ref = selection.get("route")
            if not isinstance(route_ref, dict):
                continue
            component_path = _referenced_path(selection_path, route_ref.get("path"))
            component_digest = component_digests.get(component_path)
            if component_digest is None:
                continue
            route_ref["sha256"] = component_digest
            updates[selection_path] = _json_bytes(selection)
        return updates

    def import_file(
        self,
        source: str | Path,
        profile_path: str | Path,
        *,
        approve_for_auto: bool = False,
        replace_route: str | Path | None = None,
    ) -> ImportedAsset:
        requested = load_site_profile(profile_path)
        route_path = Path(source).expanduser().resolve()
        payload = route_path.read_bytes()
        route = json.loads(payload, parse_constant=reject_json_constant)
        source_digest = hashlib.sha256(payload).hexdigest()
        _validate_bound_route(route, requested)
        path, profile, profile_raw = self._load_managed_profile(profile_path)
        if replace_route is None:
            suffix = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            relative_path = Path("routes") / f"flight_route_{suffix}.json"
        else:
            candidate = Path(replace_route).expanduser().absolute()
            if any(part.is_symlink() for part in (candidate, *candidate.parents)):
                raise ValueError("edited route must not use symlinks")
            replace_path = candidate.resolve()
            routes_root = (path.parent / "routes").resolve()
            if (
                not _is_inside(replace_path, routes_root)
                or not replace_path.is_file()
                or replace_path.is_symlink()
                or replace_path.suffix.casefold() != ".json"
            ):
                raise ValueError(
                    "edited route must be an existing regular JSON file under the site's routes folder"
                )
            relative_path = replace_path.relative_to(path.parent)
        return self._commit(
            route_path,
            path,
            profile,
            profile_raw,
            approve_for_auto=approve_for_auto,
            relative_path=relative_path,
            expected_source_sha256=source_digest,
        )


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
