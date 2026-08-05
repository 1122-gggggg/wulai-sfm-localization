"""Replaceable ports for site, route, and inspection-target imports."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class AssetCheck:
    key: str
    label: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class ValidatedSitePackage:
    folder: Path
    site_id: str
    display_name: str
    coordinate_frame_id: str
    reference_count: int
    checks: tuple[AssetCheck, ...]


@dataclass(frozen=True)
class ImportedSite:
    profile_path: Path
    site_id: str
    already_present: bool = False


@dataclass(frozen=True)
class ImportedAsset:
    profile_path: Path
    asset_path: Path


class SitePackagePort(Protocol):
    def validate_folder(self, folder: str | Path) -> ValidatedSitePackage: ...

    def import_folder(self, folder: str | Path) -> ImportedSite: ...


class RoutePort(Protocol):
    def import_file(
        self, source: str | Path, profile_path: str | Path
    ) -> ImportedAsset: ...


class InspectionTargetPort(Protocol):
    def import_file(
        self, source: str | Path, profile_path: str | Path
    ) -> ImportedAsset: ...
