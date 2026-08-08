"""Operator actions and button specifications, independent of Tk layout."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from site_asset_interfaces import InspectionTargetPort, RoutePort, SitePackagePort


@dataclass(frozen=True)
class CommandButton:
    label: str
    command: str


# Flight command strings remain centralized and unchanged. The view only renders them.
FLIGHT_MODE_BUTTONS = (
    CommandButton("手動/搖桿 (Esc)", "manual"),
    CommandButton("恢復電腦控制", "pc_control"),
    CommandButton("懸停", "hover"),
    CommandButton("原地降落", "land"),
)

MISSION_MODE_BUTTONS = (
    CommandButton("起飛", "takeoff"),
    # 定位鎖定 removed 2026-08-06: the boot lock engages automatically once 開始定位
    # is pressed (see OperatorApp.update_boot_lock), so the button only ever
    # re-triggered something the operator had already started. The "boot_lock"
    # control action itself stays in the contract for the automatic path.
)


@dataclass(frozen=True)
class ActionResult:
    message: str
    profile_path: Path | None = None
    asset_path: Path | None = None
    kind: str | None = None


class SiteAssetActions:
    """Button-facing orchestration with no dependency on Tk or the drone backend."""

    def __init__(
        self,
        sites: SitePackagePort,
        routes: RoutePort,
        targets: InspectionTargetPort,
        current_profile: str | Path | None = None,
    ):
        self.sites = sites
        self.routes = routes
        self.targets = targets
        self.current_profile = (
            None
            if current_profile in (None, "")
            else Path(current_profile).expanduser().resolve()
        )

    def import_site_folder(self, folder: str | Path) -> ActionResult:
        imported = self.sites.import_folder(folder)
        self.current_profile = imported.profile_path
        suffix = "（已存在，驗證一致）" if imported.already_present else ""
        return ActionResult(
            f"場域 {imported.site_id} 匯入完成{suffix}；尚未套用、未解鎖飛行",
            imported.profile_path,
            kind="site",
        )

    def import_route(self, source: str | Path) -> ActionResult:
        if self.current_profile is None:
            raise ValueError("請先匯入建圖端場域資料夾")
        imported = self.routes.import_file(source, self.current_profile)
        return ActionResult(
            f"航線已匯入 {imported.asset_path.name}；飛行核准已維持關閉",
            imported.profile_path,
            imported.asset_path,
            "route",
        )

    def import_targets(self, source: str | Path) -> ActionResult:
        if self.current_profile is None:
            raise ValueError("請先匯入建圖端場域資料夾")
        imported = self.targets.import_file(source, self.current_profile)
        return ActionResult(
            f"巡檢目標已匯入 {imported.asset_path.name}；飛行核准已維持關閉",
            imported.profile_path,
            imported.asset_path,
            "targets",
        )


def require_safe_site_switch(
    *, is_live: bool, flight_state: str | None, commands_inflight: bool
) -> None:
    if not is_live:
        return
    state = str(flight_state or "").strip().lower()
    if state != "landed":
        raise ValueError(f"目前飛行狀態為 {state or 'unknown'}；只有 landed 可套用")
    if commands_inflight:
        raise ValueError("仍有飛行指令處理中，暫不允許切換場域")


def replace_site_profile_argument(argv: list[str], profile: Path) -> list[str]:
    """Return argv with exactly one --site-profile, preserving all other flags."""
    output = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == "--site-profile":
            index += 2
            continue
        if value.startswith("--site-profile="):
            index += 1
            continue
        output.append(value)
        index += 1
    output.extend(("--site-profile", str(Path(profile).resolve())))
    return output
