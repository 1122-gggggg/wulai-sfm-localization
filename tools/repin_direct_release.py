#!/usr/bin/env python3
"""一鍵重算 direct release 全鏈 SHA（只重簽、不改內容）。

適用情境：改了 release 內容（direct_localizer_profile.json 的值、
localization bank 檔等）之後跑一次，30 秒內完成 RegenArts 上次手工
做的整條重簽。

只重算 SHA，不改任何內容值：bundle files[] 以外的欄位、凍結值、
檔名、路徑結構一律只讀。寫回格式與
tools/build_direct_site_release.py 的 write_json 逐位元組一致
（json.dumps(ensure_ascii=False, indent=2, allow_nan=False) + 尾換行）；
mission selection 沿用 tools/pin_mission_selection.py 的 sort_keys 格式。

鏈（由下而上）：
  bundle files[] sha/size
  -> bundle 檔 sha -> 兩份 site_profile 的 asset_sha256.localization_bundle
  -> profile 檔 sha -> 兩份 site_profile 的 asset_sha256.localizer_profile
  -> source_manifest.json generated 段（bundle/profile/site 三條）
  -> compat/localizer_direct_manifest.json artifacts 兩 sha
  -> mission selection 的 localizer pin（其他五 pin 只驗證不重寫）

fail-closed：寫入前先全量算完，任一非 SHA 欄對不上就報錯不寫
（files[] 路徑缺檔、model_sha256 與實際不符、names 列數與 npy 不符、
bank names 與 keyframes/manifest/poses 對不上、其他 pin 與實際不符）。

用法：
  .venv/bin/python tools/repin_direct_release.py \\
      --release 地圖檔/場域/river_site/releases/<release_id> [--check-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = REPO_ROOT / "定位演算法" / "deploy_code" / "sfm_direct_deploy"
CONTROL_DIR = REPO_ROOT / "控制介面程式"
MISSION_SELECTIONS = CONTROL_DIR / "mission_selections"

_CHUNK = 8 * 1024 * 1024
_BUNDLE_FILE_KEYS = frozenset({"path", "sha256", "size_bytes"})
_SHA256_CACHE: dict[tuple[int, int, int, int], str] = {}


class RepinError(RuntimeError):
    """任一前置檢查或 pin 對不上；呼叫端印出後以 exit 1 離開。"""


def sha256_file(path: Path) -> str:
    resolved = path.resolve()
    try:
        st = resolved.stat()
        key = (int(st.st_dev), int(st.st_ino), int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        key = None
    if key is not None and key in _SHA256_CACHE:
        return _SHA256_CACHE[key]

    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(block)
    val = digest.hexdigest()
    if key is not None:
        _SHA256_CACHE[key] = val
    return val


def dump_builder(payload: object) -> str:
    """與 build_direct_site_release.write_json 逐位元組一致。"""
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def dump_selection(payload: object) -> str:
    """與 pin_mission_selection 的寫入格式一致（sort_keys）。"""
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True) + "\n"


def load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RepinError(f"讀不到 {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RepinError(f"JSON 損壞 {path}: {exc}") from exc


def resolve_inside(base: Path, raw: object, *, label: str, root: Path) -> Path:
    """把 bundle/manifest 內的相對路徑解到 root 內；逃逸、絕對路徑都拒絕。"""
    if not isinstance(raw, str) or not raw:
        raise RepinError(f"{label} 必須是非空相對路徑")
    if Path(raw).is_absolute():
        raise RepinError(f"{label} 不可是絕對路徑: {raw}")
    target = Path(os.path.normpath(os.path.join(str(base), raw)))
    try:
        target.relative_to(root)
    except ValueError:
        raise RepinError(f"{label} 逃出 release 根目錄: {raw}") from None
    return target


def resolve_release(arg: str) -> dict[str, Path]:
    release = Path(arg).expanduser()
    if not release.is_absolute():
        release = (Path.cwd() / release).resolve()
    else:
        release = release.resolve()
    if release.parent.name != "releases":
        raise RepinError(f"--release 必須指到 <site>/releases/<id> 目錄: {release}")
    site_dir = release.parent.parent
    paths = {
        "release": release,
        "localization": release / "localization",
        "bundle": release / "localization" / "direct_bundle.json",
        "profile": release / "localization" / "direct_localizer_profile.json",
        "site": release / "site_profile.json",
        "root_site": site_dir / "site_profile.json",
        "source": release / "provenance" / "source_manifest.json",
        "localizer_manifest": release / "compat" / "localizer_direct_manifest.json",
    }
    for key, path in paths.items():
        if key in ("release", "localization"):
            if not path.is_dir():
                raise RepinError(f"目錄不存在: {path}")
            continue
        if not path.is_file() or path.is_symlink():
            raise RepinError(f"缺必要檔案（或為 symlink）: {path}")
    return paths


def find_selections(release: Path) -> list[Path]:
    """找出 localizer pin 指進這個 release 的 mission selection。"""
    found: list[Path] = []
    if not MISSION_SELECTIONS.is_dir():
        return found
    for selection in sorted(MISSION_SELECTIONS.glob("*.json")):
        try:
            raw = selection.read_text(encoding="utf-8")
            document = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict):
            continue
        localizer = document.get("localizer")
        if not isinstance(localizer, dict):
            continue
        pinned = localizer.get("path")
        if not isinstance(pinned, str):
            continue
        target = Path(os.path.normpath(os.path.join(str(selection.parent), pinned)))
        try:
            if target.resolve().is_relative_to(release):
                found.append(selection)
        except OSError:
            continue
    return found


def check_bank_consistency(release: Path, localization: Path, bundle: dict) -> None:
    """fail-closed：bank 行數、names 唯一性、names 與 keyframes/manifest/poses 子集。"""
    bank = bundle.get("reference_bank")
    if not isinstance(bank, dict):
        raise RepinError("bundle reference_bank 不是 object")
    descriptors = resolve_inside(
        localization,
        bank.get("descriptors"),
        label="bundle reference_bank.descriptors",
        root=release,
    )
    names_path = resolve_inside(
        localization,
        bank.get("names"),
        label="bundle reference_bank.names",
        root=release,
    )
    for path in (descriptors, names_path):
        if not path.is_file() or path.is_symlink():
            raise RepinError(f"bank 檔缺失或為 symlink: {path}")
    names = load_json(names_path)
    if not isinstance(names, list) or not names:
        raise RepinError(f"bank names 必須是非空陣列: {names_path}")
    names = [str(name) for name in names]
    if len(set(names)) != len(names):
        raise RepinError(f"bank names 有重複: {names_path}")
    try:
        import numpy as np
    except ImportError as exc:
        raise RepinError("bank 行數檢查需要 numpy（請用 .venv 跑本工具）") from exc
    try:
        shape = np.load(descriptors, mmap_mode="r", allow_pickle=False).shape
    except (OSError, ValueError) as exc:
        raise RepinError(f"讀不到 bank npy: {descriptors}: {exc}") from exc
    if len(shape) != 2 or shape[0] != len(names):
        raise RepinError(f"bank {descriptors} 形狀 {shape} 與 names 列數 {len(names)} 不符")

    _check_bank_subsets(names, localization, bundle, release)


def _check_bank_subsets(names: list[str], localization: Path, bundle: dict, release: Path) -> None:
    def _key(uri: object) -> str:
        text = str(uri)
        marker = "keyframes/images/"
        tail = text.split(marker, 1)[1] if marker in text else text
        parts = Path(tail).parts
        return "/".join(parts[-2:]) if len(parts) >= 2 else tail

    keyframes_path = resolve_inside(
        localization,
        bundle.get("keyframes_manifest"),
        label="bundle keyframes_manifest",
        root=release,
    )
    try:
        keyframes_keys = {
            _key(json.loads(line)["image_uri"])
            for line in keyframes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise RepinError(f"讀不到 keyframes manifest: {keyframes_path}: {exc}") from exc
    missing = [name for name in names if name not in keyframes_keys]
    if missing:
        raise RepinError(
            f"bank names 有 {len(missing)} 筆不在 keyframes.jsonl，例如: {missing[:3]}"
        )
    manifest_path = resolve_inside(
        localization,
        bundle.get("reference_manifest"),
        label="bundle reference_manifest",
        root=release,
    )
    try:
        manifest_names = {
            str(json.loads(line)["image_name"])
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise RepinError(f"讀不到 reference manifest: {manifest_path}: {exc}") from exc
    missing = [name for name in names if name not in manifest_names]
    if missing:
        raise RepinError(
            f"bank names 有 {len(missing)} 筆不在 reference_manifest.jsonl，例如: {missing[:3]}"
        )
    poses_path = localization / "reference_poses.json"
    if poses_path.is_file():
        poses = load_json(poses_path)
        if not isinstance(poses, dict) or not isinstance(poses.get("poses"), dict):
            raise RepinError(f"reference_poses.json 結構異常: {poses_path}")
        pose_names = poses["poses"]
        missing = [name for name in names if name not in pose_names]
        if missing:
            raise RepinError(
                f"bank names 有 {len(missing)} 筆在 reference_poses.json 缺 pose"
                f"，例如: {missing[:3]}"
            )


def _verify_shard_list_consistency(shards: list, label: str) -> None:
    for shard in shards:
        if not isinstance(shard, dict):
            raise RepinError(f"{label} 條目必須是 object")
        files = shard.get("files")
        if not isinstance(files, list):
            raise RepinError(f"{label} files 必須是陣列")
        if "shard_root_sha256" in shard:
            expected_root = hashlib.sha256(
                "".join(f"{e['path']} {e['sha256']}\n" for e in files).encode("utf-8")
            ).hexdigest()
            if shard["shard_root_sha256"] != expected_root:
                raise RepinError(f"{label} {shard.get('shard_id')} shard_root_sha256 不符")


def check_shard_consistency(release: Path, localization: Path) -> None:
    shard_path = localization / "shard_manifest.json"
    if not shard_path.is_file():
        return
    manifest = load_json(shard_path)
    if not isinstance(manifest, dict):
        raise RepinError(f"shard_manifest.json 根必須是 object: {shard_path}")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise RepinError(f"shard_manifest.json shards 必須是陣列: {shard_path}")
    _verify_shard_list_consistency(shards, "shard")
    depth_shards = manifest.get("depth_shards")
    if isinstance(depth_shards, list):
        _verify_shard_list_consistency(depth_shards, "depth shard")


def recompute_bundle_files(release: Path, localization: Path, bundle: dict) -> list[dict]:
    """重算 files[] 每條 path 的 sha256 + size_bytes（順序、其他欄位不動）。"""
    entries = bundle.get("files")
    if not isinstance(entries, list) or not entries:
        raise RepinError("bundle files[] 必須是非空陣列")
    fresh: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise RepinError("bundle files[] 條目必須是 object")
        if set(entry) != _BUNDLE_FILE_KEYS:
            raise RepinError(f"bundle files[] 條目鍵異常: {sorted(entry)}")
        target = resolve_inside(
            localization, entry["path"], label="bundle files[].path", root=release
        )
        if not target.is_file() or target.is_symlink():
            raise RepinError(f"bundle files[] 路徑缺檔或為 symlink: {entry['path']}")
        actual_size = target.stat().st_size
        if not isinstance(entry["size_bytes"], int) or isinstance(entry["size_bytes"], bool):
            raise RepinError(f"bundle files[] size_bytes 非整數: {entry['path']}")
        fresh.append(
            {
                "path": entry["path"],
                "sha256": sha256_file(target),
                "size_bytes": actual_size,
            }
        )
    return fresh


def verify_model_digests(release: Path, bundle: dict) -> None:
    """fail-closed：bundle model_sha256 必須與實際 model bin 一致（本工具不改它）。"""
    declared = bundle.get("model_sha256")
    if not isinstance(declared, dict):
        raise RepinError("bundle model_sha256 不是 object")
    model_dir = release / "model"
    for name, expected in declared.items():
        target = model_dir / str(name)
        if not target.is_file():
            raise RepinError(f"model 檔缺失: {target}")
        actual = sha256_file(target)
        if actual != expected:
            raise RepinError(f"bundle model_sha256.{name} 與實際不符（model 變更超出本工具範圍）")


def site_pin_targets(profile_path: Path, profile: dict) -> dict[str, Path]:
    """site_profile 內五個 asset 路徑（route_json 為 null 則略過）。"""
    assets = profile.get("assets")
    if not isinstance(assets, dict):
        raise RepinError(f"{profile_path} assets 不是 object")
    digest = profile.get("asset_sha256")
    if not isinstance(digest, dict):
        raise RepinError(f"{profile_path} asset_sha256 不是 object")
    base = profile_path.parent
    targets: dict[str, Path] = {}
    mapping = {
        "map_ply": assets.get("map_ply"),
        "localization_bundle": assets.get("localization_bundle"),
        "localizer_profile": profile.get("localizer_profile"),
        "map_reference_poses": profile.get("map_reference_poses"),
        "map_align": profile.get("map_align"),
        "shard_manifest": assets.get("shard_manifest"),
        "inductor_prewarm": assets.get("inductor_prewarm"),
    }
    for key, raw in mapping.items():
        if raw is None:
            continue
        if not isinstance(raw, str):
            raise RepinError(f"{profile_path} {key} 路徑非字串")
        targets[key] = base / Path(raw)
    return targets


def verify_stable_pins(targets: dict[str, Path], digest: dict, *, skip: set[str]) -> None:
    """fail-closed：非本次重簽範圍的 pin 必須與實際一致，否則停住不寫。"""
    for key, target in targets.items():
        if key in skip:
            continue
        expected = digest.get(key)
        if not isinstance(expected, str):
            raise RepinError(f"asset_sha256.{key} 非字串 digest")
        if not target.is_file():
            raise RepinError(f"asset_sha256.{key} 指到的檔缺失: {target}")
        actual = sha256_file(target)
        if actual != expected:
            raise RepinError(
                f"asset_sha256.{key} 與實際不符（非本次重簽範圍，停住不寫）: "
                f"pinned={expected} actual={actual}"
            )


def check_pin_file(base: Path, entry: object, *, label: str) -> str:
    """驗一條 mission selection pin，回傳實際 sha（None 則回傳空字串表示略過）。"""
    if entry is None:
        return ""
    if not isinstance(entry, dict):
        raise RepinError(f"{label} 必須是 object 或 null")
    raw_path = entry.get("path")
    expected = entry.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected, str):
        raise RepinError(f"{label} 的 path/sha256 非字串")
    target = Path(os.path.normpath(os.path.join(str(base), raw_path)))
    if not target.is_file():
        raise RepinError(f"{label} 指到的檔缺失: {target}")
    actual = sha256_file(target)
    if actual != expected:
        raise RepinError(
            f"{label} 與實際不符（只驗證不重寫，停住不寫）: pinned={expected} actual={actual}"
        )
    return actual


def verify_with_loaders(
    *,
    bundle_path: Path,
    bundle_sha: str,
    profile_path: Path,
    profile_sha: str,
    selections: list[Path],
) -> list[str]:
    """唯讀驗證：DirectMapAssets.load、load_direct_profile、mission 六 pin 全對。"""
    for package in (str(DEPLOY_DIR), str(CONTROL_DIR)):
        if package not in sys.path:
            sys.path.insert(0, package)
    try:
        from direct_map import DirectMapAssets
        from direct_profile import load_direct_profile
        from mission_manifest import load_mission_selection
    except ImportError as exc:
        raise RepinError(f"載入驗證器失敗: {exc}") from exc
    report: list[str] = []
    assets = DirectMapAssets.load(bundle_path, expected_sha256=bundle_sha)
    report.append(
        f"DirectMapAssets.load ok: {len(assets.ref_names)} refs, "
        f"bank={assets.bank_name}, sha={assets.sha256[:12]}…"
    )
    profile = load_direct_profile(profile_path, expected_sha256=profile_sha)
    report.append(
        f"load_direct_profile ok: name={profile.name}, "
        f"map_scale={profile.map_scale}, sha={profile.sha256[:12]}…"
    )
    for selection in selections:
        mission = load_mission_selection(selection, workspace_root=REPO_ROOT, verify_files=True)
        pins = dict(mission.component_sha256)
        order = (
            ["localizer", "map", "vehicle", "route", "site"]
            + sorted(key for key in pins if key.startswith("calibration_"))
            + sorted(
                key
                for key in pins
                if key not in ("localizer", "map", "vehicle", "route", "site")
                and not key.startswith("calibration_")
            )
        )
        detail = ", ".join(f"{key}✓" for key in order if key in pins)
        report.append(f"mission {mission.selection_id} 六 pin 全對: {detail}")
    return report


def _verify_source_generated_inputs(generated: dict, release: Path) -> None:
    for key in (
        "map/map.ply",
        "localization/reference_poses.json",
        "localization/T_align_gravity.json",
    ):
        expected = generated.get(key)
        target = release / Path(key)
        if not isinstance(expected, str) or not target.is_file():
            raise RepinError(f"source_manifest generated.{key} 異常或缺檔")
        actual = sha256_file(target)
        if actual != expected:
            raise RepinError(
                f"source_manifest generated.{key} 與實際不符（停住不寫）: "
                f"pinned={expected} actual={actual}"
            )


def _audit_source_manifest(
    source: dict, release: Path, bundle_sha: str, profile_sha: str, site_sha: str
) -> str:
    generated = source.get("generated")
    if not isinstance(generated, dict):
        raise RepinError("source_manifest.json generated 不是 object")
    _verify_source_generated_inputs(generated, release)
    for key in (
        "localization/direct_bundle.json",
        "localization/direct_localizer_profile.json",
        "site_profile.json",
    ):
        if key not in generated:
            raise RepinError(f"source_manifest generated 缺鍵: {key}")
    generated["localization/direct_bundle.json"] = bundle_sha
    generated["localization/direct_localizer_profile.json"] = profile_sha
    generated["site_profile.json"] = site_sha
    if "localization/shard_manifest.json" in generated:
        shard_path = release / "localization/shard_manifest.json"
        if shard_path.is_file():
            generated["localization/shard_manifest.json"] = sha256_file(shard_path)
    if "compat/inductor_prewarm.json" in generated:
        prewarm_path = release / "compat/inductor_prewarm.json"
        if prewarm_path.is_file():
            generated["compat/inductor_prewarm.json"] = sha256_file(prewarm_path)
    return dump_builder(source)


def _audit_site_profiles(
    site: dict, root_site: dict, paths: dict[str, Path], bundle_sha: str, profile_sha: str
) -> tuple[str, str, str]:
    for _, document, path in (
        ("release site_profile", site, paths["site"]),
        ("root site_profile", root_site, paths["root_site"]),
    ):
        targets = site_pin_targets(path, document)
        verify_stable_pins(
            targets,
            document["asset_sha256"],
            skip={"localization_bundle", "localizer_profile", "shard_manifest", "inductor_prewarm"},
        )
        document["asset_sha256"]["localization_bundle"] = bundle_sha
        document["asset_sha256"]["localizer_profile"] = profile_sha
        if "shard_manifest" in document["asset_sha256"] and "shard_manifest" in targets:
            document["asset_sha256"]["shard_manifest"] = sha256_file(targets["shard_manifest"])
        if "inductor_prewarm" in document["asset_sha256"] and "inductor_prewarm" in targets:
            document["asset_sha256"]["inductor_prewarm"] = sha256_file(targets["inductor_prewarm"])
    site_text = dump_builder(site)
    site_sha = hashlib.sha256(site_text.encode("utf-8")).hexdigest()
    root_site_text = dump_builder(root_site)
    return site_text, site_sha, root_site_text


def _audit_localizer_manifest(
    paths: dict[str, Path], release: Path, bundle_sha: str, profile_sha: str
) -> tuple[str, str]:
    manifest = load_json(paths["localizer_manifest"])
    if not isinstance(manifest, dict):
        raise RepinError("localizer_direct_manifest.json 根必須是 object")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RepinError("localizer manifest artifacts 不是 object")
    for key in ("bundle", "profile"):
        entry = artifacts.get(key)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise RepinError(f"localizer manifest artifacts.{key} 異常")
        target = resolve_inside(
            paths["localizer_manifest"].parent,
            entry["path"],
            label=f"localizer manifest artifacts.{key}.path",
            root=release,
        )
        if not target.is_file():
            raise RepinError(f"localizer manifest 指到的檔缺失: {target}")
    artifacts["bundle"]["sha256"] = bundle_sha
    artifacts["profile"]["sha256"] = profile_sha
    manifest_text = dump_builder(manifest)
    manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    return manifest_text, manifest_sha


def _audit_update_manifests(paths: dict[str, Path]) -> dict:
    release = paths["release"]
    localization = paths["localization"]

    bundle = load_json(paths["bundle"])
    if not isinstance(bundle, dict):
        raise RepinError("direct_bundle.json 根必須是 object")
    profile_text = paths["profile"].read_bytes()
    profile_sha = hashlib.sha256(profile_text).hexdigest()

    fresh_files = recompute_bundle_files(release, localization, bundle)
    verify_model_digests(release, bundle)
    check_bank_consistency(release, localization, bundle)
    check_shard_consistency(release, localization)

    bundle_fresh = dict(bundle)
    bundle_fresh["files"] = fresh_files
    bundle_text = dump_builder(bundle_fresh)
    bundle_sha = hashlib.sha256(bundle_text.encode("utf-8")).hexdigest()

    site = load_json(paths["site"])
    root_site = load_json(paths["root_site"])
    if not isinstance(site, dict) or not isinstance(root_site, dict):
        raise RepinError("site_profile.json 根必須是 object")
    site_text, site_sha, root_site_text = _audit_site_profiles(
        site, root_site, paths, bundle_sha, profile_sha
    )

    source = load_json(paths["source"])
    if not isinstance(source, dict):
        raise RepinError("source_manifest.json 根必須是 object")
    source_text = _audit_source_manifest(source, release, bundle_sha, profile_sha, site_sha)
    manifest_text, manifest_sha = _audit_localizer_manifest(paths, release, bundle_sha, profile_sha)

    return {
        "bundle": bundle,
        "fresh_files": fresh_files,
        "bundle_text": bundle_text,
        "bundle_sha": bundle_sha,
        "profile_sha": profile_sha,
        "site_text": site_text,
        "site_sha": site_sha,
        "root_site_text": root_site_text,
        "source_text": source_text,
        "manifest_text": manifest_text,
        "manifest_sha": manifest_sha,
    }


def _update_selections(release: Path, manifest_sha: str) -> tuple[list[Path], dict[Path, str]]:
    selections = find_selections(release)
    if not selections:
        raise RepinError("找不到 localizer pin 指進這個 release 的 mission selection")
    selection_texts: dict[Path, str] = {}
    for selection in selections:
        document = load_json(selection)
        if not isinstance(document, dict):
            raise RepinError(f"mission selection 根必須是 object: {selection}")
        base = selection.parent
        for calibration in document.get("calibrations", []):
            check_pin_file(base, calibration, label="mission calibrations pin")
        for key in ("map", "vehicle", "route", "site"):
            check_pin_file(base, document.get(key), label=f"mission {key} pin")
        localizer = document.get("localizer")
        if not isinstance(localizer, dict):
            raise RepinError("mission localizer pin 異常")
        check_pin_file(base, localizer, label="mission localizer pin")
        localizer["sha256"] = manifest_sha
        selection_texts[selection] = dump_selection(document)
    return selections, selection_texts


def _check_disk_profiles_and_manifests(paths: dict[str, Path], state: dict, show: object) -> None:
    site_disk = load_json(paths["site"])
    root_disk = load_json(paths["root_site"])
    source_disk = load_json(paths["source"])
    manifest_disk = load_json(paths["localizer_manifest"])
    assert isinstance(site_disk, dict)
    assert isinstance(root_disk, dict)
    assert isinstance(source_disk, dict)
    assert isinstance(manifest_disk, dict)
    show(
        "release site_profile asset_sha256.localization_bundle",
        site_disk["asset_sha256"]["localization_bundle"],
        state["bundle_sha"],
    )
    show(
        "release site_profile asset_sha256.localizer_profile",
        site_disk["asset_sha256"]["localizer_profile"],
        state["profile_sha"],
    )
    if "shard_manifest" in site_disk.get("asset_sha256", {}):
        shard_path = paths["release"] / "localization/shard_manifest.json"
        show(
            "release site_profile asset_sha256.shard_manifest",
            site_disk["asset_sha256"]["shard_manifest"],
            sha256_file(shard_path) if shard_path.is_file() else "",
        )
    if "inductor_prewarm" in site_disk.get("asset_sha256", {}):
        prewarm_path = paths["release"] / "compat/inductor_prewarm.json"
        show(
            "release site_profile asset_sha256.inductor_prewarm",
            site_disk["asset_sha256"]["inductor_prewarm"],
            sha256_file(prewarm_path) if prewarm_path.is_file() else "",
        )
    show(
        "root site_profile asset_sha256.localization_bundle",
        root_disk["asset_sha256"]["localization_bundle"],
        state["bundle_sha"],
    )
    show(
        "root site_profile asset_sha256.localizer_profile",
        root_disk["asset_sha256"]["localizer_profile"],
        state["profile_sha"],
    )
    show(
        "source_manifest generated bundle",
        source_disk["generated"]["localization/direct_bundle.json"],
        state["bundle_sha"],
    )
    show(
        "source_manifest generated profile",
        source_disk["generated"]["localization/direct_localizer_profile.json"],
        state["profile_sha"],
    )
    show(
        "source_manifest generated site",
        source_disk["generated"]["site_profile.json"],
        state["site_sha"],
    )
    if "localization/shard_manifest.json" in source_disk.get("generated", {}):
        shard_path = paths["release"] / "localization/shard_manifest.json"
        show(
            "source_manifest generated shard_manifest",
            source_disk["generated"]["localization/shard_manifest.json"],
            sha256_file(shard_path) if shard_path.is_file() else "",
        )
    if "compat/inductor_prewarm.json" in source_disk.get("generated", {}):
        prewarm_path = paths["release"] / "compat/inductor_prewarm.json"
        show(
            "source_manifest generated inductor_prewarm",
            source_disk["generated"]["compat/inductor_prewarm.json"],
            sha256_file(prewarm_path) if prewarm_path.is_file() else "",
        )
    show(
        "localizer manifest artifacts.bundle",
        manifest_disk["artifacts"]["bundle"]["sha256"],
        state["bundle_sha"],
    )
    show(
        "localizer manifest artifacts.profile",
        manifest_disk["artifacts"]["profile"]["sha256"],
        state["profile_sha"],
    )


def _check_only_diff(
    paths: dict[str, Path],
    state: dict,
    selections: list[Path],
) -> int:
    problems: list[str] = []

    def _show(label: str, pinned: object, fresh: str) -> None:
        status = "OK" if pinned == fresh else "MISMATCH"
        print(f"[check] {label}: pinned={pinned} fresh={fresh} {status}")
        if pinned != fresh:
            problems.append(label)

    for old, new in zip(state["bundle"]["files"], state["fresh_files"]):
        if old["sha256"] != new["sha256"] or old["size_bytes"] != new["size_bytes"]:
            _show(
                f"bundle files {old['path']}",
                f"{old['sha256']}/{old['size_bytes']}",
                f"{new['sha256']}/{new['size_bytes']}",
            )
    _check_disk_profiles_and_manifests(paths, state, _show)
    for selection in selections:
        on_disk = load_json(selection)
        assert isinstance(on_disk, dict)
        _show(
            f"mission {selection.name} localizer pin",
            on_disk["localizer"]["sha256"],
            state["manifest_sha"],
        )
    if problems:
        print(
            f"[check] FAIL: {len(problems)} 處 SHA 不符（未寫入）",
            file=sys.stderr,
        )
        return 1
    for line in verify_with_loaders(
        bundle_path=paths["bundle"],
        bundle_sha=state["bundle_sha"],
        profile_path=paths["profile"],
        profile_sha=state["profile_sha"],
        selections=selections,
    ):
        print(f"[verify] {line}")
    print("[check] 全鏈 SHA 一致")
    return 0


def _apply_writes(
    paths: dict[str, Path],
    state: dict,
    selection_texts: dict[Path, str],
    selections: list[Path],
) -> None:
    writes: list[tuple[Path, str]] = [
        (paths["bundle"], state["bundle_text"]),
        (paths["site"], state["site_text"]),
        (paths["root_site"], state["root_site_text"]),
        (paths["source"], state["source_text"]),
        (paths["localizer_manifest"], state["manifest_text"]),
        *selection_texts.items(),
    ]
    for path, text in writes:
        if path.read_bytes() != text.encode("utf-8"):
            path.write_text(text, encoding="utf-8")
            print(f"[repin] 已更新 {path}")
        else:
            print(f"[repin] 未變 {path}")
    print(f"[repin] bundle={state['bundle_sha']}")
    print(f"[repin] profile={state['profile_sha']}")
    print(f"[repin] site_profile={state['site_sha']}")
    print(f"[repin] localizer_manifest={state['manifest_sha']}")
    for line in verify_with_loaders(
        bundle_path=paths["bundle"],
        bundle_sha=state["bundle_sha"],
        profile_path=paths["profile"],
        profile_sha=state["profile_sha"],
        selections=selections,
    ):
        print(f"[verify] {line}")
    print("[repin] 全鏈重簽完成")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, help="<site>/releases/<id> 目錄")
    parser.add_argument("--check-only", action="store_true", help="只驗證不寫入；SHA 不符就列出來")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        paths = resolve_release(args.release)
        state = _audit_update_manifests(paths)
        selections, selection_texts = _update_selections(paths["release"], state["manifest_sha"])
        if args.check_only:
            return _check_only_diff(paths, state, selections)
        _apply_writes(paths, state, selection_texts, selections)
        return 0
    except RepinError as exc:
        print(f"[repin] FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
