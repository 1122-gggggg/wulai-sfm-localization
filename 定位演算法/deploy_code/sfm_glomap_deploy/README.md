# 定位 runtime（direct 後端）

本目錄是可部署定位 runtime 的唯一入口，只剩 `direct` 後端：

- `production_localizer_factory.py`：唯一的生產 builder。`build_production_localizer`
  簽章不變，`backend` 只接受 `"direct"`（其他值直接 `ValueError`）；`Camera`
  改從 `pose_types` 匯入；`validate_camera_tuple` 保留給 worker/preflight 用。
  direct 沒有內建閾值：SHA 驗證過的 `direct-deployment-profile/v1` 就是全部
  gate 設定，`production_profile` 必填，BoQ cache / IVF index / matcher 覆寫一律拒絕。
- `localizer_registry.py`：只註冊 `direct`。capabilities：
  `required_assets=("localizer_profile",)`、
  `unsupported_assets=("megaloc_cache", "track_landmarks", "reference_index")`、
  `supports_production_profile=True`。
- `pose_types.py`：中立型別（`Pose`、`Localizer`、`BuiltLocalizer`、
  `LocalizerCapabilities`、`LocalizerProvider`）外加 `Camera`
 （`model`/`width`/`height`/`params`，由舊 `reloc_localizer_edm.py` 搬入，語意不變）。
- `artifact_integrity.py`：bundle / profile 的 SHA-256 驗證，direct builder 照用。

實際的 direct 部署實作（map 資產、profile loader、tracker adapter、live provider）
住在 `定位演算法/deploy_code/sfm_direct_deploy/`，本目錄只負責組裝與驗證，
不碰該目錄下任何檔案。

飛控、安全監控、Olympe frame source 與人工控制工具的唯一 owner 是
`定位演算法/flight_control/`。本目錄不再保存 `path_follow_flight.py`、
`olympe_frame_source.py` 或其他飛控相容副本；需要兩邊共用的模組也只能有一份。
所有權由下列純地面檢查強制：

```bash
python 定位演算法/validation/check_runtime_mirrors.py
```

自主航線入口永久鎖定；起飛只能由現場操作員在桌面 UI 親自執行。

泛化政策：

- 選參只使用 flight-disjoint 的 P116-P120 開發 fold。
- P157/P167 在 freeze 之後只做回歸，不得當 tuning 來源。
- 接受條件是 macro 提升，且 worst-fold 不回退。
- 確定性 2D appearance / image-plane stress 只是補充證據，不能證明 3D
  視差泛化。
- 沒有新的 held-out 航線與對應 ground truth，就不得宣稱任意 3D 視角覆蓋。
