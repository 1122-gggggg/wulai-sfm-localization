# 建圖工具（`EDM工具包/build`）

這個目錄只剩一條建圖路線：`edm_fixedpose_retriangulate.py`，做出 COLMAP
model + MegaLoc bank，餵給 `direct` 後端（two-rate：CPU KLT 快迴路 + 背景
GPU MegaLoc/EDM reloc）。

| | 新路線 |
|---|---|
| 腳本 | `edm_fixedpose_retriangulate.py` |
| 檢索 | MegaLoc，input 322（`edm_fixedpose_retriangulate.py:69`），`sideview` bank |
| 匹配 | EDM，640×640、coarse top-k 2240（`edm_fixedpose_retriangulate.py:66-67`） |
| 關鍵點身分 | 2 px 量化 cell（`CELL_PX = 2.0`，`edm_fixedpose_retriangulate.py:56`） |
| 2D→3D | 真的重新三角化，輸出 COLMAP `points3D.bin` |
| 參考影像 | 留在磁碟 `keyframes/images/<video_id>/*.jpg`，由 `keyframes.jsonl` 的 `image_sha256` 逐張驗 |
| 定位後端 | `direct`（two-rate：CPU KLT 快迴路 + 背景 GPU MegaLoc/EDM reloc） |
| 五大件的 bundle 檔 | `direct_bundle.json`（`direct-localization-bundle/v1`） |
| 五大件的 profile 檔 | `direct_localizer_profile.json`（`direct-deployment-profile/v1`） |

欄位級規格見 [`控制介面程式/site_profiles/建圖端輸出規格.md`](../../../控制介面程式/site_profiles/建圖端輸出規格.md)。

---

## 新路線：`edm_fixedpose_retriangulate.py`（`direct` 後端）

輸入是一次 GlueMap 重建的**凍結位姿**（1058 張影像，其中 12 張 POSE_ONLY；
`edm_fixedpose_retriangulate.py:54-55`）。位姿一格都不動，只把觀測層整個換成
EDM 密集匹配 → 2 px 量化 → 固定位姿重新三角化，產出一個新的 COLMAP model。

所有路徑都是 CLI 參數，預設從 workspace 推導：

```bash
.venv/bin/python 定位演算法/EDM工具包/build/edm_fixedpose_retriangulate.py --help
```

| 參數 | 預設 |
|---|---|
| `--map-root` | `river-deploy-5060-20260908/river_gluemap_all8_direct_20260831` |
| `--experiment-root` | `<map-root>/experiments/edm_fixedpose_retriangulate` |
| `--edm-repo` | `定位演算法/deploy_code/runtime/EDM` |
| `--edm-checkpoint` | `定位演算法/deploy_code/runtime/EDM/weights/edm_outdoor.ckpt` |
| `--megaloc-source` | `執行環境/torch_hub_cache/gmberton_MegaLoc_main` |
| `--megaloc-checkpoint` | `~/.cache/huggingface/hub/models--gberton--MegaLoc/snapshots/*/model.safetensors` 的第一筆 |
| `--baseline-smoke` | `<experiment-root>/records/smoke_localization.json` |

`river_map_quality` 與 `sfm_diagnosis` 由
`定位演算法/deploy_code/sfm_direct_deploy/vendor` 提供，腳本自己掛 `sys.path`
（`edm_fixedpose_retriangulate.py:35-42`），不需要設 `PYTHONPATH`。
`--megaloc-source` / `--megaloc-checkpoint` 會覆寫 map root 內
`localization/localizer_config.json` 的對應欄位，所以那份交付檔裡的建圖機絕對路徑
不會被使用。

### 階段、指令、產物、閘值

| 階段 | 指令 | 產物 | 硬閘值（不通過就 `SystemExit`） |
|---|---|---|---|
| 0 幾何自檢 | `selftest` | 只印 `selftest ok` | 2 px cell 量化必須把 1.9 px 判同格、2.1 px 判異格；完美投影 Sampson < 0.1 px、擾動 > 3 px（`:525-561`） |
| 1 配對 | `pairs` | `pairs.jsonl`、`pairs_summary.json` | `num_reg_images == 1058`、POSE_ONLY 張數 `== 12`、`n_pairs <= 25000`（`:54-55,60`、`:568-573,629`） |
| 2 VRAM bench | `bench` | `bench/rt5060.json`、`bench/verdict.json` | `peak_allocated_mib <= 7000`；`verdict.status == OOM_RISK_8GB` 直接停（`:71`、`:663-675`） |
| 3 覆蓋診斷 | `diagnose` | `diagnose/cover.json`、`diagnose/verdict.json` | `mean_r_cover >= 0.15`；`verdict.status == INSUFFICIENT` 直接停（`:79`、`:678-690`） |
| 4 密集匹配 | `match` | `matches/*.npz`、`matches/inliers/*.npz`、`matches/filter_summary.json` | 每對用凍結位姿做 Sampson `<= 3.0 px` + cheirality + 三角化角 `>= 1.0 deg` 過濾（`:61-62`、`:321-357`） |
| 5 重新三角化 | `triangulate` | `database.db`、`model/{cameras,images,points3D,rigs,frames}.bin`、`model/occupancy.json`、`database_receipt.json`、`triangulate_receipt.json` | keypoint cap 50000（OOM 退 20000）；重建後 `num_reg_images == 1058`；POSE_ONLY 影像**不得**長出 3D 觀測；`occupied_cell_fraction["2"] > 0.00495`（`:64-65,73`、`:1280-1298`） |
| 6 單幀煙霧 | `smoke` | `smoke/smoke_edm_tracks.json`（失敗另寫 `smoke/smoke_fail.json`） | `status == LOCALIZED_STRONG` **且** `r_lift >= 0.15` **且** `ransac_inliers >= 80` **且** `reprojection_p90 <= 3.0`（`:1448-1453`） |

（上表 `:NNN` 皆指 `定位演算法/EDM工具包/build/edm_fixedpose_retriangulate.py` 的行號。）

`bench`、`smoke` 與所有 `p17x_*` 實驗臂都先呼叫 `require_free_gpu()`：`nvidia-smi`
只要看到任何 compute process 就拒絕開工（`:220-225`）。

實測產物（`river-deploy-5060-20260908/.../model/occupancy.json`）：1058 張影像、
2,437,743 個 3D 點、`occupied_cell_fraction["2"] = 0.00925`（閘值 0.00495 的 1.9 倍）。

### 為什麼不能把 `map_model` 指回 GlueMap 原圖

這不是偏好問題，是閘值問題。同一支 smoke 在**未重三角化**的 GlueMap model 上跑：

```json
"baseline_smoke": {"raw_matches": 35478, "valid_2d3d": 707, "r_lift": 0.0199}
```

來源：`river-deploy-5060-20260908/river_gluemap_all8_direct_20260831/experiments/edm_fixedpose_retriangulate/model/occupancy.json:21-26`。

707 / 35478 = **1.99%** 的 EDM 匹配才能 lift 成 2D–3D 對應，離 smoke 的放行門檻
`r_lift >= 0.15`（`edm_fixedpose_retriangulate.py:1450`）差了 7.5 倍。原因是
GlueMap 的觀測是 SIFT/GlueMap 關鍵點，跟 EDM 的 detector-free 匹配落點不同格；
重三角化就是把觀測層換成 EDM 自己的落點。上游交付包也把這條寫成硬規定
（`river-deploy-5060-20260908/README.md:78`、`P174_AND_NEXT.md:98`）。

### 重跑成本

`match` 約 4 分鐘 GPU（重建 npz + inliers），`triangulate` 重建 `database.db`
（約 302 MB）與 `model/`（來源：
`river-deploy-5060-20260908/.../FINDINGS.md:300-306`）。完整一輪（含 `pairs`、
`bench`、`diagnose`）是數小時 GPU 等級的工作，不要在驗證流程裡順手觸發。

### 建圖之後

`model/` + `keyframes/` + MegaLoc bank 還不是可匯入的場域包。要再產生
`direct_bundle.json`、`direct_localizer_profile.json` 與 `site_profile.json`
五大件；欄位規格見
[`控制介面程式/site_profiles/建圖端輸出規格.md`](../../../控制介面程式/site_profiles/建圖端輸出規格.md)，
runtime 分工見
[`定位演算法/deploy_code/sfm_direct_deploy/README.md`](../../deploy_code/sfm_direct_deploy/README.md)。
