# `sfm_direct_deploy` — `direct` 定位後端的部署 package

`direct` 是系統的第三個定位後端，與 `edm`、`xfeat` 並列，由
`localizer_registry.py` 依 `site_profile.localizer` 建立。它跑的是 two-rate 迴路：

```text
每一幀（CPU）                          背景（GPU，不在關鍵路徑）
  KLT 追蹤 960×540 灰階                  MegaLoc top-2 檢索
  → pycolmap absolute-pose PnP           → EDM 密集匹配 → lift 2D–3D
  → FAST_TRACK / VO_ONLY / DEAD_RECKON   → PnP reloc → handover **replace** 活點集
```

快迴路完全不碰 GPU，reloc worker 完全不碰關鍵路徑。這個分工是刻意的，不是實作
巧合：改動它就等於改動 P174 那組凍結組態的前提。

## 檔案分工

| 檔案 | 責任 |
|---|---|
| `direct_paths.py` | `vendor_sys_path()`：把 `vendor/` 冪等地掛進 `sys.path` |
| `direct_map.py` | `DirectMapAssets.load()`：讀 `direct_bundle.json`、驗 SHA-256、解析 model / keyframes / MegaLoc bank / intersection cells / depth 路徑 |
| `direct_profile.py` | `load_direct_profile()`：讀 `direct_localizer_profile.json`，回傳巢狀 frozen dataclass |
| `live_provider.py` | `LiveMapEDMProvider`：GPU 側 MegaLoc + EDM + PnP，回傳 `RelocFix` |
| `two_rate_tracker.py` | `RelocWorker`（單一背景 thread）與 `TwoRateTracker`（快迴路 + handover + VO + dead reckon） |
| `direct_localizer_adapter.py` | `DirectTrackerAdapter`：把 `TwoRateTracker` 包成 `live_localizer_worker` 認得的 tracker contract |
| `vendor/` | vendored 上游 package，見下 |
| `VENDOR_PROVENANCE.json` | vendor 來源、逐檔 SHA-256、本地 patch 清單 |

資產契約：`direct_bundle.json`（schema `direct-localization-bundle/v1`）與
`direct_localizer_profile.json`（schema `direct-deployment-profile/v1`）。欄位表見
[`控制介面程式/site_profiles/建圖端輸出規格.md`](../../../控制介面程式/site_profiles/建圖端輸出規格.md)。
建圖流程見 [`定位演算法/EDM工具包/build/README.md`](../../EDM工具包/build/README.md)。

**生產路徑不讀任何環境變數旋鈕。** P174 那組 env（`P174_AND_NEXT.md:58-65`）已經
逐項寫進 `direct_localizer_profile.json`，並由 `site_profile.asset_sha256` 綁定。
要換數值就是換一份 profile 檔 + 換一次 SHA，不是設環境變數。

## vendor 來源與搬遷規則

`vendor/river_map_quality`（36 檔）與 `vendor/sfm_diagnosis`（94 檔）是從交付包
`river-deploy-5060-20260908/software/` 複製進來的，來源路徑與逐檔 SHA-256 記在
`VENDOR_PROVENANCE.json`：

- `river_map_quality` ← `software/compatibility/river_map_quality/src/river_map_quality`
  （`VENDOR_PROVENANCE.json:59`）
- `sfm_diagnosis` ← `software/normal_map_pipeline/src/sfm_diagnosis`
  （`VENDOR_PROVENANCE.json:159`）

本地 patch 有兩筆，逐筆記在 `VENDOR_PROVENANCE.json.local_patches`（每筆帶
`upstream_sha256`；`packages.*.files` 保留的一律是上游 digest）：

1. py3.10 相容性：`river_map_quality/baseline.py` 改成 `from datetime import timezone`
   並 `UTC = timezone.utc`，因為 `datetime.UTC` 是 3.11+ 才有，而本工作區 runtime 是
   CPython 3.10.12。
2. 效能：`river_map_quality/official_edm_adapter.py` 的 `lift_reference_matches` 改成
   向量化鄰域搜尋。半徑、admission 規則與 (distance, Point3D ID) 併發序全部不變，輸出
   逐位元相同；`tests/localization/deploy/test_lift_reference_matches.py` 保留改版前的
   純量實作當 oracle 做差分比對。這筆不屬於下面規則 3 的兩類，是操作員明確核准後落地
   的，仍欠一次回上游。

搬遷回上游的規則：

1. 改 vendor 檔之前先確認那不是應該改在 `sfm_direct_deploy/*.py`（本地 runtime）
   的東西。vendor 是第三方邊界，本地邏輯不往裡面塞。
2. 真的要改 vendor，必須在 `local_patches` 加一筆（`change` + `file` + `reason`
   + `upstream_sha256` + `patched_sha256`）。`packages.*.files` 記的一律是**上游**
   digest，不隨本地 patch 改動 —— 那是 re-vendor 時比對用的基準；本地檔案的實際
   digest 由 `patched_sha256` 與 `SHA256SUMS` 各記一份。無 provenance 的改動視為
   不可追溯，等同污染第三方邊界。
3. 本地改動不限類別。py3.10 相容性、上游 backport，以及**任何能證明有收益的優化**
   都可以留在本地。條件是規則 2 的可追溯性，加上 `tests/` 裡一份能證明等價或證明
   收益的測試 —— 對不該改變輸出的優化，測試要用改版前的實作當 oracle 做差分比對。
   本地 patch 仍然欠上游一次回收：規則 4 的 re-vendor 是整包換，沒有回上游的 patch
   會被蓋掉。
4. 重新 vendor 時整包換，不做逐檔挑選，並重算 `VENDOR_PROVENANCE.json`。

## 不許做的事

以下每一條都是 `P174_AND_NEXT.md:91-98` 的明文禁令，來源是已花掉的實驗成本，
不是保守偏好：

| 禁令 | 出處 | 理由 |
|---|---|---|
| 不許再拿 P167 / P173 / P174 回頭調 `TRACK_CAP`、VO lag、PnP、`VO_MIN_LIVE`、`DEAD_RECKON` 門檻 | `P174_AND_NEXT.md:93` | 旋鈕本來就是在 P167 上選的；再用同三條路調就是對測試集過擬合。準度只能靠**一條全新航線、凍結組態只跑一次** |
| LocoTrack-S/B 不進 29 fps 快迴路 | `P174_AND_NEXT.md:94` | 73 ms GPU vs KLT 2 ms CPU；small 只縮 transformer，CNN 共用，而且會跟 EDM 搶 GPU |
| 不做光度前處理（gamma / CLAHE） | `P174_AND_NEXT.md:95` | 部署會遇到所有光線；已量過，P174（mean luma 52）不做也有 99.0% 覆蓋（`FINDINGS.md:91`） |
| `VO_WINDOW_BA` 不開 | `P174_AND_NEXT.md:96` | 更慢（loop 5.48 → 6.47 ms）而黑洞末段誤差沒比較好（0.009 → 0.013）（`FINDINGS.md:96`） |
| 不自寫 CUDA / sm_120 JIT 擴充 | `P174_AND_NEXT.md:97` | SplatHLoc 已因此停掉 |
| `map_model` 不許指回 GlueMap 原圖 | `P174_AND_NEXT.md:98`、`river-deploy-5060-20260908/README.md:78` | GlueMap 觀測上的 baseline smoke lift 率只有 707/35478 = 1.99%，遠低於 15% 放行門檻（`.../model/occupancy.json:21-26`） |

搬機（5060）上**准做**的只有機器相關量測：實測快迴路與 reloc 毫秒、用實測
`reloc.median_ms` 設 `reloc.period_s`、handover 排程改用本機延遲、必要時降
`reference_cache_size`（`P174_AND_NEXT.md:71-89`）。這些都不是準度旋鈕。

## 已知數字（不要當作公尺誤差）

P174 是三條路線裡最接近「新路徑」的一條，但**沒有絕對 GT**：99.40% 是有 pose 的
幀比例，不是公尺精度（`P174_AND_NEXT.md:9,17`）。1829 幀中 `FAST_TRACK` 1507、
`RELOC_SEED` 81、`VO_ONLY` 230、`DEAD_RECKON` 0、`NO_POSE` 11（全在冷啟動）
（`records/replay_p174_ship.summary.json` 的 `status_counts`）。live 誤差未知；
最長一段 173 幀在 reloc 回來前已漂到 0.17 地圖單位（`P174_AND_NEXT.md:52-54`）。

因此新地圖進系統時 `localizer_quality_receipt` 一律先是未通過、`flight.approved`
為 false，要等一條全新航線的一次性現場驗收才能轉正。
