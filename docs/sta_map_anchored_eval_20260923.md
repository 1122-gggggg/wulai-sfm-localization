# 地圖錨定 STA 定位：跨影片離線評估（2026-09-23）

分支 `exp/sta-localization`。只新增離線研究工具，不改 runtime、profile、飛控，
不接硬體。定位路徑完全不使用 EDM、KLT 與 PnP。

- 核心幾何：[`定位演算法/validation/sta_map_anchored.py`](../定位演算法/validation/sta_map_anchored.py)
- 評估工具：[`定位演算法/validation/sta_map_anchored_eval.py`](../定位演算法/validation/sta_map_anchored_eval.py)
- 單元測試：[`tests/localization/deploy/test_sta_map_anchored.py`](../tests/localization/deploy/test_sta_map_anchored.py)
- 產物：`outputs/analysis/sta_map_anchored_20260923/`（不進版控）

## 方法

```text
query 影像 ─ BoQ sideview bank 檢索（排除 query 自己的影片）─ top-5 參考關鍵影格
  每個參考 r：STA(query, r) → T_{q,r}（X_r = R X_q + t）、pose conf、r 的 pointmap
            γ_r = weighted median( z_map / z_STA )，只取 r 自己觀測到的地圖點像素
            C_q = C_r + R_{w,r} · γ_r t，R_{w,q} = R_{w,r} R
  位姿空間共識 → 地圖座標（GlueMap 凍結座標系）的 query 位姿
```

相對先前推導的兩處修改：

1. **γ 的尺度錨點**：不用相鄰關鍵影格的基線比，而用參考影格自己觀測到的
   EDM 重三角化地圖點的相機深度。只比 z，避免 STA 隱含焦距誤差；用加權中位數
   而非 ViSTA Eq. 11 的最小平方，因為地圖點含離群值。ViSTA 原始碼的 scale edge
   本來就只比深度（`estimate_scale_with_depth_and_confidence`）。
2. **共識要獨立**：兩個互相一致的假設必須來自不同影片，或同一影片相隔至少
   72 幀（3 個關鍵影格間距），否則不接受。

## 評估設計

- **Query**：BoQ sideview bank 的 941 張地圖關鍵影格，縮到 1280×720 模擬即時串流
  （`query_camera` 的 K 正好是關鍵影格 K 除以 2.1）。
- **參考**：同一個 bank，但排除 query 所屬影片，所以參考一定來自另一趟飛行。
- **真值**：query 在 GlueMap 模型中的凍結位姿。誤差單位是地圖單位 u。
- **EDM 基準**：正式 provider 的 EDM + lift + PnP，強制使用同一份排名的前 2 名參考
  （`reloc.top_k = 2`），`LOCALIZED_STRONG` 才算接受。
- **已知偏差**：query 自己的觀測參與過地圖點三角化。EDM 的 2D–3D lift 與 STA 的
  γ 都用到這些點，兩者都略為有利；這不是重建層級的留一驗證（vendor
  `official_edm_adapter_loo.py` 也標明同一限制）。
- **不碰保留資料**：只用建圖的 8 支影片，不用 P167/P173/P174。

### 凍結參數（程式 `PARAMS`）

| 參數 | 值 | 來源 |
|---|---:|---|
| 檢索 top-k | 5 | 共識至少需要兩個獨立假設 |
| pose conf 下限 | 0.75 | 論文 τp |
| γ 最少樣本 | 30 | |
| γ 的 log MAD 上限 | 0.2 | 深度場與地圖點不一致就拒絕 |
| 共識位置容差 | 0.10 u | |
| 共識旋轉容差 | 5° | |
| 同影片獨立間距 | 72 幀 | |
| 粗大錯誤 | > 0.3 u | |

### 判定門檻（程式 `CRITERIA`，在看任何準度結果之前寫定）

主要指標是 STA 共識接受的輸出。全部通過才算 GO（值得進入串接設計）：

| 項目 | 門檻 | 依據 |
|---|---|---|
| 位置誤差 p50 | ≤ 0.05 u | 現行 FAST_TRACK 雜訊約 0.02–0.05 u |
| 位置誤差 p95 | ≤ 0.12 u | 現行 reloc 交接修正 p95 0.054–0.120 u |
| 旋轉誤差 p95 | ≤ 5° | |
| 粗大錯誤率 | ≤ 1% | 錯誤定位比不定位危險 |
| 接受率 | ≥ 50% | |

另外報告（不影響判定）：不做共識的 top-1、依參考距離分段、與 EDM 的配對比較、
EDM 失敗時 STA 的表現、STA 各階段延遲。

## 環境

- ViSTA-SLAM commit `b13ac4415c48636f302609fdd824f927cdc0723d`，放在
  `執行環境/models/vista_slam/src`（被 `models` 規則忽略，不進版控）。
- STA 權重 `frontend_sta_weights.pth`，SHA-256
  `8b8cc1c49172731f8c8de928810aa969502411b7f263d6013951207d27fa06a3`。
- 授權：DUSt3R 衍生檔案與權重為 CC BY-NC-SA 4.0（限非商業）。
- torch 2.11.0+cu128、RTX 5060 Laptop。未編譯 curope，使用 vendor 內建的 PyTorch RoPE。
- vendor 注意力層固定呼叫 xformers；未安裝 xformers（會改動正式 venv 的 torch），
  改由評估工具注入等價的 SDPA 轉接層。vendor 原始碼未修改。
- 精度與論文相同：fp32（允許 TF32），輸入 224×224。16:9 影像只保留中央約 56% 寬度。

### 自檢（`selfcheck`，不屬於評估資料條件）

| 項目 | 結果 |
|---|---|
| SDPA 轉接層 vs vendor 純 PyTorch 注意力 | 最大絕對差 1.2e-7 |
| 像素對應 vs ViSTA 前處理（8 個點，blob 重心） | 最大 0.055 px |
| 位姿方向慣例（27 組同影片相鄰關鍵影格） | 平移方向誤差中位數：`T = ref←query` 13.5°，反向 167.9° |

同影片相鄰影格的相對旋轉很小，旋轉誤差（1.07° 對 1.44°）不足以分辨方向；
平移方向可以分辨，確認 `T_{q,r}` 把 query 座標轉到參考座標。

## 結果

（待填，產生後不回頭修改上面的參數與門檻。）

## 後續步驟

- 步驟 3（救援片段重播）需要下一場實飛錄下的定位輸入影格。
- 步驟 4（串接快迴路）視本評估結果再決定。
