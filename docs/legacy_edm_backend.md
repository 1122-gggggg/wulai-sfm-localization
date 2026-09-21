# 舊 EDM 後端歷史紀錄

以下為已移除 `.pt` 後端的實驗與配置紀錄，不是目前 direct 後端的啟動指引。
目前接口見 [根目錄 README](../README.md)；安全策略見 [SAFETY.md](../控制介面程式/SAFETY.md)。

## 固定的 EDM 正式參數

> **這一節描述的是 `sfm_glomap_deploy`（EDM `.pt` bundle）後端，該後端已從 worktree
> 移除，`定位演算法/configs/edm_production_profile.json` 也不再存在。**
> `localizer_registry` 現在只註冊 `direct`，其凍結參數在 release 內的
> `localization/direct_localizer_profile.json`，實測與已落地的優化見
> [`docs/direct_backend_ledger.md`](direct_backend_ledger.md)。以下保留為歷史紀錄。

`定位演算法/configs/edm_production_profile.json` 是共用的正式設定：1024×576、
PyTorch CUDA FP16、coarse top-k 3225、confidence 0.2、reference tensor cache 32、
TRACK/WEAK/LOST top-k 1/3/5、BOOT BoQ top-k 10（先驗證前 2 張，不足才展開）、
batch size 2、LOST grace 2、recovery bank/scan 192/2、correspondence 上限 900、
inliers 80/50/30。BoQ 全域檢索在 BOOT 執行；LOST 預設每個 episode 一次，場域 profile
可設定週期重試。共用預設的 temporal reference 關閉，PnP acquire/track/RANSAC gate 固定為 5/6/5，
capture-time 預測上限為 0.25 秒。

**追蹤器模式：同步（`SFM_EDM_ASYNC_TRACKER` 預設 `0`）。** 每一個發佈出去的 pose 都是當幀
重新對上地圖的 EDM+PnP 結果。2026-09-04 曾把 async fast/slow 解耦轉為預設（fast path 用光流
帶 pose、slow path 才跑 EDM），2026-09-05 以七段 720p 語料庫複查後改回：async 的「成功」有
88% 是沒有當幀視覺確認的光流推算（3,587 個成功幀裡只有 429 幀真的重新對上地圖），
最長一段連續 710 幀（約 89 秒）；七段成功率也從 sync 的 80.6% 掉到 62.9%。設 `=1` 可換取
p50 約 26 ms → 5 ms 的延遲，代價是上述未確認推算。詳見
`docs/verified_localization_optimization_ledger.md` 的 2026-09-05 章節。

### coarse tail 融合（不改任何參數）

matcher 的 kernel 實作換過，**參數一個都沒動**。上游 coarse head 會實體化一個
9216×9216 的 fp32 confidence matrix（每個 batch element 324 MiB），分成 exp、兩次
L1 normalize、相乘四個 kernel，之後再讀一次做 row max。`edm_matcher.py` 在
`_import_edm()` 內把這段換成一個 `torch.compile` 融合的 reduction，矩陣不再落地。

RTX 5060 Laptop、真實 720p 影格實測：b=1（TRACK，一張 reference）41.52 → 25.57 ms，
峰值記憶體 1457 → 494 MiB；b=2（WEAK/LOST 的 reference 配對）89.50 → 57.36 ms。
選出的 match **集合完全相同**（3095/3095 共同、0 只在單邊；逐列 argmax 0/3225 不一致；
mconf 最大差 4.8e-07），只有 `torch.topk` 在 ~5e-7 等值處的 tie-break 順序不同。

`SFM_EDM_FUSED_COARSE=0` 可退回上游實作。編譯產物存在 `執行環境/inductor_cache/`
（可重建，冷啟約 5 秒，已排除版控）；`SFM_EDM_COARSE_WARMUP_BATCHES`（預設 `1,2`）
控制啟動時預熱哪些 batch size — 一次只進一張 query，所以 batch 維度是 reference 數，
上限就是 `match_batch_size`。

**但有五個欄位是地圖尺度相依的**，不能跨場域共用：`radius`、`max_jump`、
`adaptive_jump_floor` / `_bootstrap` / `_ceiling`，係數分別是
0.16 / 0.40 / 0.0006 / 0.004 / 0.0016 乘上該場域的
`S = 2·p95(‖center − componentwise_median‖)`。共用檔的值是照 urai 的尺度定的
（S = 5.007236），其他場域沒重算就會鬆掉：

| 場域 | refs | S |
|---|---:|---:|
| urai | 1383 | 5.007236 |
| river_site | 454 | 1.900843 |
| river_site B0+P116/P117 | 340 | 1.470463 |


需要校正時在 `定位演算法/configs/edm_profiles/<site>.json` 建一份場域專屬 profile。

## 模擬實機串流

錄影檔的碼率是實機無線鏈路的 5–12 倍，直接 replay 會高估定位表現。
`ANAFI_LINK_SIM=1` 會依 ANAFI v1.4 白皮書 §5.2 的串流契約重新編碼
（720p、H264 main profile、5 Mb/s、45 slices × 16 px、periodic intra-refresh）：

```bash
ANAFI_LINK_SIM=1 ANAFI_LINK_LATENCY_MS=280 ANAFI_LINK_LOSS_PCT=1.0 \
  ./控制介面程式/影片模擬串流/啟動.sh <video>
```

`SFM_HOLD_ON_LOW_CONF=1` 為精度優先模式：連續低信心即暫停串流（等同懸停），
held frame 改走 LOST recovery（提高 local top-k + 依場域 profile 排程 BoQ 全域檢索）。
預設開啟。實機端對應的是 `SFM_GATE_WEAK`（預設開啟，WEAK fix 直接 hover）。
