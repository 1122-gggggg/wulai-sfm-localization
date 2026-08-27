# 定位 runtime

本目錄擁有可部署的定位 runtime：EDM/XFeat tracker、bundle loader、artifact integrity、
MegaLoc cache 與共用 pose 型別。UI worker 與離線 replay 直接從這裡匯入。

飛控、安全監控、Olympe frame source 與人工控制工具的唯一 owner 是
`定位演算法/flight_control/`。本目錄不再保存 `path_follow_flight.py`、
`olympe_frame_source.py` 或其他飛控相容副本；需要兩邊共用的模組也只能有一份。
所有權由下列純地面檢查強制：

```bash
python 定位演算法/validation/check_runtime_mirrors.py
```

EDM 模型的離線檢查與短重播：

```bash
python 定位演算法/validation/offline_model_smoke.py --model edm
python 定位演算法/validation/benchmark_edm_site_replay.py \
  --require-cuda --max-frames 3 \
  --site-profile 地圖檔/場域/river_site/site_profile.json \
  --video /path/to/P1190119.MP4 --out /tmp/edm_gpu_smoke.json
```

完整重播與 PnP correspondence cap A/B 使用同一支
`benchmark_edm_site_replay.py`，分別傳入 `--max-corr-total 450`、`600`、`750`、
`900`。結果保存影片、site profile、bundle 與 localizer profile 的 SHA-256，
以及逐幀拒絕原因、跳躍確認、品質與延遲資料。

自主航線入口永久鎖定；起飛只能由現場操作員在桌面 UI 親自執行。

泛化政策：

- 選參只使用 flight-disjoint 的 P116-P120 開發 fold。
- P157/P167 在 freeze 之後只做回歸，不得當 tuning 來源。
- 接受條件是 macro 提升，且 worst-fold 不回退。
- 確定性 2D appearance / image-plane stress 只是補充證據，不能證明 3D
  視差泛化。
- 沒有新的 held-out 航線與對應 ground truth，就不得宣稱任意 3D 視角覆蓋。
