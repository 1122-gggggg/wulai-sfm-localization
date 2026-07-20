# 場域地圖定位系統

這個倉庫提供可重用的視覺定位、操作介面、任務編修與 ANAFI 安全整合。每個新場域必須自行提供一致的場域資產；本倉庫不含任何實際場域的點雲、localization bundle、影片、飛行紀錄或模型權重。

系統架構、目錄所有權與相容鏡像規則見 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

## 目錄

```text
定位演算法/       定位 runtime、飛控、安全邏輯、驗證工具與 EDM 上游程式
控制介面程式/     site profile、操作 UI、串流入口與任務工具
地圖檔/            使用者放置地圖、bundle 與 mission route 的位置（不納入 Git）
```

## 新增場域

1. 把點雲地圖放到 `地圖檔/maps/`，定位 bundle 放到 `地圖檔/bundles/`。
2. 以 `控制介面程式/site_profiles/example_site_edm.json` 為範本建立 site profile，填入相對路徑、相機內參與對應定位資產。
3. 將 EDM 權重放到 `定位演算法/deploy_code/runtime/EDM/weights/`，並依該目錄的 `README.md` 安裝相依套件。
4. 先以離線影片驗證：

   ```bash
   SFM_SITE_PROFILE=/absolute/path/to/site.json \
     ./控制介面程式/影片模擬串流/啟動.sh /absolute/path/to/video.mp4
   ```

5. 以統一任務入口執行編修與地面驗證：

   ```bash
   python 控制介面程式/mission_pipeline.py \
     --site-profile /absolute/path/to/site.json \
     --mode dry-run
   ```

正式任務模式不會猜測預設場域。未指定 profile 時會 fail closed；舊 per-asset 參數僅能在明確加入 `--allow-legacy-assets` 的遷移作業中使用。只有地圖、bundle、相機內參與安全航線位於同一座標系，且操作人員已完成安全驗證後，才能進入真機流程。

## 固定的 EDM 正式參數

`定位演算法/configs/edm_production_profile.json` 是所有場域共用的正式設定：1024×576、PyTorch CUDA FP16、coarse top-k 3225、confidence 0.2、reference tensor cache 32、TRACK/WEAK/LOST top-k 1/3/5、BOOT MegaLoc top-k 10（先驗證前 2 張，不足才展開）、batch size 2、LOST grace 12、recovery bank/scan 192/2、correspondence 上限 900、inliers 80/50/30。MegaLoc 只在 BOOT 與每個 LOST episode 各執行一次，temporal reference 關閉，PnP acquire/track/RANSAC gate 固定為 5/6/5，capture-time 預測上限為 0.25 秒。

## 驗證

```bash
pytest -q
python 定位演算法/validation/check_runtime_mirrors.py
python 控制介面程式/mission_pipeline.py --mode flight-selftest
```

## 注意事項

「地圖」不只是 PLY 點雲。定位還需要同場域建置的 localization bundle，以及和影片／相機一致的內參；EDM 也需要另行取得模型權重。

起飛、降落與即時飛行控制只能由現場操作員在桌面 UI 執行。任何 agent 或自動化都不得代為起飛。請先完成離線影片、地面、模擬與拆槳測試。
