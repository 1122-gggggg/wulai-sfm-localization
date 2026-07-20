# 場域地圖定位系統

這個倉庫提供可重用的定位演算法與操作介面。每個新場域都必須自行提供與該場域一致的資產；本倉庫不含任何實際場域的點雲、定位 bundle、影片、飛行紀錄或模型權重。

## 目錄

```
定位演算法/       定位器、飛控鏡像、驗證工具與 EDM runtime 原始碼
控制介面程式/     操作 UI、串流入口、場域設定與任務工具
地圖檔/            使用者放置場域地圖與定位 bundle 的位置（不納入 Git）
```

## 新增場域

1. 把點雲地圖放到 `地圖檔/maps/`，定位 bundle 放到 `地圖檔/bundles/`。
2. 以 `控制介面程式/site_profiles/example_site_edm.json` 為範本建立自己的 site profile，填入相對路徑、相機內參與對應的定位資產。
3. 將 EDM 權重放到 `定位演算法/deploy_code/runtime/EDM/weights/`，並依該目錄的 `README.md` 安裝相依套件。
4. 離線影片驗證時，指定 profile：

   ```bash
   SFM_SITE_PROFILE=/absolute/path/to/your_site.json \\
     ./控制介面程式/影片模擬串流/啟動.sh /absolute/path/to/video.mp4
   ```

真機入口不會選擇預設場域。只有在 profile 的地圖、定位 bundle、相機內參與安全航線位於同一座標系，且已由操作人員完成安全驗證後，才能進行真機操作。

## 固定的 EDM 正式參數

`定位演算法/configs/edm_production_profile.json` 是所有場域共用的正式設定：1024×576、PyTorch CUDA FP16、coarse top-k 3225、confidence 0.2、reference tensor cache 32、TRACK/WEAK/LOST top-k 1/3/5、BOOT MegaLoc top-k 10（先驗證前 2 張，不足才展開）、batch size 2、LOST grace 12、recovery bank/scan 192/2、correspondence 上限 900、inliers 80/50/30。MegaLoc 只在 BOOT 與每個 LOST episode 各執行一次，temporal reference 關閉，PnP acquire/track/RANSAC gate 固定為 5/6/5，capture-time 預測上限為 0.25 秒。

## 注意事項

「地圖」在本系統中不只是 PLY 點雲。定位還需要由相同場域建置出的 localization bundle，以及和影片/相機一致的內參。EDM 執行也需要另行取得模型權重。

起飛、降落與即時飛行控制僅能由現場操作人員在 UI 執行。請先使用離線影片與地面測試驗證新場域設定。
