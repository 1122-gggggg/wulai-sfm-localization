# 場域資產

此目錄只保留結構，不提交實際場域資料。

## 場域包（建議做法）

每個場域一包，自成一體，可整包複製到別台機器：

```
場域/<site>/
├── README.md    出處、建置日期、狀態、已知限制
├── maps/        顯示用點雲 PLY；參考位姿、COLMAP model 與相機資料依需求放入
├── bundles/     與該場域配對的 EDM localization bundle（已含 MegaLoc `ref_global`）
├── routes/      選配的顯示航線；真機用 route 另需人員審核與同座標系核准
└── reports/     建置報告、manifest、驗證數據
```

site profile 用相對於 profile 檔案的路徑指進來，例如
`../../地圖檔/場域/river_site/bundles/river_site_reloc_map_edm.pt`。
EDM 不讀取獨立 `megaloc_cache`，也不使用 legacy `track_landmarks`；這兩個
profile 欄位應維持 `null`。

**場域包裡不放演算法。** `localizer_deploy_dir` 一律指向
`定位演算法/deploy_code/sfm_glomap_deploy`，這樣換地圖不會連演算法一起換。

## 舊式扁平佈局

`maps/`、`bundles/`、`mission_routes/` 只保留給未指定 site profile 的舊式工具。
使用 site profile 時，新建航線會跟隨該 profile 已解析的場域資產，寫入
`場域/<site>/routes/`；正式多場域資料一律以 `場域/` 為準。
