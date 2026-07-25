# 場域資產

此目錄只保留結構，不提交實際場域資料。

## 場域包（建議做法）

每個場域一包，自成一體，可整包複製到別台機器：

```
場域/<site>/
├── README.md    出處、建置日期、狀態、已知限制
├── maps/        點雲 PLY、參考位姿、COLMAP model、相機內參
├── bundles/     與該地圖配對的 localization bundle、MegaLoc cache
├── routes/      經人員審核、與地圖同座標系的安全航線
└── reports/     建置報告、manifest、驗證數據
```

site profile 用相對於 profile 檔案的路徑指進來，例如
`../../地圖檔/場域/river_site/bundles/river_site_reloc_map_edm.pt`。

**場域包裡不放演算法。** `localizer_deploy_dir` 一律指向
`定位演算法/deploy_code/sfm_glomap_deploy`，這樣換地圖不會連演算法一起換。

## 舊式扁平佈局

`maps/`、`bundles/`、`mission_routes/` 保留給只有單一場域的簡易用法，
也是 `workspace_layout.py` 的預設路徑。多場域請用 `場域/`。
