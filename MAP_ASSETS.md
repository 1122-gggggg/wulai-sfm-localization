# 可下載的場域資產

大型地圖與模型不放入 Git 歷史，請從 [map-assets-2026-07-20 release](https://github.com/1122-gggggg/wulai-sfm-localization/releases/tag/map-assets-2026-07-20) 下載。每個檔案都是 `tar.zst`，可用 `tar --zstd -xf <檔名>` 解壓。

| 場域 | 下載檔 | SHA-256 | 內容 |
|---|---|---|---|
| 烏來 | [wulai-localization-assets-20260720.tar.zst](https://github.com/1122-gggggg/wulai-sfm-localization/releases/download/map-assets-2026-07-20/wulai-localization-assets-20260720.tar.zst) | `912ccca14cbfc309dba79ae2bd73239c8cc62772b08d0f80af8ea7d8e0a65aba` | updated 點雲、定位 bundle、MegaLoc cache、TRACK landmarks、內參與驗證報告。 |
| 河濱 | [river-edm-localization-assets-20260720.tar.zst](https://github.com/1122-gggggg/wulai-sfm-localization/releases/download/map-assets-2026-07-20/river-edm-localization-assets-20260720.tar.zst) | `018806b0ee5178de40c6279e1ee95ce0f460005b424adac92bacd7c6c4425f0f` | EDM bundle、兩份點雲、reference poses、EDM 權重與封裝設定。 |

範例：

```bash
curl -LO https://github.com/1122-gggggg/wulai-sfm-localization/releases/download/map-assets-2026-07-20/river-edm-localization-assets-20260720.tar.zst
echo '018806b0ee5178de40c6279e1ee95ce0f460005b424adac92bacd7c6c4425f0f  river-edm-localization-assets-20260720.tar.zst' | sha256sum -c -
tar --zstd -xf river-edm-localization-assets-20260720.tar.zst
```

烏來封包內的定位 bundle 是既有 XFeat 資產；若要使用本倉庫固定的 EDM 正式設定，請用烏來點雲與參考資料建立對應的 EDM bundle，不能混用不同 matcher 的 bundle。河濱封包則包含 EDM bundle。
