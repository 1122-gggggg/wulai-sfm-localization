# Localization 可攜真機套件

## 相容範圍

此套件可搬到相容的另一台電腦，不是跨作業系統的單一執行檔。已固定的目標為：

- Linux x86_64
- CPython 3.10 與 `venv` 支援
- 與包內 CUDA 12.8 wheels 相容的 NVIDIA 驅動
- 系統套件：`ffmpeg`、`python3-tk`、X11 或 XWayland
- Parrot ANAFI 與 SkyController 3 的 USB／網路連線

建議在搬移前保留至少 12 GiB 可用空間，供 package-local `.venv` 安裝使用。

## 一鍵啟動

```bash
./一鍵啟動.sh
```

首次執行會先驗證 `MANIFEST.tsv`、runtime artifacts、場域 assets 與離線
wheelhouse，再從包內 wheels 建立 `.venv`。不會連線到 Python package index。
建立完成的 `.venv` 會綁定目前 package/manifest identity；若套件被更新、搬入既有
`.venv`，或缺少 `PORTABLE_PACKAGE.json`，一鍵啟動會拒絕執行，請移除舊 `.venv`
後由同一個套件重新建立。
啟動 UI 本身不會起飛；起飛只能由操作員在 UI 完成四步 preflight 後觸發。

預設場域為 `控制介面程式/site_profiles/river_site_edm.json`。AUTO 使用第三步
確認時綁定的 immutable route snapshot；起飛懸停後才等待 TRACK、pose freshness、
inliers 與 reprojection gates，全部通過後才沿路線移動。

AUTO 中可切回手動。關閉 UI 或終端機時程式會先停止 AUTO、送出零 PCMD，並要求
原地降落；實際飛行仍須由現場安全操作員監看連線與飛機狀態。

## 完整性與診斷

```bash
python3.10 tools/package_manifest.py verify --root .
SFM_LAUNCH_DRY_RUN=1 SFM_MAX_PERFORMANCE=0 ./一鍵啟動.sh
```

若要換場域，必須把 profile 及其 SHA 綁定的地圖、重力對齊、定位 bundle、
reference poses 與 route 一起匯入；profile 不允許引用套件外路徑或 symlink。
