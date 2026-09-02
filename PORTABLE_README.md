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

預設任務為
`控制介面程式/mission_selections/river_gluemap_all8_direct_localization.json`。真機入口會先
驗證所有 component SHA，再產生唯讀 site-profile snapshot；不接受另一份 profile
直接覆寫解析結果。此全八段地圖只有同資料集單幀煙霧測試，尚無獨立 holdout 與
ANAFI camera-pipeline 品質證據，因此 resolver 會阻擋真機定位；同時無 route，不能 AUTO。
ANAFI 羅盤仍由真機韌體即時回讀；完成場域驗證、補畫航線並
完成校正且回讀為有效後，preflight 第一步會自動通過。AUTO 仍只會在操作員完成
其餘步驟並親自按下按鈕後，使用第三步確認時綁定的 immutable route snapshot。

AUTO 中可切回手動。關閉 UI 或終端機（Ctrl+C／SIGTERM／SIGHUP）時，backend 會先
鎖存並拒絕所有晚到的電腦移動指令，再取消 AUTO、送零 PCMD，並用不依賴 AUTO
worker 是否正常退出的路徑要求原地降落。只有 touchdown 已確認才關 UI／斷線；
否則保留介面與連線供重試。實際飛行仍須由現場安全操作員監看。

## 完整性與診斷

```bash
python3.10 tools/package_manifest.py verify --root .
SFM_LAUNCH_DRY_RUN=1 SFM_MAX_PERFORMANCE=0 ./一鍵啟動.sh
```

若要換場域，必須提供 hash-pinned mission selection 及其 vehicle、site、map、localizer、
route 與 calibration components；resolver 不允許引用套件外路徑或 symlink。
