# 執行環境（實體）

- `torch_hub_cache/` — runtime 模型與權重
- `../requirements/runtime.txt` — 主專案 direct pins
- `../requirements/runtime-lock.txt` — 正式 hash lock
- `package_git/` — 舊 runtime 包 git 歷史（可選）

正式 portable 只依 `RUNTIME_ARTIFACTS.json` 收錄 EDM checkpoint 與 MegaLoc 必要檔案。
本機 cache 中的 XFeat、BoQ 或 ResNet 備份不屬於 production payload，也不應整包複製。

Python：`/home/allen/localization/.venv/bin/python`
