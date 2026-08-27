# 模擬器（實體）

三種工具保留不同責任，不互相冒充完整系統驗證：

| 工具 | 用途 | 不取代 |
|---|---|---|
| `測試影片/` + `控制介面程式/影片模擬串流/` | EDM、720p 串流與操作 UI 回歸 | Sphinx firmware／PCMD 物理響應 |
| `parrot_stimulate/` | 主要 Sphinx/Olympe PCMD、逐點 route、安全 gate、response 與固定 worst-case 驗證 | EDM 影像定位、真機安全核准 |
| `sphinx_anafi_path_convergence/` | 保留十種 convergence controller 的比較與歷史回歸 | 正式 waypoint controller 或 production readiness gate |

`parrot_stimulate/src/anafi_pcmd_sim/scale_free_control.py` 是 Python 3.10
操作介面與 Python 3.11 Sphinx route 共用的純控制核心；真機 autonomous 仍保持鎖定。
完整無飛行驗證由根目錄 `./驗證系統.sh` 一次執行兩個 Python 環境。

整合只使用 `parrot_stimulate/` 的程式、鎖檔、固定 firmware 與測試。
原 `封存/parrot_stimulate_standalone_20260803/` 已於 2026-08-05 經操作員明確
授權永久刪除；它原本不納入外層 Git，因此目前不能由此 repository 回復。
`parrot_stimulate/` 現在是工作區內唯一的正式 Sphinx/Olympe 模擬器來源。

`launch_sphinx_anafi_empty.sh`、`sphinx_path_follow_smoke.py` 為舊 harness 的相容入口，
不應再新增第三套控制公式。
