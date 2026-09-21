# 專案品質檢查與改善（2026-09-22）

## 範圍與結論

針對整個 repository 執行第一方 Python 靜態檢查、既有品質門檻、離線測試、依賴安全
檢查、模組所有權及工作區檢查，再聚焦修正失敗項目。保留開始時已有的 6 個本機提交、
24 個已修改檔案及尚未追蹤的實驗／測試／文件。第三方 vendor、私有地圖及模型不做
全面重寫。本次新增修正未變更真機起飛、降落、按住微移或 Esc 交接語意。

專案原本已有相依 hash lock、CI、回歸測試、發布完整性與飛行安全邊界，但近期修改
使型別、複雜度、格式及部分模擬測試失效。這次恢復既有門檻，並讓開發流程及目前
支援的後端資訊更清楚。通過品質門檻不代表沒有缺陷，也不等同真機驗收。

## 可量測的改善

以下為 Ruff C901 超過 10 的 production 函式數／該區最大複雜度。保留原預算，
沒有新增忽略規則或提高上限：

| 區域 | 修改前 | 修改後 | 原有上限 |
|---|---:|---:|---:|
| tools | 11 / 41 | 5 / 20 | 5 / 20 |
| deploy | 2 / 16 | 2 / 13 | 2 / 13 |
| flight | 7 / 28 | 2 / 12 | 3 / 12 |
| validation | 3 / 17 | 1 / 13 | 1 / 15 |
| control | 29 / 25 | 25 / 25 | 25 / 25 |

`olympe_live_backend.py` 原為 5,317 行，超過 5,234 行上限。將唯讀遙測記錄與
影像交付診斷移至 `backend_telemetry.py` 後降至 5,201 行；飛行命令仍在原 owner。
新模組加入共用格式檢查與發布收據清單。

## 修正內容

- **責任拆分**：分離抵風積分的時鐘／量測／累積、降落確認的樣本接納、定位交接
  的 frame walk、模擬定位的交付與記錄、實驗評分、航線驗收情境及模擬遙測／hooks。
- **型別與協定**：修復 optional 速度樣本索引與 Pose 回傳型別；簡化 worker header
  大小選擇及錯誤 payload 欄位讀取，保留各種協定版本的既有驗證。
- **可重現模擬**：修正只把位置加速四倍、時間戳卻不變的 plant，補齊速度向量。
  完整路線回歸改用虛擬時鐘及正式控制器，14 項 workflow 測試約 0.6 秒完成。
  不靠縮短安全確認視窗加速；失敗時也會停止測試 worker。
- **驗收正確性**：修正最後一次陣風超過 5 秒恢復時限仍可能被接受的漏洞，加入
  5.0／5.1 秒邊界回歸。離線驗收沿用正式 `AUTO_MAX_DURATION_S`，避免仍使用舊
  300 秒任務預算；完整私有航線矩陣的 wall-time 預算隨案例數計算。
- **文件與發布**：補充開發指南；修正測試收集範圍、目前唯一的 direct 後端與
  安全策略引用。舊 EDM 說明移至歷史文件。登記既有航線稽核輸出分類，更新 source
  manifest 與 SHA256SUMS。修正 source-only 誤收錄本機模型與韌體，讓 source
  完整性可在缺少這些資產的乾淨 checkout 驗證；portable 仍校驗其實際資產。

## 驗證紀錄

修改前主測試為 **2,563 passed / 4 failed / 2 skipped**。兩個失敗為不收斂的 UI
模擬路線；另外兩個為私有航線數增加後的 wall-time 上限及舊 300 秒預算。

本次已通過的針對性檢查：

- 飛控、定位交接、實驗濾波、worker 與操作後端：1,073 passed / 1 skipped。
- 模擬與報告工具的針對性驗證：57 passed。
- 完整私有場域航線驗收：20 passed。
- 發布與 source/portable 資產分界回歸：63 passed。
- 22 個 shell 腳本語法檢查通過。
- 獨立 Python 3.11 Parrot 子專案：108 passed，lint/format 通過。
- mypy：21 個既有 typed boundary 通過。
- 既有 Ruff lint/security、複雜度、module ownership、純離線 flight selftest 與
  `pip check` 通過。依賴安全 gate 通過，沿用原本有期限的例外；不宣稱所有依賴
  都不存在漏洞。audit 與 SBOM 留在被忽略的 `outputs/security/`。

最後完整離線測試：**2,573 passed / 2 skipped，0 failed**，256.33 秒。
整體 coverage（含分支）為 **58.93%**，通過既有 50% 下限：

```bash
.venv/bin/python -m pytest -q --timeout=300 -m 'not hardware' \
  --cov --cov-config="$PWD/pyproject.toml" --cov-report=term
```

coverage 設定須使用絕對路徑，讓改變工作目錄的子程序仍沿用相同 branch 設定。
快速驗證入口另通過 87 個 smoke tests。完成前以暫存區產生乾淨 Git archive，
在沒有本機模型／韌體 payload 的 archive 中執行 source manifest 驗證並通過。
新提交不增加大型二進位檔；受保護的 8 個 live 命令／清理函式 AST 與原始版本一致。

## 邊界

仍有 35 個超過 C901=10 的函式，受既有分區預算約束；Tk 主程式與 live backend
仍屬大型模組。它們不是這次全部重新設計完成的模組，後續拆分應搭配相應契約測試。
型別檢查有明確範圍，第三方 vendor 不受第一方格式門檻約束。

本次不執行真機操作，也不重新驗證完整 GPU 推論、GUI、乾淨 portable 安裝或實飛
物理精度。這些需使用受核准的資產與相應環境另行驗證。Git 推送保存原始碼、設定、
測試與文件，不包含 `.gitignore` 排除的模型、影片、場域資料與飛行紀錄。
