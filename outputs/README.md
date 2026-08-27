# outputs 治理

`outputs/` 只保存本機產生的驗證、實驗與操作證據，不是程式碼、場域資產或可攜式
發布包。Git 忽略其內容，唯一保留的版本控制檔案是本 README；不要把地圖、bundle、
影片、模型權重或 virtualenv 放進這裡。

## 頂層分類

`tools/workspace_audit.py` 會對 `outputs/` 的每個頂層項目分類。正式發布前使用：

```bash
python tools/workspace_audit.py --strict-output-names
```

- `flight_logs/`：操作與飛控 session 紀錄。這些資料屬安全證據，不得由驗證腳本
  自動刪除或覆寫。
- `validation/`、`validation_receipts/`、`production_stream_bench/`、`security/`：
  系統驗證、portable 驗證、效能收據、dependency audit 與 SBOM；保留收據 JSON
  及其對應的 log 目錄。
- `edm_`、`exact_latency_`、`localization_fps_`、`onnx_flow_`、`optimization_`、
  `regression_`、`reverse_topk_`、`video720_` 開頭的項目：實驗證據。
- `audit_YYYYMMDD` 與已登記的治理文件名稱：稽核或決策鏈證據。

新增頂層輸出時，應先放入上述既有分類；若確實需要新類別，先在
`tools/workspace_audit.py` 的分類表加入明確規則及測試，再產生輸出。未分類項目會在
strict audit 中失敗，避免產物悄悄脫離保留與審查規則。

## 收據與保留

- 驗證收據使用 UTC 時間戳，例如 `validation_20260809T045547368091Z.json`，不可
  手動修改內容來宣稱通過；重跑驗證以新收據為準。
- 收據中的 log、manifest、SHA-256、commit 與 dirty 狀態必須保持可追溯關係。
- 稽核是唯讀操作，不會替操作者清理大型檔案。清理 flight log、validation receipt
  或實驗證據前，先由操作者確認保留期限及備份狀態。
- 若磁碟空間不足，先停止建立大型產物，再由操作者處理；不要為了讓 audit 變綠而
  刪除資產或降低門檻。

## 發布邊界

可攜式發布包的 authoritative `MANIFEST.tsv` 與 `SHA256SUMS` 位於發布包根目錄，
由根目錄的 `tools/package_manifest.py` 產生／驗證。`執行環境/` 是 legacy runtime
資料夾，不攜帶第二套 manifest 或 digest；除本治理 README 外，outputs 內容不應被
加入發布 manifest。

workspace audit、portable clean-install 與 UI smoke 都必須維持純地面／模擬邊界；
本目錄的證據管理不會授權或觸發真機飛行。
