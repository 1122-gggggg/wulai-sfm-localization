# Reference index 離線建置契約

`tools/build_reference_index.py` 會把一組已完成正規化的 reference descriptor
建立成可攜式 IVF index。這是純離線建置工具：它只讀指定的 JSON 與 `.npy`，不會
下載模型、存取網路或替使用者尋找其他資產。

## 輸入契約

`--descriptors` 必須是單一 NumPy `.npy` 檔案，內容為非空的二維 `float32` 矩陣。
每一列必須是有限值且 L2 norm 為 1（容許誤差 `1e-3`）。不接受 `.npz`、object
array、NaN、Inf、零向量或未正規化資料。

`--names` 必須是 JSON 陣列，含有和 descriptor 列數完全相同的非空字串；名稱必須
唯一，並且是可重現的穩定 reference identifier。建置時 index 會依名稱排序，讓
輸入列的排列不影響發布內容。

## 建置

```bash
python tools/build_reference_index.py \
  --descriptors /offline/map/reference_descriptors.float32.npy \
  --names /offline/map/reference_names.json \
  --output /offline/map/reference_index \
  --model-identity megaloc:river-site:v1 \
  --nlist 256 \
  --seed 17 \
  --kmeans-iterations 8 \
  --batch-size 4096 \
  --max-query-probes 8 \
  --max-query-candidates 4096
```

`--seed` 與其他演算法參數會寫入 `metadata.json`。在相同 Python/NumPy runtime、
相同輸入 bytes 與相同參數下，輸出的 required files 及 `SHA256SUMS.json` 應完全
相同。輸出資料夾必須不存在；工具拒絕覆寫既有 index。

建置完成的目錄包含 `SHA256SUMS.json`。site profile 的
`assets.reference_index` 應指向該 manifest，而
`asset_sha256.reference_index` 綁定 manifest 本身；runtime CLI 的
`--reference-index-sha256` 傳遞同一個值。runtime 會再驗證 manifest 內每個檔案的
SHA-256、model identity、dimension、
descriptor count 與 names 集合。把整個 index 目錄與 manifest 一起放入離線 portable
asset bundle，不要只複製單一 `.npy`。

## 失敗處理

任何輸入格式、shape、finite、norm、名稱數量或 output collision 問題都回傳非零
狀態，且不發布部分 index。`reference_index.py` 使用暫存目錄與 atomic rename；建置
失敗時暫存內容會清除。完成後仍應以現有 site profile/preflight 與 portable manifest
驗證流程檢查整包資產。
