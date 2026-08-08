# 定位演算法（實體）

| 路徑 | 角色 |
|---|---|
| `deploy_code/sfm_glomap_deploy/` | UI worker 使用的定位 runtime 與共用型別 owner |
| `flight_control/` | 飛控、安全監控、Olympe 串流與人工工具 owner |
| `configs/` | production 設定 |
| `validation/` | benchmark、replay、品質 gate 與研究實驗 |
| `pipeline/` | 定位 pipeline 工具 |

候補方法不再另建頂層 symlink 索引，權威路徑為：

- ONNX/TensorRT matcher：`deploy_code/sfm_glomap_deploy/edm_onnx_matcher.py`
- NeuFlow refresh：`validation/neuflow_refresh_experiment.py`
- projection-guided tracker：`validation/projection_guided_tracker.py`

兩個 runtime 目錄不提交同名 Python 副本；入口依賴的共用模組直接從其唯一 owner
匯入。`validation/check_runtime_mirrors.py` 檢查 owner 存在、舊副本未回流，且沒有
未分類的同名 runtime 檔案。
