# 定位演算法（實體）

| 路徑 | 角色 |
|---|---|
| `deploy_code/sfm_glomap_deploy/` | UI worker 使用的唯一部署實作 |
| `flight_control/` | 飛控與必要的 deploy 相容鏡像 |
| `configs/` | production 設定 |
| `validation/` | benchmark、replay、品質 gate 與研究實驗 |
| `pipeline/` | 定位 pipeline 工具 |

候補方法不再另建頂層 symlink 索引，權威路徑為：

- ONNX/TensorRT matcher：`deploy_code/sfm_glomap_deploy/edm_onnx_matcher.py`
- NeuFlow refresh：`validation/neuflow_refresh_experiment.py`
- projection-guided tracker：`validation/projection_guided_tracker.py`

鏡像一致性由 `validation/check_runtime_mirrors.py` 檢查。
