# Deploy Localization To Another PC

## Minimum Runtime Package

For localization only, copy these as real files. Use `rsync -aL` so symlinks in
`bundles/` and `maps/` are dereferenced.

Required:

- `定位/mission/operator_interface/`
- `定位/mission/flight_control/`
- `定位/configs/`
- `定位/deploy_code/sfm_glomap_deploy/` with the runtime tracker/localizer files
- `定位/bundles/current_reloc_map_updated_v3.pt`
- `定位/maps/current_realrgb_v3.ply`
- `定位/deploy_code/sfm_glomap_deploy/map_intrinsics.json`
- package-root `torch_hub_cache/` (XFeat/LighterGlue plus the pinned 915MB
  MegaLoc safetensors file) — required offline and loaded by relative path
- package-root `MANIFEST.tsv`, `SHA256SUMS`, and `tools/package_manifest.py`
- a pinned dependency list for the runtime interpreter (see
  `requirements_runtime.txt` in the transfer package; the validated pin is
  CPython 3.10.12 with pycolmap 4.0.4)

Recommended fallback/debug:

- `定位/bundles/base_reloc_map_xfeat_tri.pt`
- `定位/bundles/base_megaloc_cache_v3.npz` containing exactly `desc` and
  ordered `names` (preferred). A legacy NPY is accepted only with a binding JSON
  sidecar containing ordered names, shape, cache SHA-256, and names SHA-256.
- `定位/maps/base_glomap_fused_0/`
- `定位/pipeline/`
- `定位/validation/benchmark_production_stream.py`

The pinned environment is CPython 3.10.12. The operator worker uses
`SFM_LOCALIZER_PYTHON` when explicitly set, otherwise the UI process's current
`sys.executable`; it no longer prefers an unrelated `/usr/bin/python3.12` merely
because that path exists.

## Localization / HLoc Tuning Package

If the target PC will also tune the localization stack, copy more than the
minimum runtime package. The needed package depends on how deep the tuning is.

### Runtime Parameter Sweeps Only

This is enough for changing matcher/state-machine parameters such as
`matcher_mode`, `nn_min_score`, `local_topk`, `weak_local_topk`,
`xfeat_topk_track`, PnP thresholds, temporal cache settings, and whether to use
NN-first or LighterGlue-first matching.

If XFeat stays fixed and only the matcher changes between LighterGlue, NN, or
`nn_then_lg`, this is the correct package. The existing relocation bundle already
contains the XFeat descriptors and 2D-3D anchors; switching to NN matching does
not require rebuilding the map or copying the full source tree.

Copy:

- everything in `Minimum Runtime Package`
- `定位/validation/benchmark_production_stream.py`
- `定位/validation/eval_stream_core.py`
- `定位/deploy_code/sfm_glomap_deploy/map_intrinsics.json` for
  `benchmark_production_stream.py`; the 938MB sparse model is not required for
  runtime parameter sweeps
- `定位/outputs/downloads_validation_20260702/` if you want the previous
  baseline reports and camera-position PLY outputs for comparison

Do not copy `定位/validation/source_videos/` unless the target PC also needs to
rerun offline validation. For field localization or matcher tuning with a live
stream, the validation videos are not required.

This level does not need the full `定位/source/` directory.

### Rebuild Or Replace The Relocalization Bundle

Copy this level if you may rebuild the XFeat/LightGlue/HLoc bundle, switch the
reference layer, regenerate 2D-3D anchors, or compare the updated bundle against
the original base map.

Copy:

- everything in `Runtime Parameter Sweeps Only`
- `定位/maps/base_glomap_fused_0/`
- `定位/maps/base_images_fused/`
- `定位/maps/current_update_workspace_v3/`
- `定位/bundles/base_reloc_map_xfeat_tri.pt`
- a canonical `{desc,names}` NPZ, or a legacy NPY plus its required binding JSON
- `定位/source/sfm_glomap/deploy/production_xfeat_tracker.py`
- `定位/source/sfm_glomap/deploy/reloc_localizer_xfeat.py`
- `定位/source/sfm_glomap/deploy/build_reloc_map_xfeat_tri.py`
- `定位/source/sfm_glomap/deploy/augment_reloc_bundle_tracking.py`
- `定位/source/sfm_glomap/deploy/benchmark_production_stream.py`
- `定位/source/sfm_glomap/deploy/map_intrinsics.json`
- `定位/source/sfm_glomap/deploy/megaloc_ref_desc_glomap_fused_322.npy`
- `定位/source/sfm_glomap/deploy/megaloc_ref_desc_glomap_fused_322.json`

This level is usually enough for HLoc-style localization experiments without
copying the full historical `定位/source/` tree.

### Full Research / Debug Copy

Only copy the whole `定位/source/` tree if you expect to inspect old experiments,
legacy bundles, rendering/simulator trials, or abandoned reconstruction outputs.
It is large and not needed for normal deployment or parameter tuning.

Do not copy for pure localization:

- `定位/source/`
- `定位/outputs/`
- `定位/validation/source_videos/`
- `../建圖/`
- `../更新地圖/`
- `/media/cihcilab/新增磁碟區/label_system/`

## Operator Interface

If using the GUI with point-cloud view and video stream, also copy:

- `定位/mission/operator_interface/flight_operator_app.py`
- `定位/mission/operator_interface/live_localizer_worker.py`
- `定位/mission/operator_interface/flight_operator_interface.html`
- `定位/maps/current_realrgb_v3.ply`

## Object Detection Runtime

If running YOLO detection together with localization, also copy:

- `定位/object_detection/configs/power_equipment_detector_runtime.yaml`
- `定位/object_detection/models/power_equipment_yolo26n_640_best.pt`
- `定位/object_detection/models/power_equipment_yolo26n_640.onnx`
- `定位/object_detection/models/power_equipment_yolo26n_640_fp16.engine`
- `定位/mission/operator_interface/object_detector_worker.py`

The TensorRT engine was built on an RTX 5090. On another GPU/driver, rebuild the
engine from `.pt` or `.onnx` if it fails to load.

## Real Drone Flight

For field flight with Parrot ANAFI / Olympe, also copy:

- `定位/mission/mission_pipeline.py`
- `定位/mission/configs/mission_defaults.json`
- `定位/mission/flight_control/`
- `定位/mission/outputs/current_safezone/` if using an already-authored route

The target PC must have the Parrot Olympe SDK and the same camera stream size
used by the runtime, currently 1280x720.

## Example Copy Command

From the source machine:

```bash
rsync -aL \
  /path/to/sfm_localization_runtime_tuning_package/ \
  user@TARGET:/path/to/sfm_localization_runtime_tuning_package/
```

After copying, verify that these are real files on the target:

```bash
ls -lh /path/to/sfm_system/定位/bundles/current_reloc_map_updated_v3.pt
ls -lh /path/to/sfm_system/定位/maps/current_realrgb_v3.ply
ls -lh /path/to/package/torch_hub_cache/checkpoints/megaloc/7cb9f7970d366fdf059963d04d372e503e8e9df9/model.safetensors
```

Then verify every packaged file and prove that model loading makes no network
connection:

```bash
cd /path/to/package
python3 tools/package_manifest.py verify
python3 sfm_system/定位/validation/offline_model_smoke.py --model all
# Expected: MegaLoc load OK and XFeat+LighterGlue offline load/match OK
```

Only for bundle rebuild / map diagnostics, also verify:

```bash
ls -lh /path/to/sfm_system/定位/maps/base_glomap_fused_0/images.bin
ls -lh /path/to/sfm_system/定位/maps/base_images_fused
ls -lh /path/to/sfm_system/定位/bundles/base_megaloc_cache_v3.npz
```
