# Agent Instructions For This Transfer Package

This package is for localization runtime tuning only.

Read `README_FOR_CODEX.md` before making changes.

Do not rebuild the SfM map, do not rebuild the XFeat 2D-3D bundle, and do not
look for the original build/update folders unless the user explicitly asks for
map rebuilding. The expected tuning scope is:

- matcher mode: `nn`, `lighterglue`, `nn_then_lg` (TRACK only; acquisition is
  controlled separately by `acquire_matcher_mode`, default lighterglue)
- NN score thresholds
- local/weak candidate top-K
- adaptive LightGlue fallback settings
- PnP/RANSAC thresholds
- temporal cache settings (active in `nn` and `nn_then_lg` TRACK)
- operator interface look-and-feel and features (layout, colors, displayed
  info, workflow) — confirm with the user before changing safety behavior
  (hover-on-loss, LOST handling, worker-failure handling)

Environment facts:

- Config field names live in `ProductionConfig` inside
  `production_xfeat_tracker.py` — trust the code, not older docs
  (`xfeat_topk_track`, `max_reproj_error_track`, etc.).
- The spawned localizer worker needs torch+torchvision+pycolmap+opencv.
  Interpreter resolution: `$SFM_LOCALIZER_PYTHON` > current UI/pipeline
  interpreter. pycolmap must load the rig-format map
  (original machine: pycolmap 4.0.4; see requirements_runtime.txt).
- Model weights load via torch.hub; offline cache ships in
  package-local `torch_hub_cache/`; keep it beside `sfm_system/` when moving
  the package. Runtime does not require copying it into `~/.cache/torch/hub`.
- `configs/localization_defaults.json` is descriptive only — nothing reads it.
- `localize_pipeline.py`: only `--mode production-stream` works in this
  package. `mission/authoring/` tools are not usable here.

The fixed relocation bundle is:

`sfm_system/定位/bundles/current_reloc_map_updated_v3.pt`

The fixed visual map for the operator UI is:

`sfm_system/定位/maps/current_realrgb_v3.ply`

Do not assume validation videos are present. They were intentionally excluded.
