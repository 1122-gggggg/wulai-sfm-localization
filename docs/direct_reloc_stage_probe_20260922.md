# Direct reloc stage probe (2026-09-22)

這是一個 bounded、單幀的 direct backend stage probe。它不啟動 operator UI、
不啟動真機、不跑 flight control，也不改 production profile。輸入固定為
P172 的 decoded ordinal 238，provider 只做一次 relocalization；模型載入與
warmup 不算入該次 reloc runtime。

## 可重跑命令

工作目錄是 `/home/allen/localization`，使用 repo venv：

```bash
/home/allen/localization/.venv/bin/python - <<'PY'
from pathlib import Path
import json, sys, time
import cv2
import numpy as np
import torch

root = Path('/home/allen/localization')
for relative in ('定位演算法/flight_control',
                 '定位演算法/deploy_code/sfm_glomap_deploy',
                 '定位演算法/deploy_code/sfm_direct_deploy',
                 '控制介面程式'):
    sys.path.insert(0, str(root / relative))
from production_localizer_factory import build_production_localizer
from real_path_follow_controller import load_map_frame

site_path = root / '地圖檔/場域/river_site/site_profile.json'
site = json.loads(site_path.read_text())
asset = lambda value: (site_path.parent / value).resolve()
camera = site['query_camera']
built = build_production_localizer(
    backend='direct',
    bundle=asset(site['assets']['localization_bundle']),
    bundle_sha256=site['asset_sha256']['localization_bundle'],
    production_profile=asset(site['localizer_profile']),
    production_profile_sha256=site['asset_sha256']['localizer_profile'],
    camera_tuple=(camera['model'], camera['width'], camera['height'], camera['params']),
    frame_source=lambda: None,
    map_frame=load_map_frame(asset(site['map_align'])),
)
tracker = built.tracker.trk
started = time.perf_counter()
tracker.provider.ensure_models()
torch.cuda.synchronize()
warmup_s = time.perf_counter() - started

cap = cv2.VideoCapture('/home/allen/下載/P1720172.MP4')
if not cap.isOpened():
    raise RuntimeError('cannot open P172')
frame = None
for ordinal in range(239):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f'decode failed at {ordinal}')
cap.release()
if frame is None:
    raise RuntimeError('no frame')

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (tracker._w, tracker._h), interpolation=cv2.INTER_AREA)
gray = np.ascontiguousarray(gray, dtype=np.uint8)
torch.cuda.synchronize()
started = time.perf_counter()
fix = tracker.provider.localize_array(gray)
torch.cuda.synchronize()
wall_ms = (time.perf_counter() - started) * 1000.0
print(json.dumps({
    'ordinal': 238,
    'warmup_s': warmup_s,
    'wall_ms_sync': wall_ms,
    'runtime_ms': fix.runtime_ms,
    'status': fix.status,
    'inliers': fix.inliers,
    'stage_ms': fix.stage_ms,
    'references': fix.reference_names,
}, ensure_ascii=False, sort_keys=True))
tracker.provider.close()
PY
```

The two `torch.cuda.synchronize()` calls around the reloc job make the outer wall
timer include asynchronous CUDA work. `ensure_models()` performs the provider's
normal model load and warmup first; `warmup_s` is reported separately. No CUDA
graph, profile override, batch-ref experiment, or cache setting is changed.

## Recorded result

The recorded result is [p172_ordinal238_stage_probe.json](../outputs/analysis/runtime_fixes_20260922/p172_ordinal238_stage_probe.json).
It was collected on an NVIDIA GeForce RTX 5060 Laptop GPU with torch
`2.11.0+cu128` and OpenCV `4.13.0`:

```json
{
  "ordinal": 238,
  "warmup_s": 1.5075224509998861,
  "wall_ms_sync": 254.85397600004944,
  "runtime_ms": 254.76893299992298,
  "status": "POSE_ESTIMATED_WEAK",
  "inliers": 463,
  "stage_ms": {
    "match_lift_ms": 212.9455459999008,
    "matched_references": 2.0,
    "reloc_pnp_ms": 28.106965000006312,
    "retrieval_ms": 13.104293999958827
  },
  "references": [
    "vid_540e7d389b1d93f4/frame_00002820.jpg",
    "vid_540e7d389b1d93f4/frame_00002742.jpg"
  ]
}
```

`match_lift_ms` is the dominant measured stage in this probe. The existing P172
artifact's 1613 ms job at ordinal 238 was not reproduced by this single-frame
probe, so its cause remains unresolved; this result does not attribute that tail
to external GPU load or claim that it is algorithmically explained.

The offline capture harness now copies a provider's finite `RelocFix.stage_ms`
mapping into each `reloc_jobs[]` row. A future bounded or full capture therefore
retains the per-job stage breakdown instead of only `runtime_ms`.
