#!/usr/bin/env bash
# Fail if shared production artifacts in deploy_code <-> flight_control drift.
#
# Intentionally not mirrored byte-for-byte:
#   - olympe_frame_source.py: the mission copy carries operator/SkyController
#     stream integration; review changes to each copy explicitly.
#   - README.md: the mission copy also documents its mission-only nudge UI.
set -e
D="$(cd "$(dirname "$0")" && pwd)/deploy_code/sfm_glomap_deploy"
F="$(cd "$(dirname "$0")" && pwd)/mission/flight_control"
rc=0
for f in autoflight.py path_follow_flight.py plan_path.py \
         production_xfeat_tracker.py real_path_follow_controller.py reloc_localizer_xfeat.py pose_types.py \
         artifact_integrity.py megaloc_cache.py; do
  diff -q "$D/$f" "$F/$f" >/dev/null 2>&1 || { echo "DRIFT: $f"; rc=1; }
done
[ $rc -eq 0 ] && echo "mirror OK (shared production artifacts match)"
exit $rc
