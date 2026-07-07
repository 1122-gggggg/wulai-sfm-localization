#!/usr/bin/env bash
# Fail if the deploy_code <-> flight_control mirror has drifted. Run after editing either.
set -e
D="$(cd "$(dirname "$0")" && pwd)/deploy_code/sfm_glomap_deploy"
F="$(cd "$(dirname "$0")" && pwd)/mission/flight_control"
rc=0
for f in autoflight.py olympe_frame_source.py path_follow_flight.py plan_path.py \
         production_xfeat_tracker.py real_path_follow_controller.py reloc_localizer_xfeat.py pose_types.py; do
  diff -q "$D/$f" "$F/$f" >/dev/null 2>&1 || { echo "DRIFT: $f"; rc=1; }
done
[ $rc -eq 0 ] && echo "mirror OK (deploy_code == flight_control)"
exit $rc
