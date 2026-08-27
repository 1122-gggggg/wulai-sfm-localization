# Sphinx / ANAFI Smoke Test - 2026-07-02

## Installed Components

- `parrot-sphinx`: 2.25.2-1+ubuntu+noble
- `parrot-ue4-empty`: 2.25.2-20260507.202611
- `sphinx`: `/usr/bin/sphinx`
- `sphinx-cli`: `/usr/bin/sphinx-cli`
- `parrot-ue4-empty`: `/usr/bin/parrot-ue4-empty`
- drone files present:
  - `/opt/parrot-sphinx/usr/share/sphinx/drones/anafi.drone`
  - `/opt/parrot-sphinx/usr/share/sphinx/drones/anafi4k.drone`
  - `/opt/parrot-sphinx/usr/share/sphinx/drones/anafi_ai.drone`

## Olympe SDK Smoke

Temporary venv:

```text
/tmp/sfm_system_olympe_venv
```

Results:

- Olympe import: PASS, version 8.4.0
- ARSDK command construction: PASS
  - `TakeOff`
  - `FlyingStateChanged(state="hovering")`
  - `moveBy`
  - `Landing`
  - `PCMD`

The package keeps reusable wheels in:

```text
/media/cihcilab/新增磁碟區/sfm_system/tools/wheelhouse/olympe
```

## Sphinx Runtime Smoke

Historical command intent (do not reuse the old `#latest` selector):

```bash
export FIRMWARE_URL="https://firmware.parrot.com/Versions/anafi/pc/<reviewed-version>/images/anafi-pc.ext2.zip"
./模擬器/launch_sphinx_anafi_empty.sh
```

The maintained launcher now rejects a missing selector and `#latest`; use one
reviewed, explicit Parrot firmware URL for reproducible runs. It starts both
Sphinx and the offscreen UE4 renderer.

Observed progress:

- Sphinx core started.
- Gazebo master connected.
- Sphinx parameter server opened on port 8383.
- UE4 application started and connected.
- Sphinx connected to `firmwared`.

Blocking result:

```text
Parrot Sphinx requires UEFI Secure Boot to be disabled.
```

`mokutil --sb-state` reported:

```text
SecureBoot enabled
```

Therefore the simulator cannot launch the ANAFI firmware on this workstation
until Secure Boot is disabled in BIOS/UEFI. After disabling Secure Boot, log out
and log back in so the `firmwared` group membership is active, then rerun:

```bash
/media/cihcilab/新增磁碟區/sfm_system/tools/start_sphinx_anafi.sh --model anafi --with-ue
source /tmp/sfm_system_olympe_venv/bin/activate
python /media/cihcilab/新增磁碟區/sfm_system/tools/olympe_sphinx_smoke.py --mode sim --ip 10.202.0.1
```
