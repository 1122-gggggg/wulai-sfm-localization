# One-shot boot autorun

This project can run one **Sphinx-only** PCMD probe after the next machine boot without an SSH
session. It is a per-user systemd service, so it does not need root access. Install it from the
clone on each host; the installer substitutes that clone's absolute path into the user-unit
template. User-service lingering must be enabled on the host to start it before a desktop login.

## Safety properties

- The unit has no target-IP argument and invokes the project's hard-locked Sphinx-only command.
- It runs only when `.boot-probe.armed` exists.
- It performs `preflight` before consuming that marker. A preflight failure leaves the marker in
  place for the next boot.
- Once preflight passes, it consumes the marker before launching simulator processes. An
  unexpected runtime failure will not cause an infinite boot-time retry loop.
- Results are written below `artifacts/boot-runs/`; system logs are available through
  `journalctl --user`.

## Prepare the next boot

```bash
cd /home/allen/localization/模擬器/parrot_stimulate
uv sync --locked
./scripts/install_user_boot_service.sh
./scripts/arm_next_boot_probe.sh
systemctl --user status anafi-pcmd-sim-next-boot.service
```

The installed service is enabled permanently but skipped unless the arm marker exists. Cancel a
pending run with:

```bash
./scripts/disarm_next_boot_probe.sh
```

## Verify after reconnecting

```bash
systemctl --user status anafi-pcmd-sim-next-boot.service
journalctl --user -u anafi-pcmd-sim-next-boot.service -b --no-pager
find artifacts/boot-runs -name report.json -print
```

## Required UEFI setting

Parrot Sphinx 2.25.2 does not start while UEFI Secure Boot is enabled. The service will safely
fail its preflight and keep the arm marker in that case. Disable Secure Boot in UEFI, save, and
let the system reboot; then this unit can perform the probe without remote interaction.

The service uses `parrot-ue4-empty -RenderOffScreen`. That is Parrot's headless mode, but its
first successful post-UEFI run is still the confirmation that this host's GPU/driver stack works
without an interactive desktop session.
