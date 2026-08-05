"""Contract checks for the opt-in, one-shot boot runner."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_user_service_is_opt_in_and_runs_only_the_project_boot_runner() -> None:
    unit = (PROJECT_ROOT / "systemd" / "user" / "anafi-pcmd-sim-next-boot.service").read_text(
        encoding="utf-8"
    )

    assert "ConditionPathExists=@PROJECT_DIR@/.boot-probe.armed" in unit
    assert "Type=oneshot" in unit
    assert "Environment=ANAFI_PCMD_SIM_PROJECT_DIR=@PROJECT_DIR@" in unit
    assert "ExecStart=@PROJECT_DIR@/scripts/boot_probe_once.sh" in unit
    assert "WantedBy=default.target" in unit


def test_boot_runner_is_fixed_to_a_safe_offline_sphinx_probe() -> None:
    runner = (PROJECT_ROOT / "scripts" / "boot_probe_once.sh").read_text(encoding="utf-8")

    assert "anafi-pcmd-sim preflight" in runner
    assert "--launch-sphinx" in runner
    assert "--duration 1.5 --pitch 20 --gaz 20" in runner
    assert "--locked --offline --no-sync" in runner
    assert 'mv -- "$ARM_FILE"' in runner
    assert '"$@"' not in runner


def test_arming_script_creates_only_the_expected_private_marker() -> None:
    armer = (PROJECT_ROOT / "scripts" / "arm_next_boot_probe.sh").read_text(encoding="utf-8")

    assert "umask 077" in armer
    assert 'ARM_FILE="$PROJECT_DIR/.boot-probe.armed"' in armer
    assert "boot probe armed" in armer


def test_boot_scripts_resolve_the_clone_root_and_installer_renders_the_unit_template() -> None:
    runner = (PROJECT_ROOT / "scripts" / "boot_probe_once.sh").read_text(encoding="utf-8")
    installer = (PROJECT_ROOT / "scripts" / "install_user_boot_service.sh").read_text(
        encoding="utf-8"
    )
    disarmer = (PROJECT_ROOT / "scripts" / "disarm_next_boot_probe.sh").read_text(encoding="utf-8")

    assert "ANAFI_PCMD_SIM_PROJECT_DIR" in runner
    assert "BASH_SOURCE[0]" in runner
    assert 'sed "s|@PROJECT_DIR@|' in installer
    assert "unresolved project path template" in installer
    assert "BASH_SOURCE[0]" in disarmer
