import pytest

from anafi_pcmd_sim.models import PilotingCommand, Scenario, all_direction_scenarios


def test_forward_up_scenario_combines_pitch_and_gaz() -> None:
    scenario = Scenario.forward_up(duration_s=1.5, pitch=20, gaz=20)

    assert scenario.command.roll == 0
    assert scenario.command.pitch == 20
    assert scenario.command.gaz == 20
    assert scenario.command.yaw == 0


@pytest.mark.parametrize("value", [-101, 101])
def test_pcmd_rejects_values_outside_firmware_range(value: int) -> None:
    with pytest.raises(ValueError, match="-100.*100"):
        PilotingCommand(roll=value, pitch=0, yaw=0, gaz=0)


def test_zero_pcmd_has_no_horizontal_movement_flag() -> None:
    assert PilotingCommand.zero().horizontal_enabled is False


def test_all_direction_scenarios_cover_each_nonzero_body_direction_once() -> None:
    scenarios = all_direction_scenarios(duration_s=1.5, magnitude=20)

    assert len(scenarios) == 26
    assert len({scenario.command for scenario in scenarios}) == 26
    by_name = {scenario.name: scenario for scenario in scenarios}
    assert by_name["right-forward-up"].command == PilotingCommand(
        roll=20,
        pitch=20,
        yaw=0,
        gaz=20,
    )
    assert by_name["left-backward-down"].command == PilotingCommand(
        roll=-20,
        pitch=-20,
        yaw=0,
        gaz=-20,
    )
    assert by_name["right"].command == PilotingCommand(
        roll=20,
        pitch=0,
        yaw=0,
        gaz=0,
    )
