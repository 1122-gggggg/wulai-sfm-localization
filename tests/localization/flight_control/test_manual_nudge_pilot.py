from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import manual_nudge_pilot as mnp


class _Value:
    def __init__(self, value=""):
        self.value = value

    def set(self, value):
        self.value = value


class _ImmediateThread:
    def __init__(self, *, target, args, name, daemon):
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon

    def start(self):
        self.target(*self.args)


def _pilot(*, pulse_s=0.0):
    pilot = mnp.NudgePilot(
        dry_run=True,
        ip="0",
        controller="dry",
        pct=mnp.NUDGE_PCT,
        pulse_s=pulse_s,
        cmd_log=mnp.CommandLog(None),
    )
    pilot.log = SimpleNamespace(event=lambda *args, **kwargs: None)
    return pilot


def test_nudge_dispatches_special_commands_and_unknown_without_pulse():
    pilot = _pilot()
    calls = []
    pilot.hover = lambda reason: calls.append(("hover", reason))
    pilot.give_to_pilot = lambda: calls.append(("manual",))
    pilot.land = lambda reason: calls.append(("land", reason))
    pilot.approve_operator_takeoff = lambda: calls.append(("approve",))
    pilot.takeoff = lambda: calls.append(("takeoff",))
    pilot.log = SimpleNamespace(
        event=lambda event, **fields: calls.append((event, fields)),
    )

    pilot.nudge("懸停")
    pilot.nudge("manual")
    pilot.nudge("原地降落")
    pilot.nudge("takeoff", operator_approved=True)
    pilot.nudge("not-a-command")

    assert calls == [
        ("hover", "key_hover"),
        ("manual",),
        ("land", "key_land"),
        ("approve",),
        ("takeoff",),
        ("unknown_nudge", {"name": "not-a-command"}),
    ]


def test_nudge_pulse_emits_direction_then_hover_and_manual_blocks_pcmd(monkeypatch):
    monkeypatch.setattr(mnp.threading, "Thread", _ImmediateThread)
    pilot = _pilot()

    pilot.nudge("前")

    assert pilot._sent == [(0, mnp.NUDGE_PCT, 0, 0), (0, 0, 0, 0)]

    pilot.safety.pilot_sticks = True
    pilot._sent.clear()
    pilot.nudge("前")
    assert pilot._sent == []


@pytest.mark.parametrize(
    ("keysym", "char", "expected", "operator_approved"),
    [
        ("space", " ", "懸停", False),
        ("t", "t", "起飛", True),
        ("w", "w", "前", False),
    ],
)
def test_ui_key_handler_uses_existing_aliases(
    keysym, char, expected, operator_approved
):
    pilot = _pilot()
    pilot.beat = Mock()
    pilot.nudge = Mock()
    status = _Value()
    last = _Value()

    mnp._handle_ui_key(
        pilot,
        status,
        last,
        SimpleNamespace(keysym=keysym, char=char),
    )

    assert pilot.beat.call_count == 2
    pilot.nudge.assert_called_once_with(
        expected,
        operator_approved=operator_approved,
    )
    assert status.value == f"cmd: {expected}  sticks=PC"
    assert last.value == status.value


def test_ui_escape_handler_gives_control_to_pilot_without_nudge():
    pilot = _pilot()
    pilot.beat = Mock()
    pilot.nudge = Mock()
    pilot.give_to_pilot = Mock()
    status = _Value()
    last = _Value()

    mnp._handle_ui_key(
        pilot,
        status,
        last,
        SimpleNamespace(keysym="Escape", char=""),
    )

    pilot.beat.assert_called_once_with()
    pilot.give_to_pilot.assert_called_once_with()
    pilot.nudge.assert_not_called()
    assert status.value == "MANUAL: 搖桿接管"
    assert last.value == status.value


def test_ui_close_sets_stop_cleans_up_and_destroys_root():
    pilot = _pilot()
    pilot.cleanup = Mock()
    status = _Value()
    last = _Value()
    root = Mock()

    mnp._handle_ui_close(pilot, status, last, root)

    assert pilot.safety.stop is True
    pilot.cleanup.assert_called_once_with()
    root.destroy.assert_called_once_with()
    assert status.value == "closing -> land"
    assert last.value == status.value
