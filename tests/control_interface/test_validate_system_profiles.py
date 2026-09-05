from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "控制介面程式"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))

from site_profile import ROUTE_EDITOR_AUTO_APPROVAL_NOTE  # noqa: E402
from validate_system_profiles import _expects_auto_approval  # noqa: E402


class _Flight:
    def __init__(self, approval_note: str | None) -> None:
        self.approval_note = approval_note


class _Profile:
    def __init__(self, approval_note: str | None) -> None:
        self.flight = _Flight(approval_note)


def test_route_editor_approval_is_expected_to_be_approved() -> None:
    assert _expects_auto_approval(_Profile(ROUTE_EDITOR_AUTO_APPROVAL_NOTE))


@pytest.mark.parametrize(
    "note",
    [
        None,
        "",
        "   ",
        "Imported asset changed; flight re-approval required.",
        "Imported for ground localization; flight approval required.",
        # A hand-written note that merely mentions AUTO must not pass: the
        # marker is the editor's exact note, not any approving-sounding text.
        "approved for AUTO by me",
        ROUTE_EDITOR_AUTO_APPROVAL_NOTE + " (edited)",
    ],
)
def test_everything_other_than_the_editor_note_expects_no_approval(note: str | None) -> None:
    assert not _expects_auto_approval(_Profile(note))


def test_profile_without_a_flight_block_expects_no_approval() -> None:
    class _Bare:
        flight = None

    assert not _expects_auto_approval(_Bare())


def test_shipped_profiles_agree_with_their_recorded_approval() -> None:
    """The real profiles must satisfy the rule the validator enforces."""
    import validate_system_profiles as vsp

    _profiles, failures = vsp.validate()
    assert failures == []
