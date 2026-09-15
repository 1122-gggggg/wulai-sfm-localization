"""Desktop live poses no longer pause AUTO for source confirmation."""

from types import SimpleNamespace

import numpy as np

import operator_tick


def test_ui_does_not_pause_auto_for_source_confirmation():
    pauses = []
    app = SimpleNamespace(
        _live_pending=object(),
        _pause_integrated_auto=pauses.append,
    )
    xyz = np.array([0.113277, 0.0, 0.0])
    assert operator_tick._live_pose_is_continuous(app, xyz) is True
    assert app._live_pending is None
    assert pauses == []
