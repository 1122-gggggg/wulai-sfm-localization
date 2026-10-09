"""Shared fixtures for the direct deploy tests."""

import sys

import pytest


@pytest.fixture(autouse=True)
def _workmap_disk_is_not_host_dependent(monkeypatch):
    """The keyframe free-space floor must not depend on this machine's disk.

    test_workmap_session exercises the floor itself with a controlled usage.
    """
    module = sys.modules.get("online_workmap")
    if module is not None:
        monkeypatch.setattr(module, "_require_free_space", lambda _path: None)
