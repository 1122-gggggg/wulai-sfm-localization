from __future__ import annotations

from types import SimpleNamespace

import pytest

from operator_site_runtime import (
    ActiveSiteRuntime,
    PreparedSiteRuntime,
    close_active_site_runtime,
    replace_active_site_runtime,
)


class _Closer:
    def __init__(self, name: str = "resource", *, fail: bool = False) -> None:
        self.closed = 0
        self.name = name
        self.fail = fail

    def close(self, *args, **kwargs) -> None:
        self.closed += 1
        if self.fail:
            raise RuntimeError(f"{self.name} close failed")


class _Logs(_Closer):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__("session_logs", fail=fail)
        self.reason = None

    def close(self, *, reason: str) -> None:
        self.reason = reason
        super().close()


class _LiveBackend:
    is_live = True

    def __init__(self, cleanup_result=True) -> None:
        self.cleanup_result = cleanup_result
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleanup_calls += 1
        return self.cleanup_result


def _prepared(name: str) -> PreparedSiteRuntime:
    return PreparedSiteRuntime(
        args=SimpleNamespace(site_profile=f"/{name}/site_profile.json"),
        interface_mode="real-flight",
        profile=SimpleNamespace(site_id=name),
        hardware_approval=None,
        map_points=(),
        route_points=(),
        mission_route_snapshot=None,
        replay_rows=(),
    )


def _active(
    name: str,
    *,
    cleanup_result=True,
    failures: tuple[str, ...] = (),
) -> ActiveSiteRuntime:
    return ActiveSiteRuntime(
        prepared=_prepared(name),
        session_logs=_Logs(fail="session_logs" in failures),
        backend=_LiveBackend(cleanup_result),
        video_stream=object(),
        live_backend=object(),
        localizer=_Closer("localizer", fail="localizer" in failures),
        detector=_Closer("detector", fail="detector" in failures),
        lost_hold=None,
    )


def test_close_runtime_requires_verified_live_cleanup_before_workers_close() -> None:
    runtime = _active("old", cleanup_result=False)

    with pytest.raises(RuntimeError, match="cleanup"):
        close_active_site_runtime(runtime, reason="site_switch")

    assert runtime.backend.cleanup_calls == 1
    assert runtime.localizer.closed == 0
    assert runtime.detector.closed == 0
    assert runtime.session_logs.closed == 0


@pytest.mark.parametrize(
    "failures",
    [
        ("localizer",),
        ("detector",),
        ("session_logs",),
        ("localizer", "detector"),
        ("localizer", "session_logs"),
        ("detector", "session_logs"),
        ("localizer", "detector", "session_logs"),
    ],
)
def test_close_runtime_attempts_every_dependent_once_and_reports_failures(
    failures: tuple[str, ...],
) -> None:
    runtime = _active("old", failures=failures)

    with pytest.raises(RuntimeError) as raised:
        close_active_site_runtime(runtime, reason="site_switch")

    assert runtime.backend.cleanup_calls == 1
    assert runtime.localizer.closed == 1
    assert runtime.detector.closed == 1
    assert runtime.session_logs.closed == 1
    message = str(raised.value)
    assert all(failure in message for failure in failures)


@pytest.mark.parametrize("cleanup_result", [None, 1, "ok"])
def test_live_cleanup_accepts_only_literal_true(cleanup_result) -> None:
    runtime = _active("old", cleanup_result=cleanup_result)

    with pytest.raises(RuntimeError, match="not confirmed"):
        close_active_site_runtime(runtime, reason="site_switch")

    assert runtime.localizer.closed == 0
    assert runtime.detector.closed == 0
    assert runtime.session_logs.closed == 0


def test_replace_runtime_closes_old_then_activates_new() -> None:
    old = _active("old")
    new_plan = _prepared("new")
    started = []

    def start(plan):
        started.append(plan.profile.site_id)
        return _active(plan.profile.site_id)

    result = replace_active_site_runtime(old, new_plan, start)

    assert result.applied
    assert result.runtime is not None
    assert result.runtime.prepared.profile.site_id == "new"
    assert started == ["new"]
    assert old.backend.cleanup_calls == 1
    assert old.localizer.closed == 1
    assert old.detector.closed == 1
    assert old.session_logs.reason == "site_switch"


def test_replace_runtime_rebuilds_old_site_when_new_start_fails() -> None:
    old = _active("old")
    new_plan = _prepared("new")
    started = []

    def start(plan):
        started.append(plan.profile.site_id)
        if plan.profile.site_id == "new":
            raise RuntimeError("new connection failed")
        return _active(plan.profile.site_id)

    result = replace_active_site_runtime(old, new_plan, start)

    assert not result.applied
    assert result.rolled_back
    assert result.runtime is not None
    assert result.runtime.prepared.profile.site_id == "old"
    assert "new connection failed" in (result.error or "")
    assert started == ["new", "old"]


def test_replace_runtime_keeps_old_site_when_cleanup_is_not_confirmed() -> None:
    old = _active("old", cleanup_result=False)
    started = []

    result = replace_active_site_runtime(
        old,
        _prepared("new"),
        lambda plan: started.append(plan.profile.site_id),
    )

    assert not result.applied
    assert not result.old_closed
    assert not result.rolled_back
    assert result.runtime is old
    assert "not confirmed" in (result.error or "")
    assert started == []


@pytest.mark.parametrize("failed_resource", ["localizer", "detector", "session_logs"])
def test_replace_runtime_does_not_publish_old_after_dependent_cleanup_failure(
    failed_resource: str,
) -> None:
    old = _active("old", failures=(failed_resource,))
    started = []

    result = replace_active_site_runtime(
        old,
        _prepared("new"),
        lambda plan: started.append(plan.profile.site_id),
    )

    assert not result.applied
    assert result.runtime is None
    assert result.old_closed
    assert not result.rolled_back
    assert failed_resource in (result.error or "")
    assert started == []


def test_replace_runtime_reports_disconnected_state_when_rollback_fails() -> None:
    old = _active("old")
    started = []

    def fail_start(plan):
        started.append(plan.profile.site_id)
        raise RuntimeError(f"{plan.profile.site_id} unavailable")

    result = replace_active_site_runtime(old, _prepared("new"), fail_start)

    assert not result.applied
    assert result.old_closed
    assert not result.rolled_back
    assert result.runtime is None
    assert result.error == "new unavailable"
    assert result.rollback_error == "old unavailable"
    assert started == ["new", "old"]
