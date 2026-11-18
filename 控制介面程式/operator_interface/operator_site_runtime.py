"""Transactional resource replacement for an in-process operator site switch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PreparedSiteRuntime:
    """Validated, inactive inputs for one site runtime."""

    args: Any
    interface_mode: Any
    profile: Any
    hardware_approval: Any
    map_points: Any
    route_points: Any
    mission_route_snapshot: Any
    replay_rows: Any


@dataclass
class ActiveSiteRuntime:
    """Resources owned by one active operator session."""

    prepared: PreparedSiteRuntime
    session_logs: Any
    backend: Any
    video_stream: Any
    live_backend: Any
    localizer: Any
    detector: Any
    lost_hold: Any


@dataclass(frozen=True)
class SiteRuntimeSwitchResult:
    runtime: ActiveSiteRuntime | None
    applied: bool
    old_closed: bool
    rolled_back: bool = False
    error: str | None = None
    rollback_error: str | None = None


class _SiteRuntimeCleanupError(RuntimeError):
    """Cleanup failed after recording which part of the runtime is reusable."""

    def __init__(self, errors: list[str], *, runtime_usable: bool):
        self.errors = tuple(errors)
        self.runtime_usable = bool(runtime_usable)
        super().__init__("site runtime cleanup failed: " + "; ".join(errors))


def _close_resource_once(
    resource: Any,
    *,
    label: str,
    reason: str,
    seen: set[int],
    errors: list[str],
) -> None:
    if resource is None or id(resource) in seen:
        return
    seen.add(id(resource))
    try:
        close = getattr(resource, "close", None)
    except Exception as exc:
        detail = str(exc) or repr(exc)
        errors.append(f"{label} lookup failed: {detail}")
        return
    if not callable(close):
        return
    try:
        if label == "session_logs":
            close(reason=reason)
        else:
            close()
    except Exception as exc:
        detail = str(exc) or repr(exc)
        errors.append(f"{label}: {detail}")


def _cleanup_backend(
    backend: Any,
    *,
    is_live: bool,
    reason: str,
) -> str | None:
    cleanup = getattr(backend, "cleanup", None)
    if callable(cleanup):
        try:
            cleanup_result = cleanup()
        except Exception as exc:
            detail = str(exc) or repr(exc)
            return f"backend cleanup failed: {detail}"
        accepted = (
            cleanup_result is True
            if is_live
            else cleanup_result in (None, True)
        )
        if not accepted:
            return (
                "backend cleanup was not confirmed "
                f"(result={cleanup_result!r})"
            )
        return None
    if is_live:
        return "live backend has no verifiable cleanup method"

    close_backend = getattr(backend, "close", None)
    if not callable(close_backend):
        return None
    try:
        result = close_backend(reason)
    except Exception as exc:
        detail = str(exc) or repr(exc)
        return f"backend close failed: {detail}"
    if hasattr(result, "closed") and not bool(result.closed):
        return f"backend close was not confirmed (result={result!r})"
    return None


def _close_dependents(
    runtime: ActiveSiteRuntime,
    *,
    reason: str,
    seen: set[int],
) -> list[str]:
    errors: list[str] = []
    _close_resource_once(
        runtime.localizer,
        label="localizer",
        reason=reason,
        seen=seen,
        errors=errors,
    )
    _close_resource_once(
        runtime.detector,
        label="detector",
        reason=reason,
        seen=seen,
        errors=errors,
    )
    # Live video is owned by the backend grabber and was stopped by cleanup.
    if runtime.live_backend is None:
        _close_resource_once(
            runtime.video_stream,
            label="video_stream",
            reason=reason,
            seen=seen,
            errors=errors,
        )
    _close_resource_once(
        runtime.session_logs,
        label="session_logs",
        reason=reason,
        seen=seen,
        errors=errors,
    )
    return errors


def close_active_site_runtime(
    runtime: ActiveSiteRuntime,
    *,
    reason: str,
) -> None:
    """Close a runtime without tearing down dependent resources prematurely.

    A live backend must literally confirm cleanup. If it cannot confirm the
    aircraft is landed, its connection, workers and session log stay intact so
    the caller can retry instead of presenting a false-success site switch.
    """

    backend = runtime.backend
    is_live = bool(getattr(backend, "is_live", False))
    seen: set[int] = {id(backend)}
    backend_failure = _cleanup_backend(
        backend,
        is_live=is_live,
        reason=reason,
    )

    # A live backend that cannot confirm touchdown must remain intact so the
    # caller can retry. Do not close workers/logs in that case.
    if is_live and backend_failure is not None:
        raise _SiteRuntimeCleanupError([backend_failure], runtime_usable=True)
    errors = _close_dependents(runtime, reason=reason, seen=seen)
    if backend_failure is not None:
        errors.insert(0, backend_failure)
    if errors:
        raise _SiteRuntimeCleanupError(errors, runtime_usable=False)


def _discard_candidate(candidate: ActiveSiteRuntime | None) -> str:
    if candidate is None:
        return ""
    try:
        close_active_site_runtime(candidate, reason="site_switch_cancelled")
    except BaseException as exc:
        return "; candidate cleanup: " + (str(exc) or repr(exc))
    return ""


def _exception_text(exc: BaseException) -> str:
    return str(exc) or repr(exc)


def _kept_old_result(old: ActiveSiteRuntime, error: str) -> SiteRuntimeSwitchResult:
    return SiteRuntimeSwitchResult(
        runtime=old, applied=False, old_closed=False, error=error,
    )


def _cutover_refusal(
    old: ActiveSiteRuntime,
    candidate: ActiveSiteRuntime | None,
    cancel_requested: Callable[[], bool] | None,
    cutover_allowed: Callable[[], bool] | None,
) -> SiteRuntimeSwitchResult | None:
    try:
        cancelled = cancel_requested is not None and cancel_requested()
        allowed = cutover_allowed is None or cutover_allowed()
    except BaseException as exc:
        return _kept_old_result(old, _exception_text(exc) + _discard_candidate(candidate))
    if cancelled or not allowed:
        return _kept_old_result(
            old, "runtime switch cancelled by safety state" + _discard_candidate(candidate)
        )
    return None


def _close_old_runtime(
    old: ActiveSiteRuntime, candidate: ActiveSiteRuntime | None
) -> SiteRuntimeSwitchResult | None:
    try:
        close_active_site_runtime(old, reason="site_switch")
    except _SiteRuntimeCleanupError as exc:
        error = str(exc) + _discard_candidate(candidate)
        if exc.runtime_usable:
            return _kept_old_result(old, error)
        return SiteRuntimeSwitchResult(
            runtime=None,
            applied=False,
            old_closed=True,
            error=error,
        )
    except Exception as exc:
        return _kept_old_result(old, _exception_text(exc) + _discard_candidate(candidate))
    return None


def _rollback_to_old_runtime(
    old: ActiveSiteRuntime,
    start_runtime: Callable[[PreparedSiteRuntime], ActiveSiteRuntime],
    error: str,
) -> SiteRuntimeSwitchResult:
    try:
        rollback = start_runtime(old.prepared)
    except BaseException as rollback_exc:
        return SiteRuntimeSwitchResult(
            runtime=None,
            applied=False,
            old_closed=True,
            error=error,
            rollback_error=_exception_text(rollback_exc),
        )
    return SiteRuntimeSwitchResult(
        runtime=rollback,
        applied=False,
        old_closed=True,
        rolled_back=True,
        error=error,
    )


def _start_replacement(
    old: ActiveSiteRuntime,
    new_plan: PreparedSiteRuntime,
    start_runtime: Callable[[PreparedSiteRuntime], ActiveSiteRuntime],
    candidate: ActiveSiteRuntime | None,
    cancelled: bool,
) -> SiteRuntimeSwitchResult:
    if cancelled:
        candidate_error = _discard_candidate(candidate)
        if candidate_error:
            return SiteRuntimeSwitchResult(
                runtime=None, applied=False, old_closed=True,
                error="runtime switch cancelled by safety command" + candidate_error,
            )
    try:
        if cancelled:
            raise RuntimeError("runtime switch cancelled by safety command")
        runtime = candidate if candidate is not None else start_runtime(new_plan)
    except BaseException as exc:
        return _rollback_to_old_runtime(old, start_runtime, _exception_text(exc))
    return SiteRuntimeSwitchResult(
        runtime=runtime,
        applied=True,
        old_closed=True,
    )


def replace_active_site_runtime(
    old: ActiveSiteRuntime,
    new_plan: PreparedSiteRuntime,
    start_runtime: Callable[[PreparedSiteRuntime], ActiveSiteRuntime],
    *,
    start_before_close: bool = False,
    cancel_requested: Callable[[], bool] | None = None,
    cutover_allowed: Callable[[], bool] | None = None,
) -> SiteRuntimeSwitchResult:
    """Replace resources transactionally, optionally warming a non-live source first."""

    candidate = None
    if start_before_close:
        try:
            candidate = start_runtime(new_plan)
        except BaseException as exc:
            return _kept_old_result(old, _exception_text(exc))

    refusal = _cutover_refusal(old, candidate, cancel_requested, cutover_allowed)
    if refusal is not None:
        return refusal
    close_failure = _close_old_runtime(old, candidate)
    if close_failure is not None:
        return close_failure
    cancelled = cancel_requested is not None and cancel_requested()
    return _start_replacement(old, new_plan, start_runtime, candidate, cancelled)


__all__ = [
    "ActiveSiteRuntime",
    "PreparedSiteRuntime",
    "SiteRuntimeSwitchResult",
    "close_active_site_runtime",
    "replace_active_site_runtime",
]
