"""Compatibility facade over the split-out runtime safety modules.

This file used to hold session logs, disk-pressure/retention policy, the
offline network guard, and the autonomy arming gate all mixed together (four
unrelated responsibilities in one file). D2: split into session_logs.py,
disk_policy.py, network_policy.py, and arming_gate.py, each independently
readable and independently tested. This module re-exports their public API
so existing `from runtime_safety import ...` call sites keep working
unchanged. New code should import the specific module it needs directly.
"""
from __future__ import annotations

from arming_gate import (
    AUTONOMY_MIN_CONSECUTIVE_FIXES,
    AUTONOMY_OK_LOC_STATES,
    autonomous_approval_blockers,
    autonomous_arming_blockers,
    manual_flight_blockers,
)
from disk_policy import (
    CRITICAL_FREE_BYTES,
    CRITICAL_FREE_PERCENT,
    GIB,
    RETENTION_MAX_AGE_DAYS,
    RETENTION_MAX_BYTES,
    WARNING_FREE_BYTES,
    WARNING_FREE_PERCENT,
    DiskStatus,
    RetentionResult,
    assess_disk_space,
    enforce_retention,
)
from network_policy import (
    configure_offline_environment,
    install_network_guard,
    network_destination_allowed,
)
from session_logs import (
    SessionCommandLog,
    SessionLogs,
    collect_runtime_identity,
)

__all__ = [
    "AUTONOMY_MIN_CONSECUTIVE_FIXES",
    "AUTONOMY_OK_LOC_STATES",
    "CRITICAL_FREE_BYTES",
    "CRITICAL_FREE_PERCENT",
    "DiskStatus",
    "GIB",
    "RETENTION_MAX_AGE_DAYS",
    "RETENTION_MAX_BYTES",
    "RetentionResult",
    "SessionCommandLog",
    "SessionLogs",
    "WARNING_FREE_BYTES",
    "WARNING_FREE_PERCENT",
    "assess_disk_space",
    "autonomous_approval_blockers",
    "autonomous_arming_blockers",
    "collect_runtime_identity",
    "configure_offline_environment",
    "enforce_retention",
    "install_network_guard",
    "manual_flight_blockers",
    "network_destination_allowed",
]
