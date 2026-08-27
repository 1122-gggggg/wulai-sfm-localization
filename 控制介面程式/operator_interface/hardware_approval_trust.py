"""Offline Ed25519 trust-chain checks for hardware AUTO approval.

The existing ``anafi-hardware-approval/v2`` receipt remains the operator-facing
receipt format.  A detached JSON envelope carries the key identity, purpose and
validity window and signs the canonical receipt together with those fields.  A
profile readiness check can therefore call :func:`verify_hardware_approval_receipt`
without importing a private key or treating a hash-pinned JSON file as proof of
who approved it.

This module deliberately does not create approval evidence.  The request and
unsigned-payload helpers only prepare data for an authorised human process.
Private keys must stay outside the repository and outside the flight runtime.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Cryptodome.PublicKey import ECC
from Cryptodome.Signature import eddsa

__all__ = [
    "AUTO_APPROVAL_PURPOSE",
    "ENVELOPE_SCHEMA",
    "HardwareApprovalError",
    "HardwareApprovalVerdict",
    "REQUEST_SCHEMA",
    "RECEIPT_SCHEMA",
    "TRUST_STORE_SCHEMA",
    "TrustedHardwareApprovalKey",
    "attach_detached_signature",
    "build_pending_request",
    "canonical_json_bytes",
    "create_unsigned_signing_payload",
    "discover_profile_bindings",
    "verify_hardware_approval_receipt",
]


RECEIPT_SCHEMA = "anafi-hardware-approval/v2"
ENVELOPE_SCHEMA = "anafi-hardware-approval-signed-envelope/v2"
REQUEST_SCHEMA = "anafi-hardware-approval-request/v2"
TRUST_STORE_SCHEMA = "anafi-hardware-approval-trust-store/v1"
AUTO_APPROVAL_PURPOSE = "hardware_approval:auto"
_REQUIRED_PERMISSION = "hardware_approval:auto"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ED25519_SPKI_PREFIX = bytes.fromhex("302a300506032b6570032100")
_SIGNATURE_SIZE = 64

_RECEIPT_KEYS = {
    "schema",
    "approved",
    "site_id",
    "coordinate_frame_id",
    "profile_sha256",
    "route_sha256",
    "bundle_sha256",
    "approved_mode",
    "aircraft",
    "controller",
    "olympe_versions",
    "approved_envelope",
    "approval_note",
}
_IDENTITY_KEYS = {"product", "serial", "firmware_versions"}
_ENVELOPE_KEYS = {
    "schema",
    "receipt",
    "purpose",
    "key_id",
    "issued_at",
    "expires_at",
}
_SIGNED_ENVELOPE_KEYS = _ENVELOPE_KEYS | {"signature_b64"}
_TRUST_STORE_KEYS = {"schema", "keys"}
_TRUSTED_KEY_REQUIRED = {
    "key_id",
    "public_key",
    "role",
    "permissions",
    "not_before",
    "not_after",
    "revoked",
}
_TRUSTED_KEY_OPTIONAL = {"algorithm", "revoked_at", "revocation_reason"}
_APPROVED_ENVELOPE_KEYS = {
    "max_altitude_m",
    "max_distance_m",
    "max_tilt_deg",
    "max_vertical_speed_ms",
    "max_rotation_speed_degs",
    "gps_required",
}
_REQUIRED_BINDINGS = {
    "site_id",
    "coordinate_frame_id",
    "profile_sha256",
    "route_sha256",
    "bundle_sha256",
}
_OPTIONAL_BINDING_FIELDS = {"approved_mode"}


class HardwareApprovalError(ValueError):
    """Raised by strict helper functions for malformed trust material."""

    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


@dataclass(frozen=True)
class HardwareApprovalVerdict:
    """Machine-readable readiness result suitable for a fail-closed gate."""

    accepted: bool
    reason_code: str
    detail: str
    key_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "key_id": self.key_id,
        }


@dataclass(frozen=True)
class TrustedHardwareApprovalKey:
    """Validated trust-store entry with the raw Ed25519 public key bytes."""

    key_id: str
    public_key: bytes
    role: str
    permissions: tuple[str, ...]
    not_before: datetime
    not_after: datetime
    revoked: bool
    revoked_at: datetime | None = None


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(source: Mapping[str, Any] | str | Path, *, label: str) -> dict[str, Any]:
    if isinstance(source, Mapping):
        value = dict(source)
    else:
        path = Path(source)
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise HardwareApprovalError(
                "MATERIAL_UNREADABLE", f"cannot read {label}: {exc}"
            ) from exc
        try:
            value = json.loads(
                content.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise HardwareApprovalError("MATERIAL_INVALID_JSON", f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise HardwareApprovalError("MATERIAL_NOT_OBJECT", f"{label} must be a JSON object")
    return value


def _validate_json_mapping(value: Mapping[str, Any], *, path: str, seen: set[int]) -> None:
    marker = id(value)
    if marker in seen:
        raise HardwareApprovalError("CANONICAL_JSON_INVALID", f"cyclic value at {path}")
    seen.add(marker)
    for key, child in value.items():
        if not isinstance(key, str):
            raise HardwareApprovalError(
                "CANONICAL_JSON_INVALID", f"object key at {path} is not a string"
            )
        _validate_json_value(child, path=f"{path}.{key}", seen=seen)
    seen.remove(marker)


def _validate_json_sequence(
    value: list[Any] | tuple[Any, ...], *, path: str, seen: set[int]
) -> None:
    marker = id(value)
    if marker in seen:
        raise HardwareApprovalError("CANONICAL_JSON_INVALID", f"cyclic value at {path}")
    seen.add(marker)
    for index, child in enumerate(value):
        _validate_json_value(child, path=f"{path}[{index}]", seen=seen)
    seen.remove(marker)


def _validate_json_scalar(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, float):
        raise HardwareApprovalError(
            "CANONICAL_JSON_INVALID", f"non-finite number at {path}"
        )
    raise HardwareApprovalError(
        "CANONICAL_JSON_INVALID", f"unsupported value type at {path}: {type(value).__name__}"
    )


def _validate_json_value(value: Any, *, path: str = "$", seen: set[int] | None = None) -> None:
    """Reject values that json.dumps would coerce ambiguously."""
    active_seen = set() if seen is None else seen
    if isinstance(value, Mapping):
        _validate_json_mapping(value, path=path, seen=active_seen)
        return
    if isinstance(value, (list, tuple)):
        _validate_json_sequence(value, path=path, seen=active_seen)
        return
    _validate_json_scalar(value, path=path)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Encode the repository's deterministic UTF-8 JSON signing form."""
    if not isinstance(value, Mapping):
        raise HardwareApprovalError("CANONICAL_JSON_INVALID", "canonical value must be an object")
    _validate_json_value(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise HardwareApprovalError("CANONICAL_JSON_INVALID", str(exc)) from exc


def _text(value: Any, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise HardwareApprovalError("SCHEMA_INVALID", f"{field} must be a non-empty string")
    if value != value.strip():
        raise HardwareApprovalError(
            "SCHEMA_INVALID", f"{field} must not have surrounding whitespace"
        )
    if "\x00" in value or any(ord(char) < 0x20 for char in value):
        raise HardwareApprovalError("SCHEMA_INVALID", f"{field} contains a control character")
    return value


def _sha256(value: Any, *, field: str, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise HardwareApprovalError("SCHEMA_INVALID", f"{field} must be a lowercase SHA-256")
    return value


def _parse_time(value: Any, *, field: str) -> datetime:
    raw = _text(value, field=field)
    if not raw.endswith("Z"):
        raise HardwareApprovalError("SCHEMA_INVALID", f"{field} must use UTC `Z` notation")
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00")
    except ValueError as exc:
        raise HardwareApprovalError("SCHEMA_INVALID", f"{field} is not ISO-8601 UTC") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HardwareApprovalError("SCHEMA_INVALID", f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _now(value: datetime | None) -> datetime:
    parsed = datetime.now(timezone.utc) if value is None else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HardwareApprovalError("TIME_INVALID", "now must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _string_list(value: Any, *, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise HardwareApprovalError("SCHEMA_INVALID", f"{field} must be a non-empty string list")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise HardwareApprovalError("SCHEMA_INVALID", f"{field} must not contain duplicates")
    return result


def _validate_identity(value: Any, *, field: str) -> None:
    if not isinstance(value, dict) or set(value) != _IDENTITY_KEYS:
        actual = set(value) if isinstance(value, dict) else set()
        raise HardwareApprovalError(
            "SCHEMA_INVALID",
            f"{field} fields mismatch: unknown={sorted(actual - _IDENTITY_KEYS)} "
            f"missing={sorted(_IDENTITY_KEYS - actual)}",
        )
    _text(value["product"], field=f"{field}.product")
    serial = value["serial"]
    if serial is not None:
        _text(serial, field=f"{field}.serial")
    _string_list(value["firmware_versions"], field=f"{field}.firmware_versions")


def _validate_approved_envelope(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != _APPROVED_ENVELOPE_KEYS:
        actual = set(value) if isinstance(value, dict) else set()
        raise HardwareApprovalError(
            "SCHEMA_INVALID",
            "approved_envelope fields mismatch: "
            f"unknown={sorted(actual - _APPROVED_ENVELOPE_KEYS)} "
            f"missing={sorted(_APPROVED_ENVELOPE_KEYS - actual)}",
        )
    if type(value["gps_required"]) is not bool:
        raise HardwareApprovalError(
            "SCHEMA_INVALID", "approved_envelope.gps_required must be boolean"
        )
    if value["gps_required"] is not False:
        raise HardwareApprovalError(
            "GPS_POLICY_UNSUPPORTED", "AUTO hardware approval must set gps_required=false"
        )
    for key in sorted(_APPROVED_ENVELOPE_KEYS - {"gps_required"}):
        item = value[key]
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise HardwareApprovalError(
                "SCHEMA_INVALID", f"approved_envelope.{key} must be finite numeric"
            )
        if float(item) <= 0.0:
            raise HardwareApprovalError("SCHEMA_INVALID", f"approved_envelope.{key} must be > 0")


def _validate_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(value)
    if set(receipt) != _RECEIPT_KEYS:
        raise HardwareApprovalError(
            "SCHEMA_INVALID",
            "receipt fields mismatch: "
            f"unknown={sorted(set(receipt) - _RECEIPT_KEYS)} "
            f"missing={sorted(_RECEIPT_KEYS - set(receipt))}",
        )
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise HardwareApprovalError("SCHEMA_UNSUPPORTED", f"expected {RECEIPT_SCHEMA}")
    if type(receipt.get("approved")) is not bool or receipt["approved"] is not True:
        raise HardwareApprovalError("RECEIPT_NOT_APPROVED", "receipt approved must be literal true")
    if _text(receipt.get("site_id"), field="site_id") == "":
        raise HardwareApprovalError("SCHEMA_INVALID", "site_id is empty")
    _text(receipt.get("coordinate_frame_id"), field="coordinate_frame_id")
    _sha256(receipt.get("profile_sha256"), field="profile_sha256")
    _sha256(receipt.get("route_sha256"), field="route_sha256")
    _sha256(receipt.get("bundle_sha256"), field="bundle_sha256")
    if receipt.get("approved_mode") != "auto":
        raise HardwareApprovalError(
            "APPROVAL_MODE_MISMATCH", "hardware approval must explicitly approve mode auto"
        )
    _validate_identity(receipt.get("aircraft"), field="aircraft")
    _validate_identity(receipt.get("controller"), field="controller")
    _string_list(receipt.get("olympe_versions"), field="olympe_versions")
    _validate_approved_envelope(receipt.get("approved_envelope"))
    _text(receipt.get("approval_note"), field="approval_note")
    return receipt


def _decode_public_key(value: Any) -> bytes:
    if not isinstance(value, str) or len(value) != 64:
        raise HardwareApprovalError("TRUST_STORE_INVALID", "public_key must be 32-byte hex")
    try:
        public_key = bytes.fromhex(value)
    except ValueError as exc:
        raise HardwareApprovalError("TRUST_STORE_INVALID", "public_key must be hex") from exc
    if len(public_key) != 32:
        raise HardwareApprovalError("TRUST_STORE_INVALID", "public_key must be 32 bytes")
    return public_key


def _validate_trust_store_shape(raw: Mapping[str, Any]) -> list[Any]:
    if set(raw) != _TRUST_STORE_KEYS:
        raise HardwareApprovalError(
            "TRUST_STORE_INVALID",
            f"trust store fields mismatch: unknown={sorted(set(raw) - _TRUST_STORE_KEYS)} "
            f"missing={sorted(_TRUST_STORE_KEYS - set(raw))}",
        )
    if raw.get("schema") != TRUST_STORE_SCHEMA:
        raise HardwareApprovalError("TRUST_STORE_INVALID", f"expected {TRUST_STORE_SCHEMA}")
    entries = raw.get("keys")
    if not isinstance(entries, list):
        raise HardwareApprovalError("TRUST_STORE_INVALID", "trust store keys must be a list")
    return entries


def _parse_trusted_key(
    entry: Any, *, index: int, known_key_ids: set[str]
) -> TrustedHardwareApprovalKey:
    if not isinstance(entry, dict):
        raise HardwareApprovalError(
            "TRUST_STORE_INVALID", f"trust store key[{index}] is not an object"
        )
    allowed = _TRUSTED_KEY_REQUIRED | _TRUSTED_KEY_OPTIONAL
    if not _TRUSTED_KEY_REQUIRED <= set(entry) or not set(entry) <= allowed:
        raise HardwareApprovalError(
            "TRUST_STORE_INVALID", f"trust store key[{index}] fields mismatch"
        )
    key_id = _text(entry.get("key_id"), field=f"trust store key[{index}].key_id")
    if key_id in known_key_ids:
        raise HardwareApprovalError(
            "TRUST_STORE_INVALID", f"duplicate key_id: {key_id}"
        )
    algorithm = entry.get("algorithm", "Ed25519")
    if algorithm != "Ed25519":
        raise HardwareApprovalError(
            "TRUST_STORE_INVALID", f"unsupported algorithm for {key_id}"
        )
    role = _text(entry.get("role"), field=f"trust store key[{index}].role")
    permissions = _string_list(
        entry.get("permissions"), field=f"trust store key[{index}].permissions"
    )
    not_before = _parse_time(
        entry.get("not_before"), field=f"trust store key[{index}].not_before"
    )
    not_after = _parse_time(
        entry.get("not_after"), field=f"trust store key[{index}].not_after"
    )
    if not_before >= not_after:
        raise HardwareApprovalError(
            "TRUST_STORE_INVALID", f"invalid validity window for {key_id}"
        )
    revoked = entry.get("revoked")
    if type(revoked) is not bool:
        raise HardwareApprovalError(
            "TRUST_STORE_INVALID", f"revoked must be boolean for {key_id}"
        )
    revoked_at = entry.get("revoked_at")
    parsed_revoked_at = (
        None
        if revoked_at is None
        else _parse_time(revoked_at, field=f"trust store key[{index}].revoked_at")
    )
    if parsed_revoked_at is not None and not revoked:
        raise HardwareApprovalError(
            "TRUST_STORE_INVALID", f"revoked_at requires revoked=true for {key_id}"
        )
    return TrustedHardwareApprovalKey(
        key_id=key_id,
        public_key=_decode_public_key(entry.get("public_key")),
        role=role,
        permissions=permissions,
        not_before=not_before,
        not_after=not_after,
        revoked=revoked,
        revoked_at=parsed_revoked_at,
    )


def _load_trust_store(
    source: Mapping[str, Any] | str | Path,
) -> dict[str, TrustedHardwareApprovalKey]:
    raw = _read_json(source, label="trust store")
    entries = _validate_trust_store_shape(raw)
    result: dict[str, TrustedHardwareApprovalKey] = {}
    for index, entry in enumerate(entries):
        trusted_key = _parse_trusted_key(
            entry, index=index, known_key_ids=set(result)
        )
        result[trusted_key.key_id] = trusted_key
    return result


def _validate_envelope(
    value: Mapping[str, Any], *, require_signature: bool,
) -> tuple[dict[str, Any], datetime, datetime, str]:
    envelope = dict(value)
    expected = _SIGNED_ENVELOPE_KEYS if require_signature else _ENVELOPE_KEYS
    if set(envelope) != expected:
        raise HardwareApprovalError(
            "SIGNATURE_ENVELOPE_INVALID",
            f"envelope fields mismatch: unknown={sorted(set(envelope) - expected)} "
            f"missing={sorted(expected - set(envelope))}",
        )
    if envelope.get("schema") != ENVELOPE_SCHEMA:
        raise HardwareApprovalError("SCHEMA_UNSUPPORTED", f"expected {ENVELOPE_SCHEMA}")
    receipt = envelope.get("receipt")
    if not isinstance(receipt, dict):
        raise HardwareApprovalError(
            "SIGNATURE_ENVELOPE_INVALID", "envelope receipt must be an object"
        )
    _validate_receipt(receipt)
    purpose = _text(envelope.get("purpose"), field="purpose")
    if purpose != AUTO_APPROVAL_PURPOSE:
        raise HardwareApprovalError("PURPOSE_MISMATCH", f"expected purpose {AUTO_APPROVAL_PURPOSE}")
    key_id = _text(envelope.get("key_id"), field="key_id")
    issued_at = _parse_time(envelope.get("issued_at"), field="issued_at")
    expires_at = _parse_time(envelope.get("expires_at"), field="expires_at")
    if issued_at >= expires_at:
        raise HardwareApprovalError("SIGNATURE_TIME_INVALID", "issued_at must be before expires_at")
    if require_signature:
        _decode_signature(envelope.get("signature_b64"))
    return envelope, issued_at, expires_at, key_id


def _decode_signature(value: Any) -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise HardwareApprovalError("SIGNATURE_INVALID", "signature_b64 must be non-empty base64")
    try:
        signature = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise HardwareApprovalError("SIGNATURE_INVALID", "signature_b64 is invalid base64") from exc
    if len(signature) != _SIGNATURE_SIZE:
        raise HardwareApprovalError("SIGNATURE_INVALID", "Ed25519 signature must be 64 bytes")
    return signature


def _unsigned_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in envelope.items() if key != "signature_b64"}


def attach_detached_signature(envelope: Mapping[str, Any], signature: bytes) -> dict[str, Any]:
    """Attach externally-produced signature bytes for a detached sidecar file."""
    if not isinstance(signature, bytes) or len(signature) != _SIGNATURE_SIZE:
        raise HardwareApprovalError("SIGNATURE_INVALID", "Ed25519 signature must be 64 bytes")
    unsigned, _, _, _ = _validate_envelope(envelope, require_signature=False)
    result = dict(unsigned)
    result["signature_b64"] = base64.b64encode(signature).decode("ascii")
    return result


def create_unsigned_signing_payload(
    receipt: Mapping[str, Any] | str | Path,
    *,
    key_id: str,
    issued_at: str,
    expires_at: str,
    purpose: str = AUTO_APPROVAL_PURPOSE,
) -> dict[str, Any]:
    """Create canonical signing input; this function never signs or approves."""
    receipt_value = _read_json(receipt, label="receipt")
    _validate_receipt(receipt_value)
    envelope = {
        "schema": ENVELOPE_SCHEMA,
        "receipt": receipt_value,
        "purpose": purpose,
        "key_id": key_id,
        "issued_at": issued_at,
        "expires_at": expires_at,
    }
    _validate_envelope(envelope, require_signature=False)
    return envelope


def _validate_expected_bindings(expected: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(expected, Mapping):
        raise HardwareApprovalError(
            "BINDING_EXPECTATIONS_REQUIRED", "expected_bindings must be an object"
        )
    values = dict(expected)
    unknown = set(values) - (_REQUIRED_BINDINGS | _OPTIONAL_BINDING_FIELDS)
    if unknown:
        raise HardwareApprovalError(
            "BINDING_EXPECTATIONS_INVALID", f"unsupported expected bindings: {sorted(unknown)}"
        )
    missing = _REQUIRED_BINDINGS - set(values)
    if missing:
        raise HardwareApprovalError(
            "BINDING_EXPECTATIONS_REQUIRED", f"missing required bindings: {sorted(missing)}"
        )
    for key in _REQUIRED_BINDINGS:
        value = values[key]
        if not isinstance(value, str) or not value.strip():
            raise HardwareApprovalError("BINDING_EXPECTATIONS_REQUIRED", f"binding {key} is empty")
        if key.endswith("_sha256") and _SHA256.fullmatch(value) is None:
            raise HardwareApprovalError(
                "BINDING_EXPECTATIONS_REQUIRED", f"binding {key} is not SHA-256"
            )
    if "approved_mode" in values and values["approved_mode"] != "auto":
        raise HardwareApprovalError("BINDING_EXPECTATIONS_INVALID", "approved_mode must be auto")
    return values


def _reject_pending_receipt(value: Mapping[str, Any]) -> None:
    if value.get("status") == "pending":
        raise HardwareApprovalError("PENDING_REQUEST", "pending request is never an approval")


def _ensure_receipt_matches(
    validated_receipt: Mapping[str, Any], envelope_receipt: Mapping[str, Any]
) -> None:
    if canonical_json_bytes(envelope_receipt) != canonical_json_bytes(validated_receipt):
        raise HardwareApprovalError(
            "RECEIPT_MISMATCH", "detached envelope does not contain the selected receipt"
        )


def _ensure_bindings_match(
    validated_receipt: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    for field, expected_value in expected.items():
        if field in _RECEIPT_KEYS and validated_receipt.get(field) != expected_value:
            raise HardwareApprovalError(
                "BINDING_MISMATCH",
                f"{field}: expected {expected_value!r}, got {validated_receipt.get(field)!r}",
            )


def _load_trusted_key(
    source: Mapping[str, Any] | str | Path, *, key_id: str
) -> TrustedHardwareApprovalKey:
    trusted_key = _load_trust_store(source).get(key_id)
    if trusted_key is None:
        raise HardwareApprovalError("UNKNOWN_KEY", f"key_id is not in trust store: {key_id}")
    return trusted_key


def _validate_key_window(
    trusted_key: TrustedHardwareApprovalKey,
    *,
    issued_at: datetime,
    expires_at: datetime,
    current: datetime,
    key_id: str,
) -> None:
    if trusted_key.revoked:
        raise HardwareApprovalError("KEY_REVOKED", f"trust-store key is revoked: {key_id}")
    if current < trusted_key.not_before:
        raise HardwareApprovalError(
            "KEY_NOT_YET_VALID", f"trust-store key is not valid yet: {key_id}"
        )
    if current >= trusted_key.not_after:
        raise HardwareApprovalError("KEY_EXPIRED", f"trust-store key expired: {key_id}")
    if issued_at > current:
        raise HardwareApprovalError(
            "RECEIPT_NOT_YET_VALID", "receipt validity starts in the future"
        )
    if current >= expires_at:
        raise HardwareApprovalError("RECEIPT_EXPIRED", "receipt validity window has expired")
    if issued_at < trusted_key.not_before or expires_at > trusted_key.not_after:
        raise HardwareApprovalError(
            "KEY_VALIDITY_MISMATCH", "receipt validity is outside signer key validity"
        )


def _validate_key_permission(trusted_key: TrustedHardwareApprovalKey, *, key_id: str) -> None:
    if _REQUIRED_PERMISSION not in trusted_key.permissions:
        raise HardwareApprovalError(
            "KEY_PERMISSION_DENIED",
            f"key {key_id} is not authorized for AUTO hardware approval",
        )


def _verify_detached_signature(
    signature_value: Mapping[str, Any], trusted_key: TrustedHardwareApprovalKey
) -> None:
    signature_bytes = _decode_signature(signature_value["signature_b64"])
    try:
        public_key = ECC.import_key(_ED25519_SPKI_PREFIX + trusted_key.public_key)
        verifier = eddsa.new(public_key, "rfc8032")
        verifier.verify(
            canonical_json_bytes(_unsigned_envelope(signature_value)), signature_bytes
        )
    except (TypeError, ValueError) as exc:
        raise HardwareApprovalError(
            "SIGNATURE_INVALID", "Ed25519 signature verification failed"
        ) from exc


def verify_hardware_approval_receipt(
    receipt: Mapping[str, Any] | str | Path,
    *,
    signature: Mapping[str, Any] | str | Path,
    trust_store: Mapping[str, Any] | str | Path,
    expected_bindings: Mapping[str, Any],
    now: datetime | None = None,
) -> HardwareApprovalVerdict:
    """Verify a signed AUTO receipt and all site/profile/hardware trust gates.

    The parent readiness gate should treat ``accepted=False`` as a hard blocker.
    ``expected_bindings`` must contain non-empty ``site_id``,
    ``coordinate_frame_id``, ``profile_sha256``, ``route_sha256`` and
    ``bundle_sha256`` values; this prevents a caller from accidentally verifying
    a receipt without binding it to the selected route and profile.
    """
    key_id: str | None = None
    try:
        receipt_value = _read_json(receipt, label="receipt")
        _reject_pending_receipt(receipt_value)
        validated_receipt = _validate_receipt(receipt_value)
        expected = _validate_expected_bindings(expected_bindings)
        signature_value = _read_json(signature, label="detached signature envelope")
        envelope, issued_at, expires_at, key_id = _validate_envelope(
            signature_value, require_signature=True
        )
        _ensure_receipt_matches(validated_receipt, envelope["receipt"])
        _ensure_bindings_match(validated_receipt, expected)
        trusted_key = _load_trusted_key(trust_store, key_id=key_id)
        current = _now(now)
        _validate_key_window(
            trusted_key,
            issued_at=issued_at,
            expires_at=expires_at,
            current=current,
            key_id=key_id,
        )
        _validate_key_permission(trusted_key, key_id=key_id)
        _verify_detached_signature(signature_value, trusted_key)
        return HardwareApprovalVerdict(
            True, "APPROVED", "signed AUTO hardware approval verified", key_id
        )
    except HardwareApprovalError as exc:
        return HardwareApprovalVerdict(False, exc.reason_code, exc.detail, key_id)
    except (OSError, TypeError, ValueError) as exc:
        return HardwareApprovalVerdict(False, "VERIFICATION_ERROR", str(exc), key_id)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise HardwareApprovalError("DISCOVERY_FAILED", f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def discover_profile_bindings(profile: str | Path) -> dict[str, Any]:
    """Discover current site/profile/route/bundle bindings for a request draft.

    Missing files remain ``None`` in the draft and therefore cannot be mistaken
    for an approval.  The profile SHA covers canonical profile content excluding
    only ``hardware_approval`` so adding detached trust references is not circular.
    """
    profile_path = Path(profile).expanduser().resolve()
    raw = _read_json(profile_path, label="site profile")
    site_id = raw.get("site_id")
    flight = raw.get("flight") if isinstance(raw.get("flight"), dict) else {}
    coordinate_frame = flight.get("coordinate_frame_id") or (
        raw.get("coordinate_frame", {}).get("id")
        if isinstance(raw.get("coordinate_frame"), dict)
        else None
    )
    assets = raw.get("assets") if isinstance(raw.get("assets"), dict) else {}
    digests = raw.get("asset_sha256") if isinstance(raw.get("asset_sha256"), dict) else {}

    def asset_digest(name: str, digest_name: str) -> str | None:
        digest = digests.get(digest_name)
        if isinstance(digest, str) and _SHA256.fullmatch(digest):
            return digest
        relative = assets.get(name)
        if not isinstance(relative, str) or not relative.strip():
            return None
        path = (profile_path.parent / relative).resolve()
        return _sha256_file(path) if path.is_file() else None

    approval_subject = dict(raw)
    approval_subject.pop("hardware_approval", None)
    return {
        "site_id": site_id if isinstance(site_id, str) else None,
        "coordinate_frame_id": coordinate_frame if isinstance(coordinate_frame, str) else None,
        "profile_sha256": hashlib.sha256(
            canonical_json_bytes(approval_subject)
        ).hexdigest(),
        "route_sha256": asset_digest("route_json", "route_json"),
        "bundle_sha256": asset_digest("localization_bundle", "localization_bundle"),
    }


def build_pending_request(
    *,
    site_id: str | None,
    coordinate_frame_id: str | None,
    profile_sha256: str | None,
    route_sha256: str | None,
    bundle_sha256: str | None,
) -> dict[str, Any]:
    """Build a non-approval request with discovered values and empty hardware identity."""
    request = {
        "schema": REQUEST_SCHEMA,
        "status": "pending",
        "purpose": AUTO_APPROVAL_PURPOSE,
        "approved": False,
        "approved_mode": "auto",
        "site_id": site_id,
        "coordinate_frame_id": coordinate_frame_id,
        "profile_sha256": profile_sha256,
        "route_sha256": route_sha256,
        "bundle_sha256": bundle_sha256,
        "aircraft": {"product": None, "serial": None, "firmware_versions": []},
        "controller": {"product": None, "serial": None, "firmware_versions": []},
        "olympe_versions": [],
        "approved_envelope": {
            "max_altitude_m": None,
            "max_distance_m": None,
            "max_tilt_deg": None,
            "max_vertical_speed_ms": None,
            "max_rotation_speed_degs": None,
            "gps_required": False,
        },
        "approval_note": (
            "REQUEST ONLY: pending field hardware identity, firmware, safety envelope, "
            "operator evidence, and detached Ed25519 approval. Never use this file as approval."
        ),
    }
    return request
