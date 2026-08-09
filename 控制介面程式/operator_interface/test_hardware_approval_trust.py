from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from Cryptodome.PublicKey import ECC
from Cryptodome.Signature import eddsa

from hardware_approval_trust import (
    ENVELOPE_SCHEMA,
    RECEIPT_SCHEMA,
    REQUEST_SCHEMA,
    TRUST_STORE_SCHEMA,
    attach_detached_signature,
    build_pending_request,
    canonical_json_bytes,
    create_unsigned_signing_payload,
    discover_profile_bindings,
    verify_hardware_approval_receipt,
)


NOW = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)
EXPECTED = {
    "site_id": "river_site_edm",
    "coordinate_frame_id": "river_site_glomap",
    "profile_sha256": "a" * 64,
    "route_sha256": "b" * 64,
    "bundle_sha256": "c" * 64,
}


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _receipt(**overrides: object) -> dict:
    value = {
        "schema": RECEIPT_SCHEMA,
        "approved": True,
        "site_id": EXPECTED["site_id"],
        "coordinate_frame_id": EXPECTED["coordinate_frame_id"],
        "profile_sha256": EXPECTED["profile_sha256"],
        "route_sha256": EXPECTED["route_sha256"],
        "bundle_sha256": EXPECTED["bundle_sha256"],
        "approved_mode": "auto",
        "aircraft": {
            "product": "ANAFI",
            "serial": "ANAFI-TEST-001",
            "firmware_versions": ["1.8.2"],
        },
        "controller": {
            "product": "SkyController 3",
            "serial": "SC3-TEST-001",
            "firmware_versions": ["1.8.1"],
        },
        "olympe_versions": ["8.4.0"],
        "approved_envelope": {
            "max_altitude_m": 20.0,
            "max_distance_m": 50.0,
            "max_tilt_deg": 15.0,
            "max_vertical_speed_ms": 2.0,
            "max_rotation_speed_degs": 20.0,
            "gps_required": False,
        },
        "approval_note": "Field operator evidence receipt; AUTO scope explicitly reviewed.",
    }
    value.update(overrides)
    return value


def _trust_store(
    key: ECC.EccKey,
    *,
    permissions: list[str] | None = None,
    revoked: bool = False,
    not_before: datetime = NOW - timedelta(days=1),
    not_after: datetime = NOW + timedelta(days=1),
) -> dict:
    public = key.public_key().export_key(format="raw")
    return {
        "schema": TRUST_STORE_SCHEMA,
        "keys": [
            {
                "key_id": "field-authority-001",
                "public_key": public.hex(),
                "role": "field_hardware_authority",
                "permissions": permissions or ["hardware_approval:auto"],
                "not_before": _timestamp(not_before),
                "not_after": _timestamp(not_after),
                "revoked": revoked,
            }
        ],
    }


def _signed_material(
    *,
    receipt: dict | None = None,
    key: ECC.EccKey | None = None,
    issued_at: datetime = NOW - timedelta(minutes=5),
    expires_at: datetime = NOW + timedelta(hours=1),
) -> tuple[dict, dict, ECC.EccKey]:
    signing_key = key or ECC.generate(curve="ed25519")
    payload = create_unsigned_signing_payload(
        receipt or _receipt(),
        key_id="field-authority-001",
        issued_at=_timestamp(issued_at),
        expires_at=_timestamp(expires_at),
    )
    signature = eddsa.new(signing_key, "rfc8032").sign(canonical_json_bytes(payload))
    return payload["receipt"], attach_detached_signature(payload, signature), signing_key


def _verify(receipt: dict, envelope: dict, trust: dict, *, now: datetime = NOW):
    return verify_hardware_approval_receipt(
        receipt,
        signature=envelope,
        trust_store=trust,
        expected_bindings=EXPECTED,
        now=now,
    )


def test_valid_ephemeral_ed25519_receipt_is_accepted() -> None:
    receipt, envelope, key = _signed_material()

    verdict = _verify(receipt, envelope, _trust_store(key))

    assert verdict.accepted is True
    assert verdict.reason_code == "APPROVED"
    assert verdict.key_id == "field-authority-001"


def test_tampering_receipt_or_detached_metadata_fails_closed() -> None:
    receipt, envelope, key = _signed_material()
    tampered_receipt = dict(receipt, route_sha256="d" * 64)
    receipt_verdict = _verify(tampered_receipt, envelope, _trust_store(key))
    assert receipt_verdict.accepted is False
    assert receipt_verdict.reason_code == "RECEIPT_MISMATCH"

    tampered_envelope = dict(envelope, expires_at=_timestamp(NOW + timedelta(hours=2)))
    metadata_verdict = _verify(receipt, tampered_envelope, _trust_store(key))
    assert metadata_verdict.accepted is False
    assert metadata_verdict.reason_code == "SIGNATURE_INVALID"


def test_expired_and_revoked_keys_are_rejected() -> None:
    receipt, envelope, key = _signed_material()

    expired = _verify(
        receipt,
        envelope,
        _trust_store(key, not_after=NOW - timedelta(seconds=1)),
    )
    assert expired.accepted is False
    assert expired.reason_code == "KEY_EXPIRED"

    revoked = _verify(receipt, envelope, _trust_store(key, revoked=True))
    assert revoked.accepted is False
    assert revoked.reason_code == "KEY_REVOKED"

    _, expired_receipt_envelope, expired_receipt_key = _signed_material(
        issued_at=NOW - timedelta(hours=2),
        expires_at=NOW - timedelta(seconds=1),
    )
    expired_receipt = expired_receipt_envelope["receipt"]
    expired_receipt_verdict = _verify(
        expired_receipt,
        expired_receipt_envelope,
        _trust_store(expired_receipt_key),
    )
    assert expired_receipt_verdict.accepted is False
    assert expired_receipt_verdict.reason_code == "RECEIPT_EXPIRED"


def test_unknown_not_yet_valid_and_unauthorized_keys_are_rejected() -> None:
    receipt, envelope, key = _signed_material()

    unknown = _verify(receipt, envelope, {"schema": TRUST_STORE_SCHEMA, "keys": []})
    assert unknown.accepted is False
    assert unknown.reason_code == "UNKNOWN_KEY"

    future = _verify(
        receipt,
        envelope,
        _trust_store(
            key,
            not_before=NOW + timedelta(minutes=1),
            not_after=NOW + timedelta(days=1),
        ),
    )
    assert future.accepted is False
    assert future.reason_code == "KEY_NOT_YET_VALID"

    unauthorized = _verify(
        receipt,
        envelope,
        _trust_store(key, permissions=["hardware_approval:manual"]),
    )
    assert unauthorized.accepted is False
    assert unauthorized.reason_code == "KEY_PERMISSION_DENIED"


def test_wrong_binding_and_purpose_do_not_verify() -> None:
    receipt, envelope, key = _signed_material()
    wrong_binding = _verify(
        receipt,
        envelope,
        _trust_store(key),
    )
    assert wrong_binding.accepted is True
    wrong_expected = dict(EXPECTED, site_id="another-site")
    wrong_binding = verify_hardware_approval_receipt(
        receipt,
        signature=envelope,
        trust_store=_trust_store(key),
        expected_bindings=wrong_expected,
        now=NOW,
    )
    assert wrong_binding.accepted is False
    assert wrong_binding.reason_code == "BINDING_MISMATCH"

    bad_purpose = dict(envelope, purpose="hardware_approval:manual")
    purpose_verdict = _verify(receipt, bad_purpose, _trust_store(key))
    assert purpose_verdict.accepted is False
    assert purpose_verdict.reason_code == "PURPOSE_MISMATCH"


def test_pending_request_is_explicitly_not_an_approval() -> None:
    request = build_pending_request(**EXPECTED)
    assert request["schema"] == REQUEST_SCHEMA
    assert request["status"] == "pending"
    assert request["approved"] is False

    receipt, envelope, key = _signed_material()
    verdict = verify_hardware_approval_receipt(
        request,
        signature=envelope,
        trust_store=_trust_store(key),
        expected_bindings=EXPECTED,
        now=NOW,
    )
    assert verdict.accepted is False
    assert verdict.reason_code == "PENDING_REQUEST"


def test_profile_binding_discovery_hashes_existing_assets(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "route.json").write_text("route", encoding="utf-8")
    (assets / "bundle.pt").write_bytes(b"bundle")
    profile = tmp_path / "site.json"
    profile.write_text(
        json.dumps(
            {
                "site_id": "site-a",
                "flight": {"coordinate_frame_id": "frame-a"},
                "assets": {
                    "route_json": "assets/route.json",
                    "localization_bundle": "assets/bundle.pt",
                },
                "asset_sha256": {},
            }
        ),
        encoding="utf-8",
    )

    bindings = discover_profile_bindings(profile)

    assert bindings["site_id"] == "site-a"
    assert bindings["coordinate_frame_id"] == "frame-a"
    assert bindings["profile_sha256"] is not None
    assert bindings["route_sha256"] is not None
    assert bindings["bundle_sha256"] is not None


def test_canonical_payload_is_stable_and_detached() -> None:
    receipt, envelope, _ = _signed_material()
    assert envelope["schema"] == ENVELOPE_SCHEMA
    assert "signature_b64" not in create_unsigned_signing_payload(
        receipt,
        key_id="field-authority-001",
        issued_at=envelope["issued_at"],
        expires_at=envelope["expires_at"],
    )
    assert base64.b64decode(envelope["signature_b64"])  # sidecar is detached data
    assert canonical_json_bytes(envelope) == canonical_json_bytes(json.loads(json.dumps(envelope)))
