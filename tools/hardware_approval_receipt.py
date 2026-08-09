#!/usr/bin/env python3
"""Verify signed hardware AUTO receipts or prepare unsigned operator material.

This command intentionally has no signing operation.  A field operator must
sign the canonical payload with a separately controlled private key, then place
the detached JSON envelope in the deployment bundle.  ``create-request`` and
``create-payload`` only write drafts and cannot unlock AUTO.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OPERATOR_INTERFACE = ROOT / "控制介面程式" / "operator_interface"
if str(OPERATOR_INTERFACE) not in sys.path:
    sys.path.insert(0, str(OPERATOR_INTERFACE))

from hardware_approval_trust import (  # noqa: E402
    HardwareApprovalError,
    build_pending_request,
    canonical_json_bytes,
    create_unsigned_signing_payload,
    discover_profile_bindings,
    verify_hardware_approval_receipt,
)


def _parse_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _parse_timestamp(value: str) -> str:
    if not value.endswith("Z"):
        raise ValueError("timestamp must use UTC Z notation")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"invalid UTC timestamp: {value}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_canonical(value: dict[str, Any], output: str) -> None:
    content = canonical_json_bytes(value) + b"\n"
    if output == "-":
        sys.stdout.buffer.write(content)
        return
    path = Path(output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"refusing to overwrite existing output: {path}")
    path.write_bytes(content)


def _verify(args: argparse.Namespace) -> int:
    expected = _parse_json_object(Path(args.expected_bindings))
    now = None if args.now is None else datetime.fromisoformat(_parse_timestamp(args.now)[:-1] + "+00:00")
    verdict = verify_hardware_approval_receipt(
        Path(args.receipt),
        signature=Path(args.signature),
        trust_store=Path(args.trust_store),
        expected_bindings=expected,
        now=now,
    )
    print(json.dumps(verdict.as_dict(), ensure_ascii=False, sort_keys=True))
    return 0 if verdict.accepted else 1


def _create_request(args: argparse.Namespace) -> int:
    bindings = discover_profile_bindings(args.profile)
    request = build_pending_request(**bindings)
    _write_canonical(request, args.output)
    return 0


def _create_payload(args: argparse.Namespace) -> int:
    payload = create_unsigned_signing_payload(
        Path(args.receipt),
        key_id=args.key_id,
        issued_at=_parse_timestamp(args.issued_at),
        expires_at=_parse_timestamp(args.expires_at),
    )
    _write_canonical(payload, args.output)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify", help="verify an existing signed AUTO receipt")
    verify.add_argument("--receipt", required=True, help="existing anafi-hardware-approval/v2 JSON")
    verify.add_argument("--signature", required=True, help="detached signed envelope JSON")
    verify.add_argument("--trust-store", required=True, help="operator-provisioned trust-store JSON")
    verify.add_argument(
        "--expected-bindings",
        required=True,
        help="JSON object containing site_id, coordinate_frame_id and asset SHA-256 values",
    )
    verify.add_argument("--now", help="verification time in UTC Z notation; defaults to current time")
    verify.set_defaults(handler=_verify)

    request = commands.add_parser("create-request", help="create a pending, unsigned request draft")
    request.add_argument("--profile", required=True, help="site profile JSON to inspect")
    request.add_argument("--output", default="-", help="output path, or - for stdout")
    request.set_defaults(handler=_create_request)

    payload = commands.add_parser(
        "create-payload", help="create unsigned canonical signing input; does not sign or approve"
    )
    payload.add_argument("--receipt", required=True, help="existing approved receipt JSON")
    payload.add_argument("--key-id", required=True, help="trust-store key_id selected by the operator")
    payload.add_argument("--issued-at", required=True, help="validity start in UTC Z notation")
    payload.add_argument("--expires-at", required=True, help="validity end in UTC Z notation")
    payload.add_argument("--output", default="-", help="output path, or - for stdout")
    payload.set_defaults(handler=_create_payload)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except (HardwareApprovalError, OSError, ValueError) as exc:
        print(f"hardware approval command failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
