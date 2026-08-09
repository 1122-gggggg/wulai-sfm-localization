# Optional hardware approval trust chain

`hardware_approval_trust.py` verifies an existing
`anafi-hardware-approval/v2` receipt for AUTO with an Ed25519 detached
signature.  The signed envelope binds the exact receipt, purpose, signer
`key_id`, and validity interval.  The operator-provisioned trust store binds
that key ID to a public key, role, validity window, revocation state, and the
`hardware_approval:auto` permission.

The verifier remains available when a deployment wants an independently signed
hardware audit record:

```python
verdict = verify_hardware_approval_receipt(
    receipt_path,
    signature=signature_envelope_path,
    trust_store=trust_store_path,
    expected_bindings={
        "site_id": profile.site_id,
        "coordinate_frame_id": profile.flight.coordinate_frame_id,
        "profile_sha256": profile_digest,
        "route_sha256": profile.asset_sha256.route_json,
        "bundle_sha256": profile.asset_sha256.localization_bundle,
    },
)
if not verdict.accepted:
    audit_warnings.append(
        f"hardware approval trust chain: {verdict.reason_code}: {verdict.detail}"
    )
```

An invalid optional receipt must never be reported as valid, but an absent receipt
is not a runtime readiness error. GPS is intentionally represented by
`approved_envelope.gps_required`; the current river deployment does not require
GPS for takeoff or AUTO, and disables the GPS-dependent distance fence when no fix
exists. This policy does not bypass route/profile hashes, four-step preflight, or
post-takeoff localization gates.

The CLI only verifies or prepares unsigned material:

```bash
python tools/hardware_approval_receipt.py create-request \
  --profile 控制介面程式/site_profiles/river_site_edm.json \
  --output river_site_receipt_v2.request.json

python tools/hardware_approval_receipt.py create-payload \
  --receipt approved_receipt.json \
  --key-id field-authority-001 \
  --issued-at 2026-08-09T12:00:00Z \
  --expires-at 2026-08-09T18:00:00Z \
  --output unsigned-envelope.json
```

No private key, trust-store production key, or hardware approval evidence is
stored in this repository.  A request draft is never accepted as a receipt.
`profile_sha256` is the SHA-256 of canonical site-profile JSON excluding only
the `hardware_approval` object. This avoids a circular digest while still
binding all map, route, localizer, coordinate-frame and flight-policy fields.
Any change outside that detached-reference object requires a new signed receipt.
