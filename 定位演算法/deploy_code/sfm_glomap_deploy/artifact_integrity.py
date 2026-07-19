"""SHA-256 verification for executable model artifacts."""
from __future__ import annotations

import hashlib
import hmac
from pathlib import Path


KNOWN_SHA256 = {
    "current_reloc_map_updated_v3.pt": "8227e3bd37d4d99966ae1fb307060f9bc8e38e6e5269ab918f64b87957c6ccab",
    "football_field_reloc_map_xfeat_tri.pt": "1c1774318a71ac29870f78ccb67001150edd934141e4aa288765e745e72db46f",
    # river_site XFeat transfer package (河濱測試/xfeaat+lightglue分支)
    "river_site_reloc_map_xfeat_tri.pt": "4a51fb1ab8157a44fca5831a2b6f2a1c5c0a90654557308765537af2c8f4a10a",
}


def expected_sha256(path: str | Path, explicit: str | None = None) -> str:
    value = explicit or KNOWN_SHA256.get(Path(path).name, "")
    value = value.strip().lower()
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(
            f"no trusted SHA-256 for {path}; pass an explicit 64-character digest"
        )
    return value


def verify_sha256(path: str | Path, expected: str | None = None) -> str:
    artifact = Path(path)
    trusted = expected_sha256(artifact, expected)
    digest = hashlib.sha256()
    with artifact.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if not hmac.compare_digest(actual, trusted):
        raise ValueError(
            f"SHA-256 mismatch for {artifact}: expected {trusted}, got {actual}"
        )
    return actual
