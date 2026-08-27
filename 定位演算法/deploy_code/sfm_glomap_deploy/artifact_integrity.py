"""SHA-256 verification for executable model artifacts."""
from __future__ import annotations

import hashlib
import hmac
from pathlib import Path


KNOWN_SHA256 = {
    "edm_outdoor.ckpt": "f686bebdd9705bf6918621a1a83695f83d698cbd8c3eed932847fe3678d13a97",
    "current_reloc_map_updated_v3.pt": "8227e3bd37d4d99966ae1fb307060f9bc8e38e6e5269ab918f64b87957c6ccab",
    # river_site XFeat transfer package (地圖檔/場域/river_site/bundles/xfeat)
    "river_site_reloc_map_xfeat_tri.pt": "4a51fb1ab8157a44fca5831a2b6f2a1c5c0a90654557308765537af2c8f4a10a",
    "target_site_v1_reloc_map_edm.pt": "b32866d6595ca30b89e7cf6cef6dacbb4692f3c4992b759d833488c31b489d05",
    "river_site_reloc_map_edm.pt": "39a817936c0ba314a739701411f974672a94f126830f9f5b9a7a4efdfee08117",
    "edm_production_profile.json": "f5a486b922f7c4a5671daf867f630e9dd614cb68ac7aa1d6e028324e8c23b3e6",
    "river_site.json": "724e35e0340ad1d694ad785d24617bddf17d5ea820796fba9e537790234e730d",
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
