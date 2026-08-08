from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "check_runtime_mirrors.py"
SPEC = importlib.util.spec_from_file_location("check_runtime_mirrors", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
check_runtime_mirrors = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_runtime_mirrors)


def test_retained_runtime_mirrors_are_identical():
    assert check_runtime_mirrors.check_mirrors() == []


ALGO_ROOT = Path(__file__).resolve().parents[2]
FLIGHT_CONTROL = ALGO_ROOT / "flight_control"
DEPLOY = ALGO_ROOT / "deploy_code" / "sfm_glomap_deploy"

#: Same-named files that MIRROR_PAIRS deliberately does not cover, and why.
#: "identical" ones are unenforced only by omission -- this test closes that gap.
#: "divergent" ones are intentional; they are listed so that a NEW same-named
#: file cannot appear without someone classifying it.
INTENTIONALLY_DIVERGENT = check_runtime_mirrors.INTENTIONALLY_DIVERGENT

#: Identical today but NOT in MIRROR_PAIRS. Audit 2026-08-07 finding F-06: the
#: deploy copy of manual_nudge_pilot.py is also untracked in git, so a one-sided
#: edit is invisible to git diff, to check_runtime_mirrors and to MANIFEST.tsv.
#: Adding it to MIRROR_PAIRS would make check_mirrors() fail on a fresh clone
#: (the mirror would be missing), so the ownership decision -- track the deploy
#: copy, or delete it -- is left to the operator. Until then this list keeps the
#: byte-comparison below running over it.
UNENFORCED_BUT_MUST_MATCH = check_runtime_mirrors.UNENFORCED_BUT_MUST_MATCH


def _same_named_files() -> set[str]:
    return check_runtime_mirrors.same_named_files(ALGO_ROOT)


def test_every_same_named_file_is_either_enforced_or_declared_divergent():
    """A same-named file in both trees must be covered by SOMETHING.

    MIRROR_PAIRS enforces 8 pairs, but 12 files share a name. The uncovered
    ones drift silently: MANIFEST.tsv hashes the two copies independently and
    regenerates both in lockstep, mtimes are already identical across a 332-line
    divergence, and one deploy copy is not even tracked in git. This test makes
    "nobody classified this file" a failure instead of a discovery.
    """
    enforced = set()
    for owner_rel, mirror_rel in check_runtime_mirrors.MIRROR_PAIRS:
        enforced.add(Path(owner_rel).name)
        enforced.add(Path(mirror_rel).name)

    classified = enforced | set(INTENTIONALLY_DIVERGENT) | set(UNENFORCED_BUT_MUST_MATCH)
    unclassified = _same_named_files() - classified
    assert not unclassified, (
        "same-named files in flight_control/ and deploy_code/sfm_glomap_deploy/ "
        f"that nobody has classified: {sorted(unclassified)}. Add them to "
        "MIRROR_PAIRS, to INTENTIONALLY_DIVERGENT, or to UNENFORCED_BUT_MUST_MATCH."
    )


def test_unenforced_identical_copies_have_not_drifted():
    """Byte-compare the same-named files that MIRROR_PAIRS does not cover.

    manual_nudge_pilot.py is the concrete case: 705 lines, identical today, the
    source of the operator UI's nudge constants, in neither enforcement list,
    and untracked in git on the deploy side. A one-sided edit to it is invisible
    to git diff, to check_runtime_mirrors and to MANIFEST.tsv -- the only
    discovery mechanism is a human remembering to diff a directory pair. This
    test is that mechanism.
    """
    drifted = []
    for name in sorted(UNENFORCED_BUT_MUST_MATCH):
        owner, mirror = FLIGHT_CONTROL / name, DEPLOY / name
        if not owner.exists() or not mirror.exists():
            # One side absent is a tracking question, not drift; the
            # classification test above is what guards the file list itself.
            continue
        if owner.read_bytes() != mirror.read_bytes():
            drifted.append(name)
    assert not drifted, (
        f"unenforced mirrored files drifted: {drifted}. These are byte-identical "
        "by policy but no production check covers them -- fix both copies."
    )


def test_divergent_frame_sources_share_live_safety_contract():
    """The frame-source implementations may differ in stream integration, but
    neither may expose the retired legacy arming path or unbounded waits.
    """
    sources = [
        (FLIGHT_CONTROL / "olympe_frame_source.py").read_text(encoding="utf-8"),
        (DEPLOY / "olympe_frame_source.py").read_text(encoding="utf-8"),
    ]
    for source in sources:
        assert "SFM_ALLOW_LEGACY_FLIGHT" not in source
        assert "TakeOff()" not in source
        assert ".wait()" not in source
        assert ".join()" not in source
        assert ".wait(timeout=0.5)" in source
        assert ".join(timeout=5.0)" in source
        assert "require_source_timestamps" in source
        assert "_source_timing_trusted_locked" in source
