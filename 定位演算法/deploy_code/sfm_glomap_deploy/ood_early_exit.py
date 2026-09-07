"""Out-of-distribution early exit for the EDM localizer (R2).

Why this exists
---------------
Every gate on the pose path today measures *consistency*: inlier count,
reprojection RMS, inlier ratio / grid spread, jump and yaw limits, gravity
(``gravity_roll_gate``), trajectory plausibility (``trajectory_plausibility``).
All of them ask "is this pose self-consistent?" -- none of them asks "is this
query even in this map?".  A query photographed at another site still produces
a MegaLoc ranking, still produces EDM correspondences, and RANSAC will still
return *a* pose; the only thing standing between that and a published pose is
the inlier floor, which is a magnitude threshold, not an identity check.

This module answers the identity question with two channels that fail for
different reasons, so a single failure mode cannot fool both:

* **VPR channel** (retrieval layer).  Reads the MegaLoc cosine scores of the
  candidate pool the tracker already computed: absolute ``top1``, the
  ``top1 - top2`` margin, and the normalised entropy of the pool.  An off-map
  query has a *low and flat* ranking -- nothing in the map looks like it, so no
  reference wins.  An in-map query has a peaked ranking.
* **Matcher channel** (correspondence layer).  Reads the EDM correspondence
  rows the tracker already collected: best-reference correspondence count and
  the confidence distribution.  An off-map query yields few correspondences on
  its best reference, and the ones it yields are low confidence.

Contract
--------
* **Veto only.**  This module never produces, repairs, or rescores a pose.
  ``HARD`` reports LOST; ``SOFT`` only tightens the existing acceptance floor.
* **Fail open.**  Any missing, empty, or non-finite feature -> ``abstain``.  A
  check that cannot be evaluated must not kill a possibly-good frame.
* **Abstain without VPR.**  Tracking frames pick references geometrically and
  never run MegaLoc, so the VPR channel has no data.  A single channel is not
  enough evidence to declare "not in this map", so the verdict abstains: the
  dangerous case (acquiring a pose from nothing, i.e. BOOT/LOST) is exactly the
  case where retrieval *does* run.
* **Conjunction, not disjunction.**  ``HARD`` requires BOTH channels to vote
  out-of-map.  That is what buys ``FAR = 0`` at a low false-rejection rate:
  each channel alone has in-map frames deep in its out-of-map region.
* **Default audit-only.**  ``SFM_EDM_OOD_MODE`` defaults to ``audit``: the
  verdict is computed and recorded, and nothing acts on it.  No PnP call is
  skipped, no threshold moves, no pose changes.  The audit reads arrays the
  tracker already has in hand -- it runs no model, allocates no GPU work, and
  forces no device sync.

Modes (``SFM_EDM_OOD_MODE``)
---------------------------
``off``    -- not evaluated at all.
``audit``  -- evaluated and recorded; never acts (default).
``soft``   -- a ``soft`` or ``hard`` verdict tightens the inlier floor.
``hard``   -- a ``hard`` verdict skips PnP for that frame and reports LOST;
              a ``soft`` verdict still only tightens.

Thresholds are read from the environment once, at tracker construction, so a
run's operating point is a property of that run and lands in its receipt.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

import numpy as np


MODE_OFF = "off"
MODE_AUDIT = "audit"
MODE_SOFT = "soft"
MODE_HARD = "hard"
MODES = (MODE_OFF, MODE_AUDIT, MODE_SOFT, MODE_HARD)

#: Channel votes. ``unknown`` means "no data", never "in map".
VOTE_UNKNOWN = "unknown"
VOTE_IN = "in"
VOTE_OUT = "out"

DECISION_ABSTAIN = "abstain"
DECISION_GO = "go"
DECISION_SOFT = "soft"
DECISION_HARD = "hard"

#: What the active mode actually did with the verdict.
APPLIED_NONE = "none"
APPLIED_AUDIT = "audit"
APPLIED_TIGHTEN = "tighten"
APPLIED_VETO = "veto"

#: Softmax temperature for the retrieval-pool entropy. BoQ scores are
#: cosine similarities of L2-normalised descriptors, and the in-map top1/top2
#: gap on the river map is O(0.01-0.05), so the temperature has to be that
#: scale for the entropy to separate a peaked pool from a flat one.
DEFAULT_ENTROPY_TEMPERATURE = 0.05
#: Minimum retrieval-pool size for the VPR channel. With fewer scores there is
#: no margin and no meaningful entropy, so the channel abstains.
MIN_VPR_POOL = 3


@dataclass(frozen=True)
class OODThresholds:
    """One operating point. Every bound is a *conjunct*: all must hold to vote
    out-of-map, so a neutral value (see the defaults) disables its term.

    The defaults are the measured operating point from
    ``outputs/r2_ood_20260906/`` -- the lowest-FRR point that still holds
    ``FAR = 0`` over the cross-product corpus -- except ``vpr_top1_out``,
    which was ratio-rescaled for the BoQ score scale (see below) and awaits
    the full cross-product re-gate. They are inert unless
    ``SFM_EDM_OOD_MODE`` is ``soft`` or ``hard``.
    """

    #: VPR: absolute best cosine at or below which the pool looks foreign.
    #: 0.32 ~= 0.5x the in-map top1 p50 (0.647 over 300 river frames,
    #: outputs/boq_resnet50_20260907/eval_75x4_top10.json) -- the same ratio
    #: as the MegaLoc operating point (0.45 ~= 0.5x its in-map p50 0.909).
    #: Full R2 cross-product re-gate (FAR=0) is still required once an
    #: off-map site exists again (地圖檔/場域/urai/ is absent, so the
    #: off-map corpus pairs are currently unrepeatable).
    vpr_top1_out: float = 0.32
    #: VPR: ``top1 - top2`` at or below which no reference wins the pool.
    #: ``math.inf`` disables the term.
    vpr_margin_out: float = math.inf
    #: VPR: normalised pool entropy at or above which the pool is flat.
    #: ``0.0`` disables the term (entropy is always >= 0).
    vpr_entropy_out: float = 0.95
    #: Matcher: best-reference correspondence count at or below which the
    #: correspondence layer also sees nothing it recognises.
    match_corr_out: int = 200
    #: Matcher: median confidence of the best reference at or below which the
    #: surviving correspondences are noise. ``1.0`` disables the term (EDM
    #: confidences are <= 1).
    match_mconf_out: float = 0.4
    #: SOFT: multiplier on the frame's inlier floor. 1.0 disables tightening.
    soft_inlier_factor: float = 1.5
    #: Softmax temperature for the entropy feature.
    entropy_temperature: float = DEFAULT_ENTROPY_TEMPERATURE

    def validate(self) -> "OODThresholds":
        if not (math.isfinite(self.vpr_top1_out) and -1.0 <= self.vpr_top1_out <= 1.0):
            raise ValueError("vpr_top1_out must be a cosine in [-1, 1]")
        if self.vpr_margin_out < 0.0:
            raise ValueError("vpr_margin_out must be non-negative")
        if not 0.0 <= self.vpr_entropy_out <= 1.0:
            raise ValueError("vpr_entropy_out must be a normalised entropy in [0, 1]")
        if self.match_corr_out < 0:
            raise ValueError("match_corr_out must be non-negative")
        if not 0.0 <= self.match_mconf_out <= 1.0:
            raise ValueError("match_mconf_out must be a confidence in [0, 1]")
        if not (math.isfinite(self.soft_inlier_factor) and self.soft_inlier_factor >= 1.0):
            raise ValueError("soft_inlier_factor must be >= 1.0")
        if not (math.isfinite(self.entropy_temperature) and self.entropy_temperature > 0.0):
            raise ValueError("entropy_temperature must be positive")
        return self

    def as_dict(self) -> dict:
        out = asdict(self)
        # JSON has no infinity; a disabled bound reports as None.
        for key, value in list(out.items()):
            if isinstance(value, float) and not math.isfinite(value):
                out[key] = None
        return out


@dataclass(frozen=True)
class OODVerdict:
    decision: str
    vpr_vote: str
    match_vote: str
    reason: str
    applied: str
    features: dict

    def as_dict(self) -> dict:
        return {
            "decision": self.decision,
            "vpr_vote": self.vpr_vote,
            "match_vote": self.match_vote,
            "reason": self.reason,
            "applied": self.applied,
            **self.features,
        }


def normalise_mode(raw: object) -> str:
    """Map an environment string to a mode; unknown values fall back to hard.

    Empty/unset means the production default (hard since 2026-09-06: 21/21
    exact gate + zero success tax on 6349 in-map frames + 100% off-map
    catch). Explicit ``audit``/``off`` still opt out. Failure mode is loud
    (extra LOST), and new maps must revalidate per the ledger anyway.
    """
    text = str(raw or "").strip().lower()
    if text in ("", "hard"):
        return MODE_HARD
    if text in ("1", "true", "yes", "on"):
        return MODE_AUDIT
    if text in ("0", "false", "no", "off"):
        return MODE_OFF
    return text if text in MODES else MODE_HARD

def mode_from_env(environ=None) -> str:
    env = os.environ if environ is None else environ
    return normalise_mode(env.get("SFM_EDM_OOD_MODE"))


def _env_float(env, name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default
    text = str(raw).strip().lower()
    if text in ("inf", "+inf", "none", "off", "disabled"):
        return math.inf
    try:
        return float(text)
    except ValueError:
        return default


def thresholds_from_env(environ=None) -> OODThresholds:
    """Build thresholds from ``SFM_EDM_OOD_*``; unparsable values keep defaults."""
    env = os.environ if environ is None else environ
    base = OODThresholds()
    corr_raw = env.get("SFM_EDM_OOD_MATCH_CORR")
    try:
        corr = int(float(corr_raw)) if corr_raw not in (None, "") else base.match_corr_out
    except ValueError:
        corr = base.match_corr_out
    return OODThresholds(
        vpr_top1_out=_env_float(env, "SFM_EDM_OOD_VPR_TOP1", base.vpr_top1_out),
        vpr_margin_out=_env_float(env, "SFM_EDM_OOD_VPR_MARGIN", base.vpr_margin_out),
        vpr_entropy_out=_env_float(env, "SFM_EDM_OOD_VPR_ENTROPY", base.vpr_entropy_out),
        match_corr_out=corr,
        match_mconf_out=_env_float(env, "SFM_EDM_OOD_MATCH_MCONF", base.match_mconf_out),
        soft_inlier_factor=_env_float(
            env, "SFM_EDM_OOD_SOFT_INLIER_FACTOR", base.soft_inlier_factor
        ),
        entropy_temperature=_env_float(
            env, "SFM_EDM_OOD_ENTROPY_TEMPERATURE", base.entropy_temperature
        ),
    ).validate()


def _finite_scores(scores) -> list[float]:
    out: list[float] = []
    if scores is None:
        return out
    for value in scores:
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            out.append(score)
    return out


def normalised_entropy(scores: list[float], temperature: float) -> float:
    """Shannon entropy of ``softmax(scores / T)``, divided by ``log(n)``.

    1.0 = every reference equally likely (nothing in the map wins the query),
    0.0 = one reference takes the whole mass.
    """
    count = len(scores)
    if count < 2 or not math.isfinite(temperature) or temperature <= 0.0:
        return float("nan")
    top = max(scores)
    weights = [math.exp((score - top) / temperature) for score in scores]
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0.0:
        return float("nan")
    entropy = 0.0
    for weight in weights:
        probability = weight / total
        if probability > 0.0:
            entropy -= probability * math.log(probability)
    return float(entropy / math.log(count))


def vpr_features(scores, temperature: float = DEFAULT_ENTROPY_TEMPERATURE) -> dict | None:
    """Retrieval-layer features, or None when the pool cannot be judged."""
    values = _finite_scores(scores)
    if len(values) < MIN_VPR_POOL:
        return None
    ordered = sorted(values, reverse=True)
    top1 = ordered[0]
    top2 = ordered[1]
    mean = math.fsum(ordered) / len(ordered)
    variance = math.fsum((value - mean) ** 2 for value in ordered) / len(ordered)
    std = math.sqrt(variance)
    tail = ordered[min(4, len(ordered) - 1)]
    return {
        "vpr_pool": len(ordered),
        "vpr_top1": top1,
        "vpr_top2": top2,
        "vpr_margin": top1 - top2,
        "vpr_gap5": top1 - tail,
        "vpr_mean": mean,
        "vpr_std": std,
        "vpr_z1": ((top1 - mean) / std) if std > 1e-12 else float("nan"),
        "vpr_entropy": normalised_entropy(ordered, temperature),
    }


def matcher_features(per_ref, confidence, by_ref=None) -> dict | None:
    """Correspondence-layer features, or None when nothing was matched.

    ``by_ref`` rows are ``(points2d, points3d, confidence, count)`` as produced
    by ``EDMLocalizer.correspondences_by_ref``; when it is None (temporal-merged
    batch) the features fall back to the concatenated arrays.
    """
    counts = []
    for value in per_ref or ():
        try:
            counts.append(int(value))
        except (TypeError, ValueError):
            continue
    conf = np.asarray(confidence, dtype=np.float64).ravel() if confidence is not None else None
    if conf is not None and conf.size:
        conf = conf[np.isfinite(conf)]
    total = int(sum(counts)) if counts else (0 if conf is None else int(conf.size))
    best = int(max(counts)) if counts else total
    best_conf = conf
    if by_ref:
        best_row = None
        best_len = -1
        for row in by_ref:
            try:
                length = len(row[1])
            except (TypeError, IndexError):
                continue
            if length > best_len:
                best_len = length
                best_row = row
        if best_row is not None and best_len > 0:
            best = max(best, int(best_len))
            candidate = np.asarray(best_row[2], dtype=np.float64).ravel()
            candidate = candidate[np.isfinite(candidate)]
            if candidate.size:
                best_conf = candidate
    features = {
        "match_refs": len(counts),
        "match_corr_total": total,
        "match_corr_best": best,
        "match_mconf_best_p50": float("nan"),
        "match_mconf_best_p90": float("nan"),
        "match_mconf_p50": float("nan"),
    }
    if best_conf is not None and best_conf.size:
        features["match_mconf_best_p50"] = float(np.median(best_conf))
        features["match_mconf_best_p90"] = float(np.quantile(best_conf, 0.9))
    if conf is not None and conf.size:
        features["match_mconf_p50"] = float(np.median(conf))
    if total <= 0 and best <= 0 and not counts:
        return None
    return features


def _vpr_vote(features: dict | None, thresholds: OODThresholds) -> tuple[str, str]:
    if not features:
        return VOTE_UNKNOWN, "vpr_absent"
    top1 = features.get("vpr_top1")
    margin = features.get("vpr_margin")
    entropy = features.get("vpr_entropy")
    if top1 is None or not math.isfinite(float(top1)):
        return VOTE_UNKNOWN, "vpr_nonfinite"
    if float(top1) > thresholds.vpr_top1_out:
        return VOTE_IN, "vpr_top1_high"
    if math.isfinite(thresholds.vpr_margin_out):
        if margin is None or not math.isfinite(float(margin)):
            return VOTE_UNKNOWN, "vpr_margin_nonfinite"
        if float(margin) > thresholds.vpr_margin_out:
            return VOTE_IN, "vpr_margin_wide"
    if thresholds.vpr_entropy_out > 0.0:
        if entropy is None or not math.isfinite(float(entropy)):
            return VOTE_UNKNOWN, "vpr_entropy_nonfinite"
        if float(entropy) < thresholds.vpr_entropy_out:
            return VOTE_IN, "vpr_entropy_peaked"
    return VOTE_OUT, "vpr_flat_and_low"


def _match_vote(features: dict | None, thresholds: OODThresholds) -> tuple[str, str]:
    if not features:
        return VOTE_UNKNOWN, "match_absent"
    best = features.get("match_corr_best")
    if best is None:
        return VOTE_UNKNOWN, "match_nonfinite"
    if int(best) > thresholds.match_corr_out:
        return VOTE_IN, "match_corr_rich"
    if thresholds.match_mconf_out < 1.0:
        mconf = features.get("match_mconf_best_p50")
        if mconf is None or not math.isfinite(float(mconf)):
            return VOTE_UNKNOWN, "match_mconf_nonfinite"
        if float(mconf) > thresholds.match_mconf_out:
            return VOTE_IN, "match_mconf_high"
    return VOTE_OUT, "match_starved"


def evaluate(
    vpr: dict | None,
    matcher: dict | None,
    thresholds: OODThresholds,
    mode: str = MODE_AUDIT,
) -> OODVerdict:
    """Fuse the two channels into one verdict for one frame.

    ``HARD`` needs both channels to vote out-of-map. One channel out (the other
    in) is ``SOFT``. An unknown channel can never produce a veto.
    """
    vpr_vote, vpr_reason = _vpr_vote(vpr, thresholds)
    match_vote, match_reason = _match_vote(matcher, thresholds)
    features: dict = {}
    if vpr:
        features.update(vpr)
    if matcher:
        features.update(matcher)

    if vpr_vote == VOTE_UNKNOWN or match_vote == VOTE_UNKNOWN:
        decision = DECISION_ABSTAIN
        reason = vpr_reason if vpr_vote == VOTE_UNKNOWN else match_reason
    elif vpr_vote == VOTE_OUT and match_vote == VOTE_OUT:
        decision = DECISION_HARD
        reason = f"{vpr_reason}+{match_reason}"
    elif vpr_vote == VOTE_OUT or match_vote == VOTE_OUT:
        decision = DECISION_SOFT
        reason = vpr_reason if vpr_vote == VOTE_OUT else match_reason
    else:
        decision = DECISION_GO
        reason = f"{vpr_reason}+{match_reason}"

    applied = APPLIED_NONE
    if mode == MODE_AUDIT:
        applied = APPLIED_AUDIT
    elif mode == MODE_HARD and decision == DECISION_HARD:
        applied = APPLIED_VETO
    elif mode in (MODE_SOFT, MODE_HARD) and decision in (DECISION_SOFT, DECISION_HARD):
        applied = APPLIED_TIGHTEN
    return OODVerdict(
        decision=decision,
        vpr_vote=vpr_vote,
        match_vote=match_vote,
        reason=reason,
        applied=applied,
        features=features,
    )


def vetoes(verdict: OODVerdict | None, mode: str) -> bool:
    """True only when the active mode turns this verdict into a LOST report."""
    return (
        verdict is not None
        and mode == MODE_HARD
        and verdict.decision == DECISION_HARD
    )


def tightened_inlier_floor(
    verdict: OODVerdict | None,
    mode: str,
    thresholds: OODThresholds,
    min_inliers: int,
) -> int:
    """Raise the frame's inlier floor for a one-channel out-of-map vote."""
    if verdict is None or mode not in (MODE_SOFT, MODE_HARD):
        return int(min_inliers)
    if verdict.decision not in (DECISION_SOFT, DECISION_HARD):
        return int(min_inliers)
    factor = float(thresholds.soft_inlier_factor)
    if not math.isfinite(factor) or factor <= 1.0:
        return int(min_inliers)
    return int(math.ceil(float(min_inliers) * factor))


if __name__ == "__main__":
    print(__doc__)
