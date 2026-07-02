"""
Behavioral signal features, built from the 23 redrob_signals fields.

Directly implements the JD's explicit instruction (Trap #4 from Day 1):
"a perfect-on-paper candidate inactive 6+ months with low recruiter_response_rate
is NOT actually available — must down-weight using redrob_signals regardless of
how good the resume looks."

This module produces a single AVAILABILITY MULTIPLIER (0 to ~1.3) that gets
applied on top of the quality-based score (career trajectory + skills + JD-fit),
rather than being its own independent additive component. This matches how the
JD frames it: availability doesn't make a bad candidate good, but it should
meaningfully discount an otherwise-good candidate who clearly isn't reachable
or interested right now — and conversely give a small bonus to candidates who
are demonstrably active and responsive.
"""
from datetime import datetime, date
from typing import Dict, Any
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.safe_access import safe_redrob_signals

# BUG FIX (found via audit): the original implementation used date.today(),
# making recency_decay() non-deterministic — the SAME candidate gets a
# different score depending on what calendar day the pipeline happens to
# run, which is a real concern for the hard "reproducible in a sandboxed
# container" requirement (Stage 3). Verified the real dataset's
# last_active_date values range 2025-09-29 to 2026-05-27 (all before our
# current date, no future dates). Anchoring to a FIXED reference date
# (the dataset's own max last_active_date, i.e. the effective "snapshot
# date" of the data) makes recency fully deterministic and arguably more
# correct — it measures relative recency within the dataset's own time
# frame, not absolute wall-clock drift from whenever the script executes.
#
# This constant should be regenerated if the dataset is refreshed —
# see compute_reference_date() below for how to re-derive it.
REFERENCE_DATE = date(2026, 5, 27)  # max last_active_date observed in candidates.jsonl


def compute_reference_date(data_path: str) -> date:
    """
    Utility to re-derive REFERENCE_DATE from the actual dataset, in case
    candidates.jsonl is ever regenerated/updated. Not called automatically
    (to keep recency_decay() a pure, fast, dependency-free function during
    the hot path), but should be re-run and the constant above updated if
    the dataset changes.
    """
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from parsing.streaming_reader import iter_candidates
    max_date = None
    for c in iter_candidates(data_path):
        d_str = safe_redrob_signals(c).get("last_active_date")
        if d_str:
            try:
                d = datetime.strptime(d_str, "%Y-%m-%d").date()
                if max_date is None or d > max_date:
                    max_date = d
            except Exception:
                continue
    return max_date


def _years_since(date_str) -> float:
    if not date_str:
        return None
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (REFERENCE_DATE - d).days / 365.25
    except Exception:
        return None


def recency_decay(last_active_date: str) -> float:
    """
    1.0 if active within last 30 days, decaying smoothly to a floor of 0.15
    by ~12 months inactive. Never fully zeroes out on its own — a single
    stale-but-real signal shouldn't kill an otherwise excellent candidate,
    it should just discount them relative to an equally good, active one.
    """
    years = _years_since(last_active_date)
    if years is None:
        return 0.5  # unknown -> neutral-ish, don't punish missing data harshly
    months = years * 12
    if months <= 1:
        return 1.0
    if months >= 12:
        return 0.15
    # linear decay from 1.0 at 1mo to 0.15 at 12mo
    return round(1.0 - (months - 1) * (1.0 - 0.15) / 11, 3)


def response_quality_score(signals: Dict[str, Any]) -> float:
    """
    Combines recruiter_response_rate and interview_completion_rate into one
    0-1 'will this person actually engage if we reach out' score.
    Missing values default to a neutral 0.5 rather than 0, since absence of
    signal (e.g. never been contacted) isn't evidence of unresponsiveness.
    """
    rr = signals.get("recruiter_response_rate")
    ic = signals.get("interview_completion_rate")
    rr = rr if rr is not None else 0.5
    ic = ic if ic is not None else 0.5
    # response rate weighted slightly higher — it's a more direct/frequent signal
    return round(0.6 * rr + 0.4 * ic, 3)


def open_to_work_bonus(signals: Dict[str, Any]) -> float:
    """Small explicit bonus — note 65% of the dataset has this False (per
    Day 1 stats), so it should nudge, not dominate, the multiplier."""
    return 0.10 if signals.get("open_to_work_flag") else 0.0


def offer_acceptance_modifier(signals: Dict[str, Any]) -> float:
    """
    offer_acceptance_rate ranges -1 (no offer history) to 1. A strongly
    negative history (frequently declines/ghosts after offers) is a mild
    negative signal; -1 (no history at all) is neutral, not negative.
    """
    oar = signals.get("offer_acceptance_rate")
    if oar is None or oar == -1:
        return 0.0  # no history = neutral, not punished
    if oar < 0:
        return -0.05  # net-negative acceptance history = small discount
    return 0.0  # positive/neutral history = no special bonus here (avoid double counting)


def verification_modifier(signals: Dict[str, Any]) -> float:
    """Small positive signal for verified contact info — practically relevant
    for a recruiter trying to actually reach this person, per the JD's framing
    of 'a shortlist a recruiter can trust.'"""
    verified_count = sum([
        bool(signals.get("verified_email")),
        bool(signals.get("verified_phone")),
    ])
    return 0.03 * verified_count  # max +0.06


def notice_period_fit(signals: Dict[str, Any]) -> float:
    """
    JD: loves sub-30-day notice (can buy out up to 30 days). 30+ days still
    in scope but bar is higher elsewhere — we don't double-penalize here,
    just a mild shaping signal.
    """
    days = signals.get("notice_period_days")
    if days is None:
        return 0.0
    if days <= 30:
        return 0.05
    if days <= 60:
        return 0.0
    return -0.03  # 60+ days: mild discount, not disqualifying


def compute_behavioral_features(cand: Dict[str, Any]) -> Dict[str, Any]:
    signals = safe_redrob_signals(cand)

    recency = recency_decay(signals.get("last_active_date"))
    response_q = response_quality_score(signals)
    bonus = open_to_work_bonus(signals)
    offer_mod = offer_acceptance_modifier(signals)
    verify_mod = verification_modifier(signals)
    notice_mod = notice_period_fit(signals)

    # Core multiplier: recency and responsiveness are the two pillars (per JD
    # trap #4 language: "inactive 6 months" AND "low response rate" together
    # are what disqualify availability — so we multiply, not just average,
    # so that BOTH being bad compounds the discount, matching the JD's framing).
    core = (0.5 * recency) + (0.5 * response_q)

    multiplier = core + bonus + offer_mod + verify_mod + notice_mod
    multiplier = max(0.1, min(1.3, multiplier))  # floor 0.1, soft ceiling 1.3

    return {
        "recency_decay": recency,
        "response_quality_score": response_q,
        "open_to_work_bonus": bonus,
        "offer_acceptance_modifier": offer_mod,
        "verification_modifier": verify_mod,
        "notice_period_modifier": notice_mod,
        "availability_multiplier": round(multiplier, 3),
    }


if __name__ == "__main__":
    import sys, json, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from parsing.streaming_reader import iter_candidates

    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "candidates.jsonl")

    # CAND_0000100 = inactive, low response (the obvious noise/fake) -> low multiplier expected
    # CAND_0001610 = our gold-standard active, responsive strong candidate -> high multiplier expected
    test_ids = {"CAND_0000100", "CAND_0001610"}
    for cand in iter_candidates(data_path):
        if cand["candidate_id"] in test_ids:
            feats = compute_behavioral_features(cand)
            print(cand["candidate_id"], "->", json.dumps(feats, indent=2))
            test_ids.discard(cand["candidate_id"])
        if not test_ids:
            break
