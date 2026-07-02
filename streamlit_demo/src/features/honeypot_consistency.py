"""
Consistency / honeypot detection features.

Built directly from Day 1 manual reading findings:
- Finding #1 (highest confidence): career_history description text is mismatched
  to its own title/company in fake/noise profiles (e.g. "Graphic Designer" job
  with a content-writing/SEO description). Real candidates' descriptions match
  their stated titles.
- Finding #3: proficiency=="expert" paired with duration_months near 0 is a
  clean, deterministic honeypot signal (claiming mastery with zero time spent).
- Additional: date-math implausibility (career_history duration sum vs
  years_of_experience), and synthetic-filler company name tells.

These features feed into BOTH the honeypot-filtering step (reject before ranking)
and the general trust-discount applied to skills (see skill_corroboration.py).
"""
import re
from datetime import datetime, date
from typing import Dict, Any, List
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.safe_access import safe_career_history, safe_skills, safe_profile
from features.company_tier import FILLER_COMPANIES as _VERIFIED_FILLER_COMPANIES

# AUDIT FIX: this list was originally hand-guessed BEFORE the full-dataset
# company census was done (see company_tier.py), and included names
# ("cyberdyne systems", "oscorp", "soylent corp", "umbrella corporation")
# that don't actually exist anywhere in the real 100K-candidate dataset,
# while using "globex" instead of the dataset's actual "Globex Inc". A diff
# against company_tier.py's FILLER_COMPANIES — which WAS derived from an
# exhaustive scan of all 63 real company names — confirmed the drift.
# Currently harmless (the substring-match fix from an earlier audit pass
# means "globex" still catches "Globex Inc", and the phantom names never
# match anything), but speculative/unverified data should not be the
# source of truth when an empirically verified equivalent exists. Now
# imports the verified list directly instead of maintaining a duplicate.
FILLER_COMPANY_NAMES = _VERIFIED_FILLER_COMPANIES

# Title -> keywords we'd expect to plausibly appear in a matching description.
# Used for the title/description mismatch check. Deliberately loose (OR-match)
# since real descriptions vary a lot; goal is to catch GROSS mismatches like
# "Graphic Designer" title with an accounting description, not to be a strict
# classifier.
TITLE_KEYWORD_HINTS = {
    "graphic designer": ["design", "creative", "brand", "visual", "adobe", "figma", "ui"],
    "content writer": ["writing", "content", "seo", "editorial", "article", "copy"],
    "accountant": ["accounting", "gaap", "ledger", "audit", "tax", "financial reporting", "gl"],
    "marketing manager": ["marketing", "campaign", "brand", "demand gen", "growth"],
    "sales executive": ["sales", "quota", "arr", "pipeline", "prospecting", "closing"],
    "operations manager": ["operations", "process", "logistics", "efficiency", "vendor"],
    "hr manager": ["hr", "recruiting", "hiring", "employee", "talent", "onboarding"],
    "project manager": ["project", "stakeholder", "timeline", "scrum", "delivery", "roadmap"],
    "customer support": ["support", "ticket", "customer", "resolution", "csat"],
    "mechanical engineer": ["mechanical", "cad", "solidworks", "fea", "ansys", "prototype"],
    "civil engineer": ["civil", "construction", "structural", "site", "autocad"],
    "business analyst": ["business analysis", "requirements", "process", "stakeholder", "diagnostics"],
    "mobile developer": ["mobile", "ios", "android", "swift", "kotlin", "react native", "flutter"],
    "devops engineer": ["devops", "ci/cd", "kubernetes", "terraform", "infrastructure", "pipeline"],
    ".net developer": [".net", "c#", "asp.net", "visual studio"],
    "data analyst": ["data", "sql", "dashboard", "analytics", "reporting", "pipeline"],
}


# NOTE (audit finding): a _years_since() helper using date.today() previously
# lived here but was dead code (never called within this file) — removed.
# It carried the same non-determinism bug fixed in behavioral_signals.py
# (see REFERENCE_DATE there). If date-based recency logic is ever needed in
# this module, import from behavioral_signals.py rather than reimplementing,
# to avoid the duplication that caused this exact bug to appear in 3
# separate files in the first place.


def has_expert_zero_duration_skills(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Day 1 finding #3. Cheapest, highest-confidence honeypot rule.
    Returns count + which skill names triggered it.
    """
    flagged = []
    for sk in safe_skills(cand):
        prof = sk.get("proficiency")
        dur = sk.get("duration_months")
        if prof == "expert" and dur is not None and dur <= 2:
            flagged.append(sk.get("name"))
    return {
        "expert_zero_duration_count": len(flagged),
        "expert_zero_duration_skills": flagged,
    }


def title_description_mismatch_score(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Day 1 finding #1. For each career_history entry, check whether the
    description text contains ANY plausible keyword for its own title.
    A "mismatch" = title has known keyword hints, but description contains
    NONE of them. High mismatch ratio = strong honeypot/noise signal.

    Deliberately conservative: titles not in TITLE_KEYWORD_HINTS are skipped
    (we don't penalize for titles we don't have a hint-list for), so this
    only fires on entries we have good confidence about.
    """
    jobs = safe_career_history(cand)
    checked = 0
    mismatches = 0
    mismatched_titles = []

    for job in jobs:
        title = (job.get("title") or "").strip().lower()
        desc = (job.get("description") or "").lower()
        # exact or substring match against known title hints
        hint_keywords = None
        for known_title, kws in TITLE_KEYWORD_HINTS.items():
            if known_title in title:
                hint_keywords = kws
                break
        if hint_keywords is None:
            continue  # no hint list for this title, skip rather than guess
        checked += 1
        if not any(kw in desc for kw in hint_keywords):
            mismatches += 1
            mismatched_titles.append(job.get("title"))

    ratio = (mismatches / checked) if checked > 0 else None
    return {
        "title_desc_checked_count": checked,
        "title_desc_mismatch_count": mismatches,
        "title_desc_mismatch_ratio": ratio,  # None if no titles were checkable
        "title_desc_mismatched_titles": mismatched_titles,
    }


def date_math_consistency_score(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compares sum of career_history duration_months against profile.years_of_experience.
    Flags large mismatches in either direction.

    Threshold calibrated via full-dataset distribution analysis: sampling
    30,000 candidates' ratios showed genuine mismatches cluster either near
    1.0 (no issue) or jump sharply past ~1.9 / below ~0.5 (clear honeypot-
    style impossibility, e.g. ratio 2.07-2.59). The original >1.6/<0.4
    threshold caught our verified-good gold-standard candidate (CAND_0001610,
    ratio 1.69 — a minor, explainable discrepancy, not a fabricated profile)
    as a false positive. Tightened to >1.9/<0.35 to separate "minor data
    noise, ignore" from "structurally impossible, flag" based on where the
    real gap in the observed distribution falls.
    """
    jobs = safe_career_history(cand)
    total_months = sum((j.get("duration_months") or 0) for j in jobs)
    yoe = safe_profile(cand).get("years_of_experience")

    if not yoe or yoe <= 0 or total_months == 0:
        return {"date_math_implied_years": None, "date_math_ratio": None,
                "date_math_suspicious": False}

    implied_years = total_months / 12
    ratio = implied_years / yoe
    suspicious = ratio > 1.9 or ratio < 0.35

    return {
        "date_math_implied_years": round(implied_years, 2),
        "date_math_ratio": round(ratio, 2),
        "date_math_suspicious": suspicious,
    }


def duplicate_description_check(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Day 1 finding: some candidates had IDENTICAL description text reused
    across different career_history entries (e.g. CAND_0000665's PharmEasy
    and Meesho jobs had word-for-word identical descriptions). This is a
    cheap synthetic-data tell, not something a real career history would do.
    """
    jobs = safe_career_history(cand)
    descs = [(j.get("description") or "").strip() for j in jobs if j.get("description")]
    non_empty = [d for d in descs if d]
    has_duplicates = len(non_empty) != len(set(non_empty))
    return {"has_duplicate_job_descriptions": has_duplicates}


def filler_company_name_check(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Day 1 finding: fictional/sitcom company names consistently correlated
    with noise/filler profiles in manual review. Soft signal only — use as
    a minor input, never as a sole disqualifier.

    Uses SUBSTRING match (not exact match) since the dataset's actual company
    field sometimes includes a suffix (e.g. "Globex Inc" rather than bare
    "Globex") — exact-set-membership matching silently missed these.
    """
    jobs = safe_career_history(cand)
    current_company = (safe_profile(cand).get("current_company") or "").lower()

    def is_filler(company: str) -> bool:
        c = (company or "").lower()
        return any(filler in c for filler in FILLER_COMPANY_NAMES)

    filler_hits = sum(1 for j in jobs if is_filler(j.get("company", "")))
    return {
        "filler_company_job_count": filler_hits,
        "current_company_is_filler": is_filler(current_company),
    }


def compute_honeypot_features(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine all consistency checks into one feature dict for this candidate.

    IMPORTANT DESIGN NOTE (found via full-dataset audit): the original single
    'honeypot_risk_score' conflated two genuinely different things:

      1. TRUE HONEYPOT signal: expert_zero_duration and date_math_suspicious.
         Full-dataset scan shows these hit only 0.02% and 0.04% of candidates
         respectively (21 and 45 out of 100,000) — right in line with the
         JD's stated "~80 honeypot candidates" (0.08%) out of the full pool.
         These are the deliberately-impossible-profile signals the JD
         specifically describes.

      2. GENERAL NOISE/FILLER signal: title_desc_mismatch, duplicate
         descriptions, filler company names. These are COMMON (hit roughly
         50%+ of the dataset) because most of the 100K candidates are
         low-effort filler profiles in unrelated fields (Operations Manager,
         Graphic Designer, etc. — see Day 1 stats: ~70%+ of titles are
         non-technical), not because they're deliberately planted traps.
         A single combined score smeared both populations together,
         making the score nearly useless as a clean honeypot filter
         (54% of a 20K sample scored >= 0.5 risk, when true honeypots are
         only ~0.08% of the pool).

    We now expose BOTH:
      - honeypot_strict_score: built ONLY from the two rare, high-precision
        signals. Use this for actual honeypot filtering/rejection.
      - profile_noise_score: built from the common signals. Use this as a
        general quality discount, not a rejection filter — most candidates
        with nonzero noise score are just irrelevant-role candidates, not
        malicious honeypots, and the scoring system already handles
        irrelevance via must_have_corroborated_ratio in jd_fit.py.
    """
    out = {}
    out.update(has_expert_zero_duration_skills(cand))
    out.update(title_description_mismatch_score(cand))
    out.update(date_math_consistency_score(cand))
    out.update(duplicate_description_check(cand))
    out.update(filler_company_name_check(cand))

    # STRICT honeypot score: only the two rare, high-precision signals.
    strict = 0.0
    if out["expert_zero_duration_count"] > 0:
        strict += min(out["expert_zero_duration_count"] * 0.4, 0.7)
    if out["date_math_suspicious"]:
        strict += 0.5
    out["honeypot_strict_score"] = round(min(strict, 1.0), 3)
    out["is_likely_honeypot"] = out["honeypot_strict_score"] >= 0.4

    # General noise/filler score: the common signals, kept separate.
    noise = 0.0
    if out["title_desc_mismatch_ratio"] is not None:
        noise += out["title_desc_mismatch_ratio"] * 0.5
    if out["has_duplicate_job_descriptions"]:
        noise += 0.25
    if out["filler_company_job_count"] > 0:
        noise += min(out["filler_company_job_count"] * 0.1, 0.25)
    out["profile_noise_score"] = round(min(noise, 1.0), 3)

    # Kept for backward compatibility with earlier testing, but scoring.py
    # (Day 3) should use honeypot_strict_score and profile_noise_score
    # separately, not this combined figure.
    out["honeypot_risk_score"] = round(min(out["honeypot_strict_score"] * 0.6 + out["profile_noise_score"] * 0.4, 1.0), 3)
    return out


if __name__ == "__main__":
    import sys, json
    sys.path.insert(0, "..")
    from parsing.streaming_reader import iter_candidates

    # Quick test against known examples from Day 1
    test_ids = {"CAND_0000100", "CAND_0003582", "CAND_0001610", "CAND_0000665"}
    for cand in iter_candidates("../../data/candidates.jsonl"):
        if cand["candidate_id"] in test_ids:
            feats = compute_honeypot_features(cand)
            print(cand["candidate_id"], "->", json.dumps(feats, indent=2))
            test_ids.discard(cand["candidate_id"])
        if not test_ids:
            break
