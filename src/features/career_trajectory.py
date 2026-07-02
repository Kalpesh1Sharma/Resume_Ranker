"""
Career trajectory features.

Built from Day 1 bugs found in the original sampling heuristics:
- is_title_chaser (old) fired on "senior-sounding words present" rather than
  actual title ESCALATION across jobs. CAND_0000665 had flat titles
  (Senior -> Senior -> Backend) and got wrongly flagged.
- is_pure_consulting_career (old) fired on a SINGLE 13-month stint at TCS
  for a fresh-grad noise profile, not the JD's actual concern (a full career
  spent entirely in consulting with zero product-company exposure).
- Day 1 also showed product-company breadth (Dream11, Flipkart, Zoho,
  Swiggy, Zomato, Uber) is a strong positive signal when present across
  MULTIPLE jobs, consistent with the JD's "4-5 yrs applied ML at product
  companies" ideal profile.
"""
import re
from typing import Dict, Any, List
from datetime import datetime
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.jd_config import CONSULTING_FIRMS
from utils.safe_access import safe_career_history, safe_profile
from features.company_tier import FILLER_COMPANIES

SENIORITY_LEVELS = {
    "intern": 0, "junior": 1, "associate": 1,
    "": 2,  # no level word = treat as mid/baseline
    "senior": 3, "sr": 3, "sr.": 3,
    "staff": 4, "lead": 4,
    "principal": 5, "head": 5, "director": 5,
    "vp": 6, "vice president": 6, "chief": 7,
}

# checked longest-prefix-first so "staff" doesn't get matched as part of
# a longer title accidentally
SENIORITY_ORDER = ["chief", "vp", "vice president", "principal", "director", "head",
                    "staff", "lead", "senior", "sr.", "sr", "associate", "junior", "intern"]


def _seniority_level(title: str) -> int:
    t = (title or "").lower()
    for word in SENIORITY_ORDER:
        if word in t:
            return SENIORITY_LEVELS.get(word, 2)
    return 2  # no seniority word found, treat as baseline/mid


def _sorted_jobs(cand: Dict[str, Any]) -> List[dict]:
    jobs = safe_career_history(cand)
    return sorted(jobs, key=lambda j: j.get("start_date") or "")


def title_escalation_features(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED VERSION of the title-chaser check. Looks at actual seniority
    progression across jobs in chronological order, not just presence of
    senior-sounding words.

    Flags as "title_chaser" only when:
      - seniority level INCREASES across 2+ consecutive job changes, AND
      - each of those jobs lasted <= 18 months (short stints)
    Threshold matches the JD's literal language: "switching companies every
    1.5 years" (line 46 of job_description.docx) = 18 months, not a rounder
    but looser number like 20 or 24.
    A flat or single-step progression (e.g. mid -> senior once, staying
    3+ years each) is normal healthy growth and should NOT be flagged.
    """
    jobs = _sorted_jobs(cand)
    if len(jobs) < 3:
        return {"title_chaser": False, "seniority_progression": [], "escalation_jump_count": 0}

    levels = [_seniority_level(j.get("title", "")) for j in jobs]
    durations = [j.get("duration_months") or 999 for j in jobs]

    escalation_jumps = 0
    for i in range(1, len(levels)):
        increased = levels[i] > levels[i - 1]
        prev_was_short = durations[i - 1] <= 18
        if increased and prev_was_short:
            escalation_jumps += 1

    is_chaser = escalation_jumps >= 2

    return {
        "title_chaser": is_chaser,
        "seniority_progression": levels,
        "escalation_jump_count": escalation_jumps,
    }


def product_vs_consulting_features(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED VERSION. The JD's actual concern is a career spent ENTIRELY in
    consulting/services firms with zero product-company exposure. A single
    short stint, or a candidate who did consulting earlier but has product
    experience too, should NOT be penalized — the JD explicitly says prior
    product experience makes a CURRENT consulting role acceptable.

    Threshold: only flag as "pure_consulting_career" if ALL jobs are at
    consulting firms AND total experience >= 2 years (long enough that
    "this is their whole career so far" is a meaningful claim, not just
    a first job out of college).

    BUG FIX (found during scoring-engine integration testing, traced
    through to a misleading reasoning-text output): has_product_company_
    experience was computed as simply consulting_ratio < 1.0, which counts
    ANY non-consulting company — including the dataset's known fictional
    filler companies (Dunder Mifflin, Stark Industries, Pied Piper, etc.,
    see company_tier.py's FILLER_COMPANIES, derived from a full company
    census) — as "product experience." Measured impact: 90,255 candidates
    showed has_product_company_experience=True; of those, 74,737 (82.8%)
    had ALL of their non-consulting jobs be filler companies, not real
    product companies. Verified this had ZERO impact on the actual top-100
    ranking (the dominant must_have_corroborated term already correctly
    suppressed these candidates), but it produced misleading reasoning
    text and was a latent risk for any future weight retuning. Fixed by
    excluding filler companies from counting as product experience.
    """
    jobs = _sorted_jobs(cand)
    if not jobs:
        return {"pure_consulting_career": False, "consulting_job_ratio": 0.0,
                "has_product_company_experience": False}

    def is_consulting(company: str) -> bool:
        c = (company or "").lower()
        return any(firm in c for firm in CONSULTING_FIRMS)

    def is_filler(company: str) -> bool:
        c = (company or "").lower()
        return any(fc in c for fc in FILLER_COMPANIES)

    consulting_jobs = [j for j in jobs if is_consulting(j.get("company", ""))]
    consulting_ratio = len(consulting_jobs) / len(jobs)
    total_months = sum((j.get("duration_months") or 0) for j in jobs)
    total_years = total_months / 12

    all_consulting = consulting_ratio == 1.0
    pure_consulting_career = all_consulting and total_years >= 2.0

    # Product experience now requires a job that is NEITHER consulting NOR
    # a known filler/fake company.
    has_real_product_job = any(
        not is_consulting(j.get("company", "")) and not is_filler(j.get("company", ""))
        for j in jobs
    )

    return {
        "pure_consulting_career": pure_consulting_career,
        "consulting_job_ratio": round(consulting_ratio, 2),
        "has_product_company_experience": has_real_product_job,
    }


def tenure_stability_features(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Average and minimum tenure across jobs (excluding current/ongoing role,
    since that one is naturally shorter and shouldn't be judged the same way).
    Very short average tenure (<12mo) across a multi-job history is a mild
    instability signal — distinct from the title_chaser pattern, which
    specifically requires seniority escalation, not just job-hopping.
    """
    jobs = _sorted_jobs(cand)
    past_jobs = [j for j in jobs if not j.get("is_current", False)]
    durations = [j.get("duration_months") for j in past_jobs if j.get("duration_months")]

    if not durations:
        return {"avg_past_tenure_months": None, "min_past_tenure_months": None,
                "job_count": len(jobs)}

    return {
        "avg_past_tenure_months": round(sum(durations) / len(durations), 1),
        "min_past_tenure_months": min(durations),
        "job_count": len(jobs),
    }


def compute_career_trajectory_features(cand: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    out.update(title_escalation_features(cand))
    out.update(product_vs_consulting_features(cand))
    out.update(tenure_stability_features(cand))
    return out


if __name__ == "__main__":
    import sys, json, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from parsing.streaming_reader import iter_candidates

    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "candidates.jsonl")

    # CAND_0000665 = false positive in old title-chaser bucket (should be False now)
    # CAND_0000003 = false positive in old consulting bucket (single short stint, should be False now)
    # CAND_0001610 = our gold-standard strong candidate (product companies, should show has_product_company_experience True)
    test_ids = {"CAND_0000665", "CAND_0000003", "CAND_0001610"}
    for cand in iter_candidates(data_path):
        if cand["candidate_id"] in test_ids:
            feats = compute_career_trajectory_features(cand)
            print(cand["candidate_id"], "->", json.dumps(feats, indent=2))
            test_ids.discard(cand["candidate_id"])
        if not test_ids:
            break
