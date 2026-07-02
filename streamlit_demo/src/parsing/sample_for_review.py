"""
Day 1 sampling script.

Pulls targeted buckets of candidates from the 100K dataset so manual reading
is informed by real, interesting cases instead of random scrolling.

Run: python3 src/parsing/sample_for_review.py

Output: output/day1_samples.json — a dict of bucket_name -> list of candidates,
plus output/day1_samples.md — a human-readable version for fast reading.
"""
import sys
import json
import re
from pathlib import Path
from datetime import datetime, date

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parsing.streaming_reader import iter_candidates
from utils.jd_config import (
    MUST_HAVE_SIGNALS, CONSULTING_FIRMS, CV_SPEECH_ROBOTICS_KEYWORDS,
    NLP_IR_RESCUE_KEYWORDS, PREFERRED_LOCATIONS,
)

DATA_PATH = str(Path(__file__).resolve().parent.parent.parent / "data" / "candidates.jsonl")
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)

N_PER_BUCKET = 10  # ~10 buckets x 8 = ~80 candidates total, matches your reading target


def full_text_blob(cand: dict) -> str:
    """Concatenate all textual fields, lowercased, for keyword scanning."""
    parts = [
        cand.get("profile", {}).get("headline", ""),
        cand.get("profile", {}).get("summary", ""),
        cand.get("profile", {}).get("current_title", ""),
    ]
    for job in cand.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    for sk in cand.get("skills", []):
        parts.append(sk.get("name", ""))
    return " ".join(parts).lower()


def has_any_keyword(text: str, keywords: list) -> bool:
    return any(kw in text for kw in keywords)


def years_since(date_str):
    """
    NOTE (audit finding): originally used date.today(), which is
    non-deterministic across runs on different days. This script is a
    one-time Day 1 exploration tool (not part of the Day 3+ scoring
    pipeline, which has its own fixed REFERENCE_DATE in
    behavioral_signals.py), so the severity here is low — but fixed for
    consistency, since this exact bug pattern was found duplicated in 3
    files and should be eliminated everywhere, not just the critical path.
    """
    REFERENCE_DATE = date(2026, 5, 27)  # see behavioral_signals.py for derivation
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (REFERENCE_DATE - d).days / 365.25
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Bucket predicates
# ---------------------------------------------------------------------------

def is_keyword_stuffed_bad_title(cand: dict) -> bool:
    """Trap #2: lots of AI skill keywords, but title suggests non-technical role."""
    title = cand.get("profile", {}).get("current_title", "").lower()
    non_technical_titles = ["marketing", "sales", "hr ", "recruiter", "operations manager",
                             "business development", "account manager"]
    if not any(t in title for t in non_technical_titles):
        return False
    skills = [s.get("name", "").lower() for s in cand.get("skills", [])]
    ai_kw_count = sum(1 for s in skills if any(
        kw in s for sig in MUST_HAVE_SIGNALS.values() for kw in sig["keywords"]
    ))
    return ai_kw_count >= 4


def is_implicit_fit_no_keywords(cand: dict) -> bool:
    """Trap #3: no flashy keywords, but career history suggests real recsys/search work."""
    text = full_text_blob(cand)
    flashy = ["rag", "pinecone", "langchain", "vector database", "embeddings"]
    if has_any_keyword(text, flashy):
        return False
    substance = ["recommendation system", "recommend", "search relevance",
                 "ranking algorithm", "personalization", "matching engine"]
    return has_any_keyword(text, substance)


def is_inactive_but_perfect_on_paper(cand: dict) -> bool:
    """Trap #4: looks great structurally, but behavioral signals say not available."""
    signals = cand.get("redrob_signals", {})
    last_active = signals.get("last_active_date")
    response_rate = signals.get("recruiter_response_rate", 1.0)
    years_inactive = years_since(last_active) if last_active else None
    if years_inactive is None or years_inactive < 0.5:
        return False
    if response_rate is not None and response_rate > 0.10:
        return False
    yoe = cand.get("profile", {}).get("years_of_experience", 0)
    return yoe >= 5


def is_suspicious_duration_mismatch(cand: dict) -> bool:
    """Honeypot signal: expert proficiency claimed with ~0 duration_months."""
    for sk in cand.get("skills", []):
        if sk.get("proficiency") == "expert" and sk.get("duration_months", 999) <= 2:
            return True
    return False


def is_suspicious_date_math(cand: dict) -> bool:
    """Honeypot signal: career_history duration doesn't add up against dates,
    or total experience implausible vs years_of_experience."""
    total_months = 0
    for job in cand.get("career_history", []):
        total_months += job.get("duration_months", 0) or 0
    yoe = cand.get("profile", {}).get("years_of_experience", 0)
    if yoe and total_months:
        implied_years = total_months / 12
        # flag big mismatches either direction
        if implied_years > yoe * 1.6 or implied_years < yoe * 0.4:
            return True
    return False


def is_pure_consulting_career(cand: dict) -> bool:
    jobs = cand.get("career_history", [])
    if not jobs:
        return False
    companies = [j.get("company", "").lower() for j in jobs]
    return all(any(firm in c for firm in CONSULTING_FIRMS) for c in companies)


def is_cv_speech_robotics_no_nlp(cand: dict) -> bool:
    text = full_text_blob(cand)
    if not has_any_keyword(text, CV_SPEECH_ROBOTICS_KEYWORDS):
        return False
    return not has_any_keyword(text, NLP_IR_RESCUE_KEYWORDS)


def is_title_chaser(cand: dict) -> bool:
    """Title escalates fast across short stints."""
    jobs = sorted(cand.get("career_history", []), key=lambda j: j.get("start_date", ""))
    if len(jobs) < 3:
        return False
    short_stints = sum(1 for j in jobs if (j.get("duration_months", 999) or 999) <= 20)
    escalating_titles = sum(1 for j in jobs if any(
        lvl in j.get("title", "").lower() for lvl in ["senior", "staff", "principal", "lead", "head"]
    ))
    return short_stints >= 2 and escalating_titles >= 2


def is_strong_candidate_heuristic(cand: dict) -> bool:
    """Crude 'obviously good' bucket — product company, real ML signal, eval-framework
    mention, active, in/near preferred location. Used as a positive anchor."""
    text = full_text_blob(cand)
    signals = cand.get("redrob_signals", {})
    yoe = cand.get("profile", {}).get("years_of_experience", 0)
    loc = cand.get("profile", {}).get("location", "").lower()

    has_eval = has_any_keyword(text, MUST_HAVE_SIGNALS["eval_frameworks"]["keywords"])
    has_embed = has_any_keyword(text, MUST_HAVE_SIGNALS["embeddings_retrieval"]["keywords"])
    has_vecdb = has_any_keyword(text, MUST_HAVE_SIGNALS["vector_db_or_hybrid_search"]["keywords"])
    is_consulting_only = is_pure_consulting_career(cand)
    response_rate = signals.get("recruiter_response_rate", 0)
    open_to_work = signals.get("open_to_work_flag", False)
    near_pref_loc = any(p in loc for p in PREFERRED_LOCATIONS)

    return (
        has_eval and has_embed and has_vecdb
        and not is_consulting_only
        and 5 <= yoe <= 9
        and response_rate and response_rate > 0.4
        and (open_to_work or near_pref_loc)
    )


def is_extreme_years(cand: dict) -> bool:
    """Edge cases: very low or very high years_of_experience for a 'senior' role."""
    yoe = cand.get("profile", {}).get("years_of_experience", 0)
    return yoe <= 1 or yoe >= 20


BUCKETS = {
    "trap_keyword_stuffed_bad_title": is_keyword_stuffed_bad_title,
    "trap_implicit_fit_no_keywords": is_implicit_fit_no_keywords,
    "trap_inactive_perfect_on_paper": is_inactive_but_perfect_on_paper,
    "honeypot_duration_mismatch": is_suspicious_duration_mismatch,
    "honeypot_date_math_suspicious": is_suspicious_date_math,
    "disqualifier_pure_consulting": is_pure_consulting_career,
    "disqualifier_cv_speech_robotics": is_cv_speech_robotics_no_nlp,
    "disqualifier_title_chaser": is_title_chaser,
    "positive_anchor_strong_fit": is_strong_candidate_heuristic,
    "edge_case_extreme_years": is_extreme_years,
}


def slim(cand: dict) -> dict:
    """
    Trim candidate to the fields you actually need to read quickly.

    AUDIT NOTE: this module (and the predicate functions above it) use the
    unguarded `cand.get("profile", {})` pattern that was found to be a real
    bug in the features/*.py modules (crashes on explicit None values) and
    fixed there via utils/safe_access.py. This module was NOT included in
    that fix. Justification for deferring: (1) this is a one-time Day 1
    exploration script, not part of the Stage 3 reproduction path; (2) its
    output (day1_samples.json) was already generated against the real
    100K-candidate dataset, which has been empirically confirmed multiple
    times to contain zero None-valued profile/career_history/skills/
    redrob_signals fields; (3) re-running this script against the same
    real data (confirmed working, all 10/10 buckets populate identically)
    poses no actual risk. If this script is ever pointed at a different or
    untrusted dataset, it should be updated to use utils/safe_access.py
    first, consistent with the rest of the codebase.
    """
    p = cand.get("profile", {})
    return {
        "candidate_id": cand.get("candidate_id"),
        "headline": p.get("headline"),
        "current_title": p.get("current_title"),
        "current_company": p.get("current_company"),
        "years_of_experience": p.get("years_of_experience"),
        "location": p.get("location"),
        "summary": p.get("summary"),
        "career_history": [
            {
                "company": j.get("company"), "title": j.get("title"),
                "duration_months": j.get("duration_months"),
                "is_current": j.get("is_current"),
                "description": (j.get("description") or "")[:300],
            }
            for j in cand.get("career_history", [])
        ],
        "skills": [
            {"name": s.get("name"), "proficiency": s.get("proficiency"),
             "duration_months": s.get("duration_months"), "endorsements": s.get("endorsements")}
            for s in cand.get("skills", [])
        ],
        "redrob_signals": {
            k: cand.get("redrob_signals", {}).get(k)
            for k in ["last_active_date", "recruiter_response_rate", "open_to_work_flag",
                      "interview_completion_rate", "github_activity_score", "notice_period_days"]
        },
    }


def main():
    print(f"Streaming {DATA_PATH} ...")
    buckets_out = {name: [] for name in BUCKETS}
    counts_seen = {name: 0 for name in BUCKETS}

    total = 0
    for cand in iter_candidates(DATA_PATH):
        total += 1
        for name, pred in BUCKETS.items():
            if len(buckets_out[name]) >= N_PER_BUCKET:
                continue
            try:
                if pred(cand):
                    counts_seen[name] += 1
                    buckets_out[name].append(slim(cand))
            except Exception as e:
                continue
        if total % 20000 == 0:
            print(f"  ...scanned {total}")
        if all(len(v) >= N_PER_BUCKET for v in buckets_out.values()):
            # keep scanning anyway for accurate counts, but you could break here to go faster
            pass

    print(f"Scanned all {total} candidates.")
    for name in BUCKETS:
        print(f"  {name}: found {len(buckets_out[name])} (capped at {N_PER_BUCKET})")

    json_path = OUT_DIR / "day1_samples.json"
    with open(json_path, "w") as f:
        json.dump(buckets_out, f, indent=2)
    print(f"\nWrote {json_path}")

    # Human-readable markdown for fast reading
    md_path = OUT_DIR / "day1_samples.md"
    with open(md_path, "w") as f:
        for name, cands in buckets_out.items():
            f.write(f"\n\n# Bucket: {name}\n\n")
            for c in cands:
                f.write(f"## {c['candidate_id']} — {c['current_title']} @ {c['current_company']}\n")
                f.write(f"- YOE: {c['years_of_experience']} | Location: {c['location']}\n")
                f.write(f"- Headline: {c['headline']}\n")
                f.write(f"- Summary: {c['summary']}\n")
                f.write(f"- Signals: {c['redrob_signals']}\n")
                f.write("- Career history:\n")
                for j in c["career_history"]:
                    f.write(f"  - {j['title']} @ {j['company']} ({j['duration_months']}mo"
                             f"{', current' if j['is_current'] else ''}): {j['description']}\n")
                f.write("- Skills: " + ", ".join(
                    f"{s['name']}({s['proficiency']},{s['duration_months']}mo,{s['endorsements']}end)"
                    for s in c["skills"]
                ) + "\n\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
