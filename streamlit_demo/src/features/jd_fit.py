"""
JD-fit features — ties together must-haves, nice-to-haves, disqualifiers, and
location/logistics from jd_config.py into a single set of fit scores.

Built corroboration-aware from the start (Day 1's biggest lesson): a must-have
or disqualifier keyword match is checked against BOTH skills AND career_history
text, and skill-only matches are discounted via skill_corroboration's findings.
This directly targets the JD's stated Trap #1 (keyword-stuffing) and Trap #3
(implicit fit without flashy keywords) simultaneously, since we score evidence
from career_history text on equal footing with the skills list rather than
trusting skills list keyword presence alone.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
from utils.jd_config import (
    MUST_HAVE_SIGNALS, NICE_TO_HAVE_SIGNALS, DISQUALIFIERS,
    CV_SPEECH_ROBOTICS_KEYWORDS, NLP_IR_RESCUE_KEYWORDS,
    PREFERRED_LOCATIONS, ACCEPTABLE_LOCATIONS,
    IDEAL_PROFILE,
)
from features.skill_corroboration import _career_history_blob, _normalize
from features.career_trajectory import product_vs_consulting_features
from utils.safe_access import safe_profile, safe_skills, safe_redrob_signals

# Phrases found via full-dataset scan that indicate ASPIRATIONAL, not actual,
# experience (e.g. "I've been taking online courses on RAG and vector
# databases" — a templated summary line that appears in 5,517/100,000
# candidates and falsely inflated must_have_corroborated for
# vector_db_or_hybrid_search from ~0.2% to ~5.6% before this fix, since the
# raw keyword "vector database" matched with zero awareness that the
# surrounding sentence explicitly means the opposite of real experience).
ASPIRATIONAL_NEGATION_PHRASES = [
    "taking online courses", "i ve been excited about", "interested in transitioning",
    "exploring how", "still building", "hoping to", "looking to break into",
    "aspiring to", "want to learn", "self taught", "no formal experience",
]

NEGATION_WINDOW_CHARS = 150  # if a must-have keyword falls within this many
                               # characters of a negation phrase, treat the
                               # match as unreliable and discard it


def _has_keyword_outside_negation_context(text: str, keywords) -> bool:
    """
    Like _text_has_any, but discards keyword matches that fall within
    NEGATION_WINDOW_CHARS of an aspirational/negation phrase. This is a
    blunt proximity heuristic, not real NLP — but it directly fixes the
    confirmed false-positive template found in the full-dataset audit.
    """
    negation_spans = []
    for phrase in ASPIRATIONAL_NEGATION_PHRASES:
        start = 0
        while True:
            idx = text.find(phrase, start)
            if idx == -1:
                break
            negation_spans.append((idx, idx + len(phrase)))
            start = idx + 1

    for kw in keywords:
        start = 0
        while True:
            idx = text.find(kw, start)
            if idx == -1:
                break
            kw_end = idx + len(kw)
            near_negation = any(
                (idx < neg_end + NEGATION_WINDOW_CHARS and kw_end > neg_start - NEGATION_WINDOW_CHARS)
                for neg_start, neg_end in negation_spans
            )
            if not near_negation:
                return True
            start = idx + 1
    return False


def _text_has_any(text: str, keywords) -> bool:
    return any(kw in text for kw in keywords)


def _full_blob(cand: Dict[str, Any]) -> str:
    """Career history + summary + skills, all normalized — used for must-have
    detection since must-haves should count whether evidence comes from
    skills OR career_history (per Day 1 Trap #3: implicit fit without
    flashy keywords still counts)."""
    career_blob = _career_history_blob(cand)  # already includes summary
    skill_text = " ".join(s.get("name", "") for s in safe_skills(cand))
    return career_blob + " " + _normalize(skill_text)


def must_have_coverage(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    For each must-have signal category, check if there's evidence ANYWHERE
    (career history OR skills). Also separately track whether the evidence
    is corroborated (appears in career_history specifically, not just skills
    list) — this corroborated count is the stronger, more trustworthy signal.
    """
    blob = _full_blob(cand)
    career_only_blob = _career_history_blob(cand)

    covered = {}
    corroborated = {}
    for key, sig in MUST_HAVE_SIGNALS.items():
        kws = sig["keywords"]
        covered[key] = _text_has_any(blob, kws)
        # Use negation-aware matching here specifically: career_history text
        # (and the summary, which is folded into career_only_blob) is where
        # the aspirational-language false positive was found. Plain skills
        # list entries don't have this problem since they're not full
        # sentences with negating context.
        corroborated[key] = _has_keyword_outside_negation_context(career_only_blob, kws)

    covered_count = sum(covered.values())
    corroborated_count = sum(corroborated.values())
    total = len(MUST_HAVE_SIGNALS)

    return {
        "must_have_covered": covered,
        "must_have_corroborated": corroborated,
        "must_have_covered_count": covered_count,
        "must_have_corroborated_count": corroborated_count,
        "must_have_coverage_ratio": round(covered_count / total, 3),
        "must_have_corroborated_ratio": round(corroborated_count / total, 3),
    }


def nice_to_have_coverage(cand: Dict[str, Any]) -> Dict[str, Any]:
    blob = _full_blob(cand)
    covered = {k: _text_has_any(blob, kws) for k, kws in NICE_TO_HAVE_SIGNALS.items()}
    return {
        "nice_to_have_covered": covered,
        "nice_to_have_count": sum(covered.values()),
    }


def disqualifier_flags(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Checks the structurally-detectable disqualifiers from jd_config.py.
    Note: framework_enthusiast and closed_source_no_validation are flagged
    as "weak proxy / not reliably detectable from structured data alone" in
    jd_config.py itself — we compute a best-effort proxy but flag low confidence.
    """
    career_blob = _career_history_blob(cand)
    consulting_feats = product_vs_consulting_features(cand)
    yoe = safe_profile(cand).get("years_of_experience", 0)

    # pure_research_no_production: crude proxy — no product/company industry
    # signal at all and explicitly research-flavored language, no "shipped"/
    # "deployed"/"production" language anywhere.
    research_words = ["research lab", "phd", "academic", "publication", "thesis"]
    production_words = ["production", "shipped", "deployed", "users", "scale", "launch"]
    pure_research = (
        _text_has_any(career_blob, research_words)
        and not _text_has_any(career_blob, production_words)
    )

    # langchain_only_recent: proxy — has langchain/openai api keywords, total
    # YOE is low-ish, and no pre-LLM-era ML keywords (sklearn, xgboost, etc.)
    langchain_kw = ["langchain", "openai api", "gpt-4", "gpt-3", "prompt engineering"]
    pre_llm_kw = ["scikit-learn", "xgboost", "lightgbm", "tensorflow", "pytorch",
                  "feature engineering", "statistical modeling"]
    langchain_only = (
        _text_has_any(career_blob, langchain_kw)
        and not _text_has_any(career_blob, pre_llm_kw)
        and yoe < 3
    )

    # cv_speech_robotics_no_nlp: reuse the same logic validated in Day 1
    cv_speech_robotics = (
        _text_has_any(career_blob, CV_SPEECH_ROBOTICS_KEYWORDS)
        and not _text_has_any(career_blob, NLP_IR_RESCUE_KEYWORDS)
    )

    return {
        "disq_pure_research_no_production": pure_research,
        "disq_langchain_only_recent": langchain_only,
        "disq_pure_consulting_career": consulting_feats["pure_consulting_career"],
        "disq_cv_speech_robotics_no_nlp": cv_speech_robotics,
        # title_chaser and closed_source proxies live in career_trajectory.py
        # and aren't duplicated here to avoid drift between modules.
    }


def location_logistics_fit(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scores location fit: preferred (Pune/Noida) > acceptable (other NCR/
    Hyderabad/Mumbai) > other India (needs willing_to_relocate) > outside
    India (needs willing_to_relocate, treated more cautiously per JD's "no
    visa sponsorship outside India" note).
    """
    loc = (safe_profile(cand).get("location") or "").lower()
    country = (safe_profile(cand).get("country") or "").lower()
    willing = safe_redrob_signals(cand).get("willing_to_relocate", False)

    if any(p in loc for p in PREFERRED_LOCATIONS):
        score = 1.0
        tier = "preferred"
    elif any(a in loc for a in ACCEPTABLE_LOCATIONS):
        score = 0.8
        tier = "acceptable"
    elif "india" in country or country == "":
        score = 0.6 if willing else 0.35
        tier = "other_india"
    else:
        score = 0.4 if willing else 0.05
        tier = "outside_india"

    return {"location_tier": tier, "location_fit_score": score}


def experience_band_fit(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Soft scoring against the JD's stated ideal band (6-8 yrs core, 5-9 yrs
    acceptable) — NOT a hard cutoff, since the JD itself says the range is
    flexible for the right candidate.

    BUG FIX (found via audit): the original implementation had flat plateaus
    (1.0 for 6-8, 0.85 for 5-9) with hard jumps between them — e.g. YOE=5.9
    scored 0.85 while YOE=6.0 scored 1.0, an artificial cliff at a boundary
    the JD explicitly calls "a range, not a requirement." Rewritten as a
    genuinely continuous piecewise-linear function: ramps smoothly up to
    1.0 across the soft_lo->lo gap, holds 1.0 across the core band, ramps
    smoothly back down across the hi->soft_hi gap, then falls off at
    different rates beyond the soft band (steeper below, since junior
    candidates rarely fit a senior IC role; gentler above, since very
    senior people can still be relevant per the JD's flexible framing).
    """
    yoe = safe_profile(cand).get("years_of_experience", 0) or 0
    lo, hi = IDEAL_PROFILE["years_experience_range"]            # 6, 8
    soft_lo, soft_hi = IDEAL_PROFILE["years_experience_soft_range"]  # 5, 9

    if lo <= yoe <= hi:
        score = 1.0
    elif soft_lo <= yoe < lo:
        # ramp UP from 0.85 (at soft_lo) to 1.0 (at lo) — continuous with
        # both the plateau above and the falloff below
        frac = (yoe - soft_lo) / (lo - soft_lo)
        score = 0.85 + frac * 0.15
    elif hi < yoe <= soft_hi:
        # ramp DOWN from 1.0 (at hi) to 0.85 (at soft_hi)
        frac = (yoe - hi) / (soft_hi - hi)
        score = 1.0 - frac * 0.15
    elif yoe < soft_lo:
        gap = soft_lo - yoe
        score = max(0.1, 0.85 - gap * 0.25)
    else:  # yoe > soft_hi
        gap = yoe - soft_hi
        score = max(0.3, 0.85 - gap * 0.10)

    return {"experience_band_score": round(score, 3)}


def compute_jd_fit_features(cand: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    out.update(must_have_coverage(cand))
    out.update(nice_to_have_coverage(cand))
    out.update(disqualifier_flags(cand))
    out.update(location_logistics_fit(cand))
    out.update(experience_band_fit(cand))
    return out


if __name__ == "__main__":
    import json
    from parsing.streaming_reader import iter_candidates

    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "candidates.jsonl")

    # CAND_0001610 = gold-standard, should show high must_have_corroborated_count
    # CAND_0000100 = obvious fake, should show low coverage + likely some disqualifier
    # CAND_0000082 = scattered-skills weak candidate, should show covered but NOT corroborated
    test_ids = {"CAND_0001610", "CAND_0000100", "CAND_0000082"}
    for cand in iter_candidates(data_path):
        if cand["candidate_id"] in test_ids:
            feats = compute_jd_fit_features(cand)
            print(cand["candidate_id"], "->", json.dumps(feats, indent=2))
            test_ids.discard(cand["candidate_id"])
        if not test_ids:
            break
