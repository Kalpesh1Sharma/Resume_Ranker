"""
Core scoring engine — combines all per-candidate features into one final
composite score and produces the ranked top-100 output.

WEIGHT DESIGN RATIONALE (every weight below is traceable to either the JD
text or an empirical audit finding — this is the file to defend in Stage 5):

1. MUST-HAVE CORROBORATION is the dominant term (not raw coverage).
   Evidence: full-dataset audit showed must_have_coverage (skills+text) for
   embeddings_retrieval was 10.38%, but must_have_corroborated (career
   history only, negation-aware) was 0.15% — a 69x gap. The JD's own stated
   trap #1 ("keyword-stuffing is not the answer") is empirically confirmed
   at full scale, not just asserted from the text. Corroboration is what
   the scorer must reward; raw coverage is what it must NOT reward heavily.

2. HONEYPOT/NOISE PENALTIES ARE MULTIPLICATIVE GATES, not additive terms.
   Evidence: honeypot_strict_score correctly isolates ~0.063% of the
   dataset (63/100,000), matching the JD's stated ~80 honeypots almost
   exactly. A candidate flagged is_likely_honeypot should be pushed to the
   bottom regardless of how good their other features look — additive
   scoring would let a few strong-looking features partially cancel out a
   structurally-impossible profile, which is wrong. Same logic, gentler
   slope, for profile_noise_score (common, not rare — ~50%+ of candidates
   have nonzero noise, so this is a discount, not a gate).

3. AVAILABILITY (behavioral_signals) IS ALSO MULTIPLICATIVE, not additive.
   Directly per the JD's trap #4 language: "a perfect-on-paper candidate
   inactive 6 months... is NOT actually available... must down-weight
   regardless of how good the resume looks." A multiplicative term is the
   only structure that can't be fully offset by an otherwise-perfect score
   — exactly matching "regardless of."

4. THE GAMING-VECTOR FIX: must_have_corroborated_ratio alone was shown
   (prior audit session) to be exploitable by stuffing keywords into the
   description text itself rather than the skills list — defeating
   corroboration while still reading as "backed by career history." Zero
   real candidates in the 100K dataset exploit this, but the fix costs
   nothing and closes a real theoretical gap: title_desc_mismatch_ratio is
   now combined multiplicatively with must_have_corroborated_ratio, so a
   candidate can't get credit for corroboration if their description text
   doesn't even plausibly match their own title.

5. SKILL CORROBORATION RATIO is NOT used as an absolute 0-1 quality score.
   Evidence: calibration check showed even the verified gold-standard
   candidate (CAND_0001610) only reaches 0.556, since real job descriptions
   are terse and don't name every tool. Used as one input among several,
   not a dominant term, and never compared to an absolute "should be near
   1.0" threshold.

6. EXPERIENCE BAND, COMPANY TIER, CAREER TRAJECTORY are moderate positive/
   negative shaping terms — real signal, but the JD explicitly says the
   experience range is "flexible for the right candidate" and doesn't
   single these out as primary differentiators the way it does for the
   four must-haves and the explicit traps.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
from scoring.candidate_features import compute_all_features


def compute_quality_score(flat: Dict[str, Any]) -> Dict[str, Any]:
    """
    The "is this person good for the role" score, BEFORE availability and
    honeypot/noise gating are applied. Range is unbounded-ish but typically
    falls in roughly [-1, 2] given the component ranges below; only the
    relative ordering matters, not the absolute scale.
    """
    # --- Component 1: must-have fit, gaming-vector-resistant (dominant term) ---
    corroborated_ratio = flat.get("must_have_corroborated_ratio", 0.0) or 0.0
    mismatch_ratio = flat.get("title_desc_mismatch_ratio")
    # AUDIT FIX (gaming vector): discount corroboration credit by how much
    # the candidate's own title/description text looks internally
    # inconsistent. mismatch_ratio is None when no checkable title was
    # found at all — in that case we don't penalize (no evidence either way).
    mismatch_discount = 1.0 if mismatch_ratio is None else (1.0 - mismatch_ratio)
    must_have_term = corroborated_ratio * mismatch_discount * 1.0  # weight 1.0 — dominant

    # --- Component 2: nice-to-haves (small bonus, JD explicitly calls these optional) ---
    nice_to_have_term = (flat.get("nice_to_have_count", 0) / 5.0) * 0.10

    # --- Component 3: skill corroboration ratio + field coherence (depth & focus) ---
    skill_corrob = flat.get("skill_corroboration_ratio") or 0.0
    skill_term = skill_corrob * 0.30

    # --- Component 4: career trajectory shaping ---
    traj_term = 0.0
    if flat.get("has_product_company_experience"):
        traj_term += 0.15
    if flat.get("pure_consulting_career"):
        traj_term -= 0.25
    if flat.get("title_chaser"):
        traj_term -= 0.20

    # --- Component 5: explicit JD disqualifiers (structural proxies) ---
    disq_term = 0.0
    for disq_key in ("disq_pure_research_no_production", "disq_langchain_only_recent",
                      "disq_cv_speech_robotics_no_nlp"):
        if flat.get(disq_key):
            disq_term -= 0.30

    # --- Component 6: experience band + company tier + location (moderate shaping) ---
    exp_term = (flat.get("experience_band_score", 0.5) - 0.5) * 0.20
    company_term = flat.get("avg_company_tier_score", 0.0) * 0.15
    location_term = (flat.get("location_fit_score", 0.5) - 0.5) * 0.10

    quality = (
        must_have_term + nice_to_have_term + skill_term + traj_term
        + disq_term + exp_term + company_term + location_term
    )

    return {
        "must_have_term": round(must_have_term, 4),
        "mismatch_discount_applied": round(mismatch_discount, 4),
        "nice_to_have_term": round(nice_to_have_term, 4),
        "skill_term": round(skill_term, 4),
        "trajectory_term": round(traj_term, 4),
        "disqualifier_term": round(disq_term, 4),
        "experience_term": round(exp_term, 4),
        "company_term": round(company_term, 4),
        "location_term": round(location_term, 4),
        "quality_score": round(quality, 4),
    }


def compute_final_score(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full pipeline for one candidate: compute all features, derive the
    quality score, then apply the multiplicative gates (honeypot, noise,
    availability) per the rationale documented at the top of this file.
    """
    result = compute_all_features(cand)
    flat = result["flat"]

    quality = compute_quality_score(flat)
    quality_score = quality["quality_score"]

    # Multiplicative gates, applied in order of severity:
    is_honeypot = flat.get("is_likely_honeypot", False)
    noise = flat.get("profile_noise_score", 0.0) or 0.0
    availability = flat.get("availability_multiplier", 0.5) or 0.5

    if is_honeypot:
        # Hard gate: honeypots are pushed to the bottom regardless of
        # quality_score. Not literally zero (avoids weird tie-breaking
        # against equally-bad real candidates) but heavily suppressed.
        honeypot_multiplier = 0.02
    else:
        honeypot_multiplier = 1.0

    noise_multiplier = 1.0 - (noise * 0.5)  # noise discounts, doesn't zero out
    noise_multiplier = max(0.3, noise_multiplier)

    final_score = quality_score * honeypot_multiplier * noise_multiplier * availability

    return {
        "candidate_id": flat["candidate_id"],
        "quality_score": quality_score,
        "quality_breakdown": quality,
        "is_likely_honeypot": is_honeypot,
        "honeypot_multiplier": honeypot_multiplier,
        "profile_noise_score": round(noise, 4),
        "noise_multiplier": round(noise_multiplier, 4),
        "availability_multiplier": round(availability, 4),
        "final_score": round(final_score, 6),
        "flat_features": flat,
    }


if __name__ == "__main__":
    from parsing.streaming_reader import iter_candidates

    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "candidates.jsonl")

    test_ids = {"CAND_0001610", "CAND_0000100", "CAND_0003582", "CAND_0000082"}
    for cand in iter_candidates(data_path):
        if cand["candidate_id"] in test_ids:
            r = compute_final_score(cand)
            print(
                r["candidate_id"], "-> final_score:", round(r["final_score"], 4),
                "| quality:", round(r["quality_score"], 4),
                "| honeypot_mult:", r["honeypot_multiplier"],
                "| noise_mult:", r["noise_multiplier"],
                "| avail_mult:", r["availability_multiplier"],
            )
            test_ids.discard(cand["candidate_id"])
        if not test_ids:
            break
