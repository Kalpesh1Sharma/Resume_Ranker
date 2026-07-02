"""
Rule-based reasoning text generator.

HARD CONSTRAINT (submission_spec.docx): the ranking step must not call any
hosted LLM APIs. This generator builds reasoning strings entirely from
already-computed feature values via string templates — fast, deterministic,
and auditable (every claim traces directly to a real field).

Design goals, per submission_spec.docx's Stage 4 reasoning-quality review:
  - No empty or templated-identical reasoning across candidates.
  - No hallucinated skills/claims not present in the candidate's actual data.
  - Reasoning should not contradict the rank (a low-corroboration candidate
    shouldn't get reasoning that reads as a confident endorsement).
"""
from typing import Dict, Any


def generate_reasoning(flat: Dict[str, Any], cand: Dict[str, Any]) -> str:
    """
    Build a short, specific, honest reasoning string from real feature
    values for one candidate. Every clause below is gated on an actual
    field being true/present — nothing is asserted unconditionally.
    """
    p = cand.get("profile") or {}
    title = p.get("current_title", "Unknown title")
    company = p.get("current_company", "Unknown company")
    yoe = p.get("years_of_experience")

    is_honeypot = flat.get("is_likely_honeypot", False)
    corrob_count = flat.get("must_have_corroborated_count", 0) or 0
    corrob_total = 4  # len(MUST_HAVE_SIGNALS), kept as a literal to avoid an
                       # extra cross-module import in this lightweight generator
    noise = flat.get("profile_noise_score", 0.0) or 0.0
    avail = flat.get("availability_multiplier", 0.5) or 0.5
    has_product_exp = flat.get("has_product_company_experience", False)
    pure_consulting = flat.get("pure_consulting_career", False)
    company_tier = flat.get("current_company_tier", "unknown")
    recency = flat.get("recency_decay", 0.5) or 0.5
    response_rate_known = "recruiter_response_rate" in (cand.get("redrob_signals") or {})

    clauses = []

    # Opening: role + experience, always present
    yoe_str = f"{yoe:.1f} yrs" if isinstance(yoe, (int, float)) else "experience unspecified"
    clauses.append(f"{title} @ {company} ({yoe_str})")

    # Honeypot/noise flag takes priority — if flagged, say so plainly and stop
    # elaborating on positives, since this candidate should not be ranked
    # highly and the reasoning must not contradict that.
    if is_honeypot:
        clauses.append("profile shows internal inconsistencies (skill/experience timing "
                        "or proficiency claims don't add up) — flagged as low-confidence")
        return "; ".join(clauses) + "."

    # Must-have corroboration — the dominant, most defensible claim
    if corrob_count == corrob_total:
        clauses.append("all 4 core must-haves (embeddings/retrieval, vector search, "
                        "Python, eval frameworks) are backed by actual career history, "
                        "not just listed as skills")
    elif corrob_count >= 2:
        clauses.append(f"{corrob_count}/4 core must-haves corroborated by career history")
    elif corrob_count >= 1:
        clauses.append(f"only {corrob_count}/4 core must-haves corroborated by career "
                        f"history — limited evidence of hands-on must-have experience")
    else:
        clauses.append("no core must-haves corroborated by career history; any matching "
                        "skills appear unbacked by demonstrated work")

    # Company/trajectory context
    if has_product_exp and company_tier in ("product_ai_native", "product_global_tier1"):
        clauses.append(f"current/past company tier is {company_tier.replace('_', ' ')}")
    elif pure_consulting:
        clauses.append("career has been entirely at IT-services/consulting firms, "
                        "no product-company experience found")
    elif has_product_exp:
        clauses.append("has product-company experience")

    # Noise/quality flag, only mentioned if non-trivial
    if noise > 0.4:
        clauses.append("profile shows some inconsistency signals (title/description "
                        "mismatch or duplicate entries) — treat with extra scrutiny")

    # Availability, only mentioned if it meaningfully helps or hurts
    if recency < 0.3:
        clauses.append("inactive for an extended period — availability uncertain")
    elif avail > 0.85:
        clauses.append("recently active and responsive")

    return "; ".join(clauses) + "."


if __name__ == "__main__":
    import sys, os, json
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from parsing.streaming_reader import iter_candidates
    from scoring.scorer import compute_final_score

    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "candidates.jsonl")

    test_ids = {"CAND_0046064", "CAND_0000100", "CAND_0003582", "CAND_0000082"}
    for cand in iter_candidates(data_path):
        if cand["candidate_id"] in test_ids:
            r = compute_final_score(cand)
            reasoning = generate_reasoning(r["flat_features"], cand)
            print(f"{cand['candidate_id']} (score={r['final_score']:.4f}):")
            print(f"  {reasoning}")
            print()
            test_ids.discard(cand["candidate_id"])
        if not test_ids:
            break
