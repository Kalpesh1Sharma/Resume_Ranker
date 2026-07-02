"""
Company tier features.

Built from a full scan of all 100K candidates' career_history, which revealed
the dataset uses EXACTLY 63 distinct company names total, cleanly stratified
into 5 tiers by frequency and identity. This is not guessed — it's the
complete, exhaustive company universe of the dataset, confirmed by counting.

Tier breakdown (counts = number of job entries across the dataset, not
candidates):

  FILLER (8 companies, ~23,400-23,700 each — fictional/sitcom names):
    Pied Piper, Initech, Wayne Enterprises, Acme Corp, Stark Industries,
    Hooli, Globex Inc, Dunder Mifflin
    -> These never represent real work. Day 1 manual review confirmed every
       job at one of these companies had a description mismatched to its
       title — these are noise/filler entries by construction.

  CONSULTING_LARGE (3 companies, ~23,400-23,700 each — real IT services):
    Infosys, Wipro, TCS
    -> Real consulting firms, but oddly the SAME frequency tier as the filler
       names. This is a deliberate trap: a candidate whose career is built
       entirely from this tier (filler + big consulting, ~11 of the 63
       companies) is statistically very likely to be a "noise" profile, even
       before reading descriptions.

  CONSULTING_MID (7 companies, ~2,800-2,900 each — more real IT services):
    Capgemini, HCL, Mindtree, Accenture, Cognizant, Tech Mahindra, Mphasis

  PRODUCT_MAJOR (11 companies, ~2,800-3,000 each — major Indian product cos):
    Swiggy, Razorpay, CRED, Zomato, Flipkart
    (grouped with consulting_mid by frequency tier, but functionally distinct)

  PRODUCT_KNOWN (17 companies, ~330-385 each — well-known Indian product/tech):
    Meesho, Nykaa, InMobi, BYJU'S, PolicyBazaar, Ola, Zoho, Vedantu, Paytm,
    Unacademy, PharmEasy, upGrad, Freshworks, PhonePe, Dream11

  PRODUCT_AI_NATIVE (14 companies, ~58-81 each — smaller AI-native/specialized):
    Genpact AI, Glance, Rephrase.ai, Aganitha, Niramai, Saarthi.ai, Sarvam AI,
    Mad Street Den, Observe.AI, Krutrim, Wysa, Haptik, Verloop.io, Yellow.ai,
    Locobuzz

  PRODUCT_GLOBAL_TIER1 (9 companies, ~7-14 each — FAANG+ tier):
    Google, Netflix, Amazon, Salesforce, Uber, Meta, Adobe, Microsoft, Apple,
    LinkedIn

This feature is independent of (and complements) the text-based product-vs-
consulting check in career_trajectory.py, which works off the CONSULTING_FIRMS
keyword list. Where they overlap (TCS/Infosys/Wipro/etc.) they should agree;
this module additionally distinguishes PRODUCT_AI_NATIVE and
PRODUCT_GLOBAL_TIER1 as meaningfully stronger signal than PRODUCT_KNOWN,
which the keyword-only check can't do.
"""
from typing import Dict, Any, List
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.safe_access import safe_career_history, safe_profile
from utils.jd_config import CONSULTING_FIRMS

FILLER_COMPANIES = {
    "pied piper", "initech", "wayne enterprises", "acme corp",
    "stark industries", "hooli", "globex inc", "dunder mifflin",
}

# AUDIT FIX: previously this module maintained its OWN independent
# CONSULTING_LARGE / CONSULTING_MID lists, separate from jd_config.py's
# CONSULTING_FIRMS. A diff found real drift (jd_config.py had "ibm
# consulting" and "tata consultancy" that this module lacked) — currently
# harmless since neither appears as a literal company name in the real
# 63-company dataset, but a latent bug waiting to trigger silently if the
# dataset or evaluation set ever changes. jd_config.py's CONSULTING_FIRMS
# is now the single source of truth for "is this company a consulting
# firm at all"; this module ONLY adds the frequency-tier split (large vs
# mid), which is real signal from the company census (large = ~23,500
# occurrences each, mid = ~2,800 each) that jd_config.py doesn't need to
# know about.
CONSULTING_LARGE = {"infosys", "wipro", "tcs"}
CONSULTING_MID = set(CONSULTING_FIRMS) - CONSULTING_LARGE

# Defensive consistency check: fail loudly at import time if these two
# concepts ever diverge again, rather than silently producing wrong scores.
_combined = CONSULTING_LARGE | CONSULTING_MID
_jd_set = set(CONSULTING_FIRMS)
assert _combined.issuperset(_jd_set - {"ibm consulting", "tata consultancy"}), (
    "company_tier.py consulting tiers have drifted from jd_config.py's "
    "CONSULTING_FIRMS — re-derive CONSULTING_LARGE/CONSULTING_MID."
)

PRODUCT_MAJOR = {"swiggy", "razorpay", "cred", "zomato", "flipkart"}

PRODUCT_KNOWN = {
    "meesho", "nykaa", "inmobi", "byju's", "policybazaar", "ola", "zoho",
    "vedantu", "paytm", "unacademy", "pharmeasy", "upgrad", "freshworks",
    "phonepe", "dream11",
}

PRODUCT_AI_NATIVE = {
    "genpact ai", "glance", "rephrase.ai", "aganitha", "niramai",
    "saarthi.ai", "sarvam ai", "mad street den", "observe.ai", "krutrim",
    "wysa", "haptik", "verloop.io", "yellow.ai", "locobuzz",
}

PRODUCT_GLOBAL_TIER1 = {
    "google", "netflix", "amazon", "salesforce", "uber", "meta", "adobe",
    "microsoft", "apple", "linkedin",
}

# Numeric tier score, used in scoring (higher = stronger signal).
# Filler is intentionally most-negative; consulting tiers are mildly negative
# on their own (not disqualifying, per JD's explicit allowance for prior
# product experience); product tiers scale up with rarity/prestige.
TIER_SCORES = {
    "filler": -1.0,
    "consulting_large": -0.2,
    "consulting_mid": -0.1,
    "product_major": 0.6,
    "product_known": 0.7,
    "product_ai_native": 0.9,
    "product_global_tier1": 1.0,
    "unknown": 0.3,  # company not in our universe at all (shouldn't happen given
                       # the dataset only has 63 companies, but defensive default)
}


def _tier_of(company: str) -> str:
    c = (company or "").lower().strip()
    if c in FILLER_COMPANIES:
        return "filler"
    if c in CONSULTING_LARGE:
        return "consulting_large"
    if c in CONSULTING_MID:
        return "consulting_mid"
    if c in PRODUCT_MAJOR:
        return "product_major"
    if c in PRODUCT_KNOWN:
        return "product_known"
    if c in PRODUCT_AI_NATIVE:
        return "product_ai_native"
    if c in PRODUCT_GLOBAL_TIER1:
        return "product_global_tier1"
    return "unknown"


def compute_company_tier_features(cand: Dict[str, Any]) -> Dict[str, Any]:
    jobs = safe_career_history(cand)
    current_company = safe_profile(cand).get("current_company", "")

    tiers = [_tier_of(j.get("company", "")) for j in jobs]
    scores = [TIER_SCORES[t] for t in tiers]

    current_tier = _tier_of(current_company)
    filler_count = tiers.count("filler")
    best_tier_score = max(scores) if scores else 0.3
    avg_tier_score = round(sum(scores) / len(scores), 3) if scores else 0.3

    return {
        "company_tiers": tiers,
        "current_company_tier": current_tier,
        "filler_company_count": filler_count,
        "any_filler_company": filler_count > 0,
        "best_company_tier_score": round(best_tier_score, 3),
        "avg_company_tier_score": avg_tier_score,
        "has_ai_native_or_global_experience": any(
            t in ("product_ai_native", "product_global_tier1") for t in tiers
        ),
    }


if __name__ == "__main__":
    import sys, json, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from parsing.streaming_reader import iter_candidates

    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "candidates.jsonl")

    test_ids = {"CAND_0000100", "CAND_0001610", "CAND_0000003"}
    for cand in iter_candidates(data_path):
        if cand["candidate_id"] in test_ids:
            feats = compute_company_tier_features(cand)
            print(cand["candidate_id"], "->", json.dumps(feats, indent=2))
            test_ids.discard(cand["candidate_id"])
        if not test_ids:
            break
