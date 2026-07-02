"""
Structured representation of the Redrob JD, extracted by hand from job_description.docx.

This is the single source of truth for scoring weights and rules.
Every feature/scoring module should import FROM HERE, not hardcode JD logic elsewhere.

Design note: the JD explicitly says "we'd rather see 10 great matches than 1000 maybes."
This means precision at the top matters more than recall across the pool — a conservative
scorer that confidently ranks 10-30 great candidates at the very top beats one that
spreads moderate scores across hundreds of "pretty good" candidates. This directly
supports optimizing for NDCG@10.
"""

# ---------------------------------------------------------------------------
# CORE MUST-HAVES (things you absolutely need)
# Absence of ALL of these = very unlikely to be a real fit, regardless of years.
# ---------------------------------------------------------------------------
MUST_HAVE_SIGNALS = {
    "embeddings_retrieval": {
        "description": "Production experience with embeddings-based retrieval, deployed to real users",
        "keywords": [
            "sentence-transformers", "sentence transformers", "openai embeddings",
            "bge", "e5 embeddings", "embedding", "embeddings", "semantic search",
            "dense retrieval", "vector search",
        ],
        "note": "JD explicitly says specific model doesn't matter — operational experience "
                 "(embedding drift, index refresh, retrieval-quality regression) does. "
                 "Look for evidence in career_history descriptions, not just skills list.",
    },
    "vector_db_or_hybrid_search": {
        "description": "Production experience with vector DBs or hybrid search infra",
        "keywords": [
            "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
            "elasticsearch", "faiss", "vector database", "vector db", "hybrid search",
        ],
    },
    "strong_python": {
        "description": "Strong Python, code quality matters",
        "keywords": [
            "python",
            # Python-ecosystem ML/data tooling implies Python even when the word
            # "python" itself isn't stated (common in this dataset — see
            # CAND_0001610, a verified strong candidate whose data never
            # literally contains the word "python" despite heavy PyTorch/
            # scikit-learn/XGBoost usage).
            "pytorch", "scikit-learn", "tensorflow", "xgboost", "lightgbm",
            "pandas", "numpy", "fastapi", "django", "flask",
        ],
        "note": "Necessary but nowhere near sufficient on its own — almost every "
                 "candidate will have this; use as a gate not a differentiator. "
                 "Inferred from ecosystem tooling, not just the literal word, since "
                 "many real profiles omit 'Python' explicitly while clearly using it.",
    },
    "eval_frameworks": {
        "description": "Hands-on experience designing evaluation frameworks for ranking systems",
        "keywords": [
            "ndcg", "mrr", "map", "precision@", "recall@",
            "a b test", "ab test",
            "offline evaluation", "online evaluation", "ranking evaluation",
            "evaluation framework",
            "offline online correlation", "offline-online correlation",
            "offline to online correlation", "offline-to-online correlation",
            "relevance labeling", "relevance judgment", "human relevance",
            "search relevance",
        ],
        "note": "This is the most differentiating must-have — most candidates with "
                 "embeddings/vector-db experience will NOT have explicit eval-framework "
                 "experience. Strong positive signal when present. IMPORTANT: text is "
                 "normalized (lowercased, punctuation stripped) before matching — so "
                 "keyword variants must account for both hyphenated and non-hyphenated, "
                 "slash and non-slash phrasing (e.g. 'a/b test' becomes 'a b test').",
    },
}

# ---------------------------------------------------------------------------
# NICE-TO-HAVES (bonus signal, never required, never disqualifying if absent)
# ---------------------------------------------------------------------------
NICE_TO_HAVE_SIGNALS = {
    "llm_finetuning": ["lora", "qlora", "peft", "fine-tuning", "finetuning", "fine tune"],
    "learning_to_rank": ["learning to rank", "ltr", "xgboost ranker", "neural ranking", "lambdamart"],
    "hr_tech_background": ["hr tech", "recruiting", "talent", "hiring platform", "ats", "marketplace"],
    "distributed_systems": ["distributed systems", "kubernetes", "kafka", "spark", "large-scale inference"],
    "open_source": ["open source", "open-source", "github contributor", "oss maintainer"],
}

# ---------------------------------------------------------------------------
# HARD / SOFT DISQUALIFIERS
# These should produce strong NEGATIVE adjustments, not just "fail to add points."
# Distinguish hard (near-zero score) vs soft (heavy penalty but not auto-zero),
# matching the JD's own language ("will not move forward" vs "probably will not").
# ---------------------------------------------------------------------------
DISQUALIFIERS = {
    "pure_research_no_production": {
        "severity": "hard",
        "jd_language": "we will not move forward",
        "detection": "Career history entirely in academic/research labs, zero evidence "
                      "of shipping to production/real users.",
    },
    "langchain_only_recent": {
        "severity": "soft",
        "jd_language": "we will probably not move forward, unless substantial pre-LLM-era "
                        "ML production experience",
        "detection": "AI-related experience is <12 months AND consists primarily of "
                      "LangChain+OpenAI API calls, with no earlier ML/IR/search production work.",
    },
    "stale_ic_18mo": {
        "severity": "soft",
        "jd_language": "we will probably not move forward",
        "detection": "Senior title with current/recent role description suggesting pure "
                      "architecture/tech-lead/management for 18+ months, no hands-on coding signal.",
    },
    "title_chaser": {
        "severity": "soft",
        "jd_language": "we're not a fit",
        "detection": "Career history shows title escalation (Senior->Staff->Principal or "
                      "similar) with company changes roughly every ~1.5 years or less, "
                      "repeated across 3+ jobs.",
    },
    "framework_enthusiast": {
        "severity": "soft",
        "jd_language": "not what we need",
        "detection": "Hard to detect from structured data alone (this is mostly a GitHub/blog "
                      "signal we don't have). Weak proxy: many trendy-framework skill keywords "
                      "with low proficiency/endorsements/duration_months, no systems-level "
                      "experience signal in career_history.",
    },
    "pure_consulting_career": {
        "severity": "hard",
        "jd_language": "bad fit experiences in both directions",
        "detection": "ALL career_history entries are at consulting firms (TCS, Infosys, Wipro, "
                      "Accenture, Cognizant, Capgemini, etc.) with zero product-company experience. "
                      "NOTE: currently at one of these but with PRIOR product-company experience "
                      "is explicitly fine — only penalize if consulting is the ENTIRE career.",
    },
    "cv_speech_robotics_no_nlp": {
        "severity": "hard",
        "jd_language": "you'd be re-learning fundamentals here",
        "detection": "Primary expertise (skills + career_history) is computer vision, speech, "
                      "or robotics, with no NLP/IR/search/ranking exposure.",
    },
    "closed_source_no_validation": {
        "severity": "soft",
        "jd_language": "we need to see how you think, not just trust that you can think",
        "detection": "5+ years entirely on closed-source proprietary systems, zero external "
                      "validation signal (we approximate this with github_activity_score == -1 "
                      "combined with no certifications/no notable public signal).",
    },
}

CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "ibm consulting",
    "mindtree", "mphasis",
    # Found via full-dataset company frequency analysis: Mindtree and Mphasis
    # are major IT-services/consulting firms appearing ~2,800 times each in
    # the dataset (same tier as Capgemini/Accenture/Cognizant), and were
    # missing from the original keyword list derived purely from the JD text.
]

CV_SPEECH_ROBOTICS_KEYWORDS = [
    "computer vision", "cv engineer", "image recognition", "object detection",
    "speech recognition", "asr", "text-to-speech", "robotics", "slam",
    "autonomous navigation", "motion planning",
]

NLP_IR_RESCUE_KEYWORDS = [
    "nlp", "natural language", "information retrieval", "search", "ranking",
    "text classification", "named entity", "ner", "embeddings", "retrieval",
]

# ---------------------------------------------------------------------------
# IDEAL CANDIDATE PROFILE (the JD's own description — used for scoring shape)
# ---------------------------------------------------------------------------
IDEAL_PROFILE = {
    "years_experience_range": (6, 8),       # "roughly" — not a hard cutoff, JD says band is flexible
    "years_experience_soft_range": (5, 9),  # stated range, also flexible
    "applied_ml_years_min": 4,              # of which 4-5 in applied ML/AI at PRODUCT companies
    "applied_ml_years_target": 5,
    "requires_product_company_experience": True,
    "requires_shipped_ranking_search_or_recsys": True,  # "at meaningful scale"
    "requires_defensible_opinions": True,   # hard to detect structurally; proxy via depth signals
}

# ---------------------------------------------------------------------------
# LOCATION / LOGISTICS
# ---------------------------------------------------------------------------
PREFERRED_LOCATIONS = ["pune", "noida"]
ACCEPTABLE_LOCATIONS = ["hyderabad", "mumbai", "delhi ncr", "delhi", "gurgaon", "gurugram", "ncr"]
# Outside India: case-by-case, no visa sponsorship — treat as low/no fit unless willing_to_relocate
NOTICE_PERIOD_IDEAL_DAYS = 30   # "love sub-30-day notice... can buy out up to 30 days"
NOTICE_PERIOD_HIGHER_BAR_DAYS = 30  # 30+ days: still in scope but bar is higher

# ---------------------------------------------------------------------------
# THE EXPLICIT HACKATHON TRAPS (stated directly in the JD's final section)
# ---------------------------------------------------------------------------
EXPLICIT_TRAPS = """
1. Keyword-stuffing trap: candidates whose skills section contains the most AI
   keywords are NOT automatically the right answer. Must reason about career
   history, not just skill list.
2. Title-without-substance trap: a candidate with all the right skill keywords
   but title "Marketing Manager" (or similar non-technical role) is NOT a fit,
   regardless of skill list completeness.
3. Implicit-fit trap (inverse of #1): a candidate who never uses words like "RAG"
   or "Pinecone" but whose career_history shows they built a recommendation
   system at a product company IS a fit. Don't require exact keyword matches.
4. Availability trap: a perfect-on-paper candidate inactive 6+ months with low
   recruiter_response_rate is NOT "actually available" for hiring purposes —
   must down-weight using redrob_signals regardless of how good the resume looks.
5. Honeypot candidates (~80 in the dataset): subtly impossible profiles
   (e.g., experience duration exceeding company's actual existence, expert
   proficiency with ~0 duration_months). Must be filtered via internal
   consistency checks, not just quality scoring.
"""

# ---------------------------------------------------------------------------
# SCORING WEIGHT SKELETON (tune in Day 4 against your hand-labeled set)
# Kept here, not in scoring code, so it's visible and easy to defend in Stage 5.
# ---------------------------------------------------------------------------
SCORING_WEIGHTS_V1 = {
    "title_role_fit": 0.30,
    "skills_match": 0.25,
    "experience_quality": 0.20,
    "behavioral_signals": 0.15,
    "education_location": 0.10,
}

if __name__ == "__main__":
    import json
    print("Must-have signal categories:", list(MUST_HAVE_SIGNALS.keys()))
    print("Disqualifier categories:", list(DISQUALIFIERS.keys()))
    print("Scoring weights:", json.dumps(SCORING_WEIGHTS_V1, indent=2))
