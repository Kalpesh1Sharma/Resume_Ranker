"""
Skill corroboration features.

Built from Day 1 finding #2: in weak/fake candidates, the skills list is
frequently "floating free" — high-proficiency skills (e.g. GANs, TTS, YOLO,
Pinecone) that NEVER appear anywhere in the candidate's actual career_history
descriptions. Real strong candidates (e.g. CAND_0001610) have their key
skills explicitly named and demonstrated in their job descriptions.

This is more subtle than simple keyword-stuffing detection (trap #1 in the
JD) because the skill names ARE real, plausible, often well-formed entries —
they just aren't backed by anything in the person's actual work history.

We don't discard uncorroborated skills outright (some real skills genuinely
aren't mentioned in a terse job description) — instead we compute a
corroboration RATIO and use it as a trust multiplier on the raw skill-match
score in the JD-fit module.

IMPORTANT CALIBRATION NOTE (found while validating against Day 1 examples):
this ratio does NOT approach 1.0 even for genuinely strong, honest candidates
— our gold-standard example (CAND_0001610, a verified strong fit) scored only
0.556, because real job descriptions are terse and don't name every tool
(e.g. their description says "FAISS" but never says "Pinecone" even though
they list Pinecone as an expert skill). Verified weak/honeypot candidates
scored 0.0-0.18 on the same metric, so the RELATIVE ordering is correct and
useful — just don't treat this as "ratio close to 1.0 = good" in absolute
terms. Use it as a comparative/relative signal in scoring, not a threshold.
"""
import re
from typing import Dict, Any, List
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.safe_access import safe_career_history, safe_profile, safe_skills


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())


def _career_history_blob(cand: Dict[str, Any]) -> str:
    """All career_history title + description text, normalized, concatenated."""
    parts = []
    for job in safe_career_history(cand):
        parts.append(job.get("title") or "")
        parts.append(job.get("description") or "")
    # also fold in the summary — people sometimes describe real work there
    # instead of in career_history descriptions
    parts.append(safe_profile(cand).get("summary") or "")
    return _normalize(" ".join(parts))


def _skill_appears_in_blob(skill_name: str, blob: str) -> bool:
    """
    Check if a skill name (or a reasonable stem of it) appears in the
    career-history text blob. Deliberately loose matching since people
    phrase things differently (e.g. skill "Sentence Transformers" vs
    description text "sentence-transformer embeddings").
    """
    name = _normalize(skill_name)
    if not name:
        return False
    # full match
    if name in blob:
        return True
    # try matching on the most distinctive word in a multi-word skill name
    # (skip generic/short words that would false-positive on everything)
    words = [w for w in name.split() if len(w) >= 4]
    if not words:
        return False
    # require at least one distinctive word to appear
    return any(w in blob for w in words)


def compute_skill_corroboration(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    For each skill, check whether it's corroborated by career_history/summary text.
    Returns overall ratio plus the list of high-proficiency UNCORROBORATED skills,
    which is the strongest version of the Day 1 red flag (e.g. Candidate 1's
    GANs/TTS/YOLO/RL scattering with a data-engineering-only career history).
    """
    skills = safe_skills(cand)
    if not skills:
        return {
            "skill_corroboration_ratio": None,
            "uncorroborated_high_prof_skills": [],
            "uncorroborated_high_prof_count": 0,
            "corroborated_skill_count": 0,
            "total_skill_count": 0,
        }

    blob = _career_history_blob(cand)
    corroborated = 0
    uncorroborated_high_prof = []

    for sk in skills:
        name = sk.get("name", "")
        prof = sk.get("proficiency")
        is_backed = _skill_appears_in_blob(name, blob)
        if is_backed:
            corroborated += 1
        elif prof in ("advanced", "expert"):
            # only flag HIGH-proficiency uncorroborated skills as the red flag —
            # a "beginner, 3 months" skill not mentioned in a terse job
            # description is normal and not suspicious on its own.
            uncorroborated_high_prof.append({"name": name, "proficiency": prof})

    ratio = corroborated / len(skills)

    return {
        "skill_corroboration_ratio": round(ratio, 3),
        "uncorroborated_high_prof_skills": uncorroborated_high_prof,
        "uncorroborated_high_prof_count": len(uncorroborated_high_prof),
        "corroborated_skill_count": corroborated,
        "total_skill_count": len(skills),
    }


def skill_field_coherence(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Day 1 observation: weak candidates often have skills scattered across many
    UNRELATED AI subfields at once (GANs + TTS + YOLO + Reinforcement Learning +
    Recommendation Systems all at "advanced") — a real specialist usually has
    depth concentrated in 1-2 related areas, not breadth across all of them
    at high proficiency. This is a soft coherence signal, not a hard rule.

    BUG FIX (found via audit): the original version counted RAW skill
    presence regardless of corroboration, which produced an inverted result
    — our verified gold-standard candidate (CAND_0001610) scored breadth=5,
    HIGHER than a known-weak scattered-skills candidate (CAND_0000082,
    breadth=4). Root cause: CAND_0001610 has one uncorroborated stray skill
    ("Speech Recognition: advanced") that's nowhere in their actual career
    history, inflating the count by one field, while the rest of their
    breadth (retrieval_ranking, nlp_llm, mlops_infra, classical_ml) IS
    genuinely demonstrated in their work — legitimate breadth, not
    suspicious scatter. Now corroboration-aware: a field only counts as
    "touched" if at least one of its triggering skills is also backed by
    career_history/summary text, consistent with how must_have_coverage in
    jd_fit.py already distinguishes claimed vs. demonstrated.
    """
    FIELD_GROUPS = {
        "retrieval_ranking": ["embeddings", "vector search", "rag", "information retrieval",
                               "recommendation systems", "ranking", "sentence transformers",
                               "faiss", "pinecone", "weaviate", "qdrant", "milvus", "hybrid search"],
        "nlp_llm": ["nlp", "llm", "fine-tuning llms", "hugging face transformers",
                    "prompt engineering", "haystack", "llamaindex"],
        "computer_vision": ["computer vision", "object detection", "yolo", "image classification",
                             "gans", "diffusion models"],
        "speech": ["speech recognition", "asr", "tts"],
        "reinforcement_learning": ["reinforcement learning"],
        "classical_ml": ["scikit-learn", "xgboost", "feature engineering", "statistical modeling"],
        "data_engineering": ["spark", "airflow", "kafka", "dbt", "data pipelines", "hadoop"],
        "mlops_infra": ["mlflow", "kubeflow", "bentoml", "docker", "kubernetes", "mlops"],
    }
    high_prof_skill_names = [s.get("name", "").lower() for s in safe_skills(cand)
                              if s.get("proficiency") in ("advanced", "expert")]
    blob = _career_history_blob(cand)

    fields_touched_raw = set()
    fields_touched_corroborated = set()
    for field, kws in FIELD_GROUPS.items():
        matching_skills = [sk for sk in high_prof_skill_names if any(kw in sk for kw in kws)]
        if matching_skills:
            fields_touched_raw.add(field)
            # corroborated if ANY matching skill's keyword also appears in career_history
            if any(any(kw in blob for kw in kws) for sk in matching_skills):
                fields_touched_corroborated.add(field)

    return {
        "high_prof_field_breadth": len(fields_touched_corroborated),  # now the corroborated count
        "high_prof_field_breadth_raw": len(fields_touched_raw),       # kept for transparency/debugging
        "high_prof_fields": sorted(fields_touched_corroborated),
        "high_prof_fields_raw": sorted(fields_touched_raw),
    }


if __name__ == "__main__":
    import sys, json
    sys.path.insert(0, "..")
    from parsing.streaming_reader import iter_candidates

    test_ids = {"CAND_0000112", "CAND_0001610", "CAND_0000131"}
    for cand in iter_candidates("../../data/candidates.jsonl"):
        if cand["candidate_id"] in test_ids:
            corrob = compute_skill_corroboration(cand)
            coherence = skill_field_coherence(cand)
            print(cand["candidate_id"], "->")
            print("  corroboration:", json.dumps(corrob, indent=2))
            print("  coherence:", json.dumps(coherence, indent=2))
            test_ids.discard(cand["candidate_id"])
        if not test_ids:
            break
