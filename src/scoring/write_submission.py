"""
Final submission CSV writer.

DAY 5 UPDATE: Two-stage ranking pipeline with semantic reranking.

ARCHITECTURE:
  Stage 1 — Rule scoring (all 100K candidates)
    All candidates scored with the rule-based feature pipeline.
    Produces a ranked list. Top 500 advance to Stage 2.
    Rationale: a candidate with 1/4 corroborated must-haves should never
    jump into the top-100 solely because they use the right vocabulary.
    The rule-based gate ensures semantic only reranks genuinely strong
    candidates.

  Stage 2 — Semantic reranking (top 500 only)
    Blend rule score with JD cosine similarity:
      final = base_score * 0.90 + semantic_score * 0.10
    Conservative weight by design — semantic supports, doesn't override.
    Multiplicative blend avoids calibration distortion from additive mixing.

  Degrades gracefully: if embedding files are not found, falls back to
  rule-only scoring with a warning. The pipeline always produces a valid
  submission regardless of whether embeddings are present.

Submission format rules (unchanged from Day 3):
  - Header: candidate_id,rank,score,reasoning
  - Exactly 100 data rows, ranks 1-100 each exactly once
  - score must be NON-INCREASING by rank
  - Equal scores: candidate_id ASCENDING tie-break
  - UTF-8 encoding
"""
import sys, os, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsing.streaming_reader import iter_candidates
from scoring.scorer import compute_final_score
from scoring.reasoning import generate_reasoning
from features.semantic import get_semantic_scorer

TOP_N = 100
SEMANTIC_POOL = 500   # only rerank top-500 with embeddings
SEMANTIC_WEIGHT = 0.10
RULE_WEIGHT = 0.90


def run_full_ranking(data_path: str):
    """
    Two-stage ranking pipeline.
    Stage 1: rule-based scoring across all 100K.
    Stage 2: semantic blend on top-500 only.
    """
    # Load semantic scorer once — 0.7s, then O(1) lookups
    semantic_scorer = get_semantic_scorer()
    if semantic_scorer:
        print("[ranking] Semantic scorer loaded — two-stage pipeline active",
              file=sys.stderr)
    else:
        print("[ranking] WARNING: Semantic scorer not available — "
              "falling back to rule-only scoring", file=sys.stderr)

    # Stage 1: rule-based scoring, all 100K
    scored = []
    for cand in iter_candidates(data_path):
        r = compute_final_score(cand)
        scored.append({
            "candidate_id": r["candidate_id"],
            "base_score": r["final_score"],
            "cand": cand,
            "flat_features": r["flat_features"],
        })

    # Sort by base score to identify top-500
    scored.sort(key=lambda x: (-x["base_score"], x["candidate_id"]))

    # Stage 2: semantic blend on top-500 only
    if semantic_scorer:
        for item in scored[:SEMANTIC_POOL]:
            cid = item["candidate_id"]
            sem_sim = semantic_scorer.similarity(cid)
            # Normalise similarity from [-1,1] to [0,1] range
            # In practice all-MiniLM similarities are [0.3, 0.8] for our data
            # but we normalise defensively
            sem_norm = (sem_sim + 1.0) / 2.0
            # Multiplicative blend — avoids additive calibration distortion
            item["score"] = (
                item["base_score"] * RULE_WEIGHT +
                sem_norm * SEMANTIC_WEIGHT
            )
        # Candidates outside top-500 keep their base score unchanged
        for item in scored[SEMANTIC_POOL:]:
            item["score"] = item["base_score"]
    else:
        for item in scored:
            item["score"] = item["base_score"]

    # Re-sort after semantic blend (scores may have shuffled top-500)
    # Explicit two-key sort: score descending, candidate_id ascending for ties
    # This matches validate_submission.py's exact tie-break requirement
    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    # Build final output — round BEFORE final sort so the order matches
    # what gets written to CSV (avoids tie-break violations from rounding)
    top_100 = []
    for item in scored[:TOP_N]:
        reasoning = generate_reasoning(item["flat_features"], item["cand"])
        top_100.append({
            "candidate_id": item["candidate_id"],
            "score": round(item["score"], 4),
            "reasoning": reasoning,
        })

    # Final sort on rounded scores with candidate_id tie-break
    top_100.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    return top_100


def write_submission_csv(top_candidates, output_path: str):
    """
    Write the final CSV with explicit invariant checking before writing.
    """
    rows = []
    for i, c in enumerate(top_candidates, start=1):
        rows.append({
            "candidate_id": c["candidate_id"],
            "rank": i,
            "score": round(c["score"], 4),
            "reasoning": c["reasoning"],
        })

    # Explicit self-check — rounding can introduce violations
    for i in range(len(rows) - 1):
        s1, s2 = rows[i]["score"], rows[i + 1]["score"]
        if s1 < s2:
            raise RuntimeError(
                f"Non-increasing score violation after rounding at rank "
                f"{rows[i]['rank']} ({s1}) < rank {rows[i+1]['rank']} ({s2})"
            )
        if s1 == s2 and rows[i]["candidate_id"] > rows[i + 1]["candidate_id"]:
            raise RuntimeError(
                f"Tie-break violation at rank {rows[i]['rank']} / "
                f"{rows[i+1]['rank']}: {rows[i]['candidate_id']} > "
                f"{rows[i+1]['candidate_id']}"
            )

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in rows:
            writer.writerow([
                row["candidate_id"], row["rank"],
                row["score"], row["reasoning"]
            ])

    return output_path


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "candidates.jsonl")
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "output", "submission_day5.csv")

    t0 = time.time()
    top_100 = run_full_ranking(data_path)
    elapsed = time.time() - t0

    write_submission_csv(top_100, output_path)

    print(f"Total time: {elapsed:.1f}s")
    print(f"Wrote: {output_path}")
    print(f"Top 5:")
    for c in top_100[:5]:
        print(f"  {c['candidate_id']}: {round(c['score'],4)}")
