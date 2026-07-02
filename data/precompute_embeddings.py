"""
precompute_embeddings.py

Run this ONCE on your local machine — not in the sandbox.
It downloads all-MiniLM-L6-v2 (91MB, one-time), embeds every candidate's
profile text, and saves three .npy files that the ranking pipeline loads
at inference time.

Usage:
    python3 precompute_embeddings.py \
        --candidates /path/to/candidates.jsonl \
        --output_dir ./artifacts

Output files (upload all three to Claude after running):
    artifacts/candidate_ids.npy        -- shape (100000,) dtype str
    artifacts/candidate_embeddings.npy -- shape (100000, 384) dtype float32
    artifacts/jd_embedding.npy         -- shape (1, 384) dtype float32

Runtime estimate:
    CPU only: ~10-20 minutes for 100K candidates
    GPU:      ~2-3 minutes

Total output size: ~150MB
"""

import argparse
import json
import os
import sys
import time
import numpy as np

# -----------------------------------------------------------------
# JD TEXT — the exact job description we're ranking against.
# Richer than just the title; includes must-haves and key signals.
# -----------------------------------------------------------------
JD_TEXT = """
Senior AI Engineer — Redrob AI

Must-have experience:
Production embeddings-based retrieval using sentence-transformers, BGE, E5,
OpenAI embeddings or similar, deployed to real users at scale.
Production vector databases and hybrid search: Pinecone, Weaviate, Qdrant,
Milvus, FAISS, Elasticsearch, OpenSearch.
Strong Python engineering, code quality, production systems.
Hands-on evaluation framework design: NDCG, MRR, MAP, precision recall,
offline evaluation, online evaluation, offline online correlation analysis,
A/B testing for ranking systems, relevance labeling, human relevance judgment,
search relevance, ranking evaluation.

Nice to have:
LLM fine-tuning LoRA QLoRA PEFT.
Learning to rank XGBoost LightGBM neural ranking LambdaMART.
Recommendation systems collaborative filtering content-based ranking.
Information retrieval semantic search dense retrieval hybrid retrieval.
NLP natural language processing text classification named entity recognition.
Distributed systems Kubernetes Kafka large-scale inference.
Open source contributor.

Role context:
Building intelligence layer for talent platform.
Candidate JD matching ranking retrieval at scale.
Shipping real systems to real users, not research.
Product company experience preferred.
Pune Noida location preferred.
"""


def build_candidate_text(cand: dict) -> str:
    """
    Build rich profile text from a candidate dict.
    Uses: title, summary, career history, skills.
    Matches the architecture recommendation:
      - Career descriptions alone are too sparse
      - Skills contain high-signal terms (FAISS, NDCG, pgvector)
      - Rich text produces meaningfully better embeddings
    """
    p = cand.get("profile") or {}
    parts = []

    # Current role — strong positional signal
    title = p.get("current_title", "")
    company = p.get("current_company", "")
    if title:
        parts.append(f"Current Role: {title}" + (f" at {company}" if company else ""))

    # Summary — candidate's own words about their work
    summary = (p.get("summary") or "").strip()
    if summary:
        parts.append(f"Summary:\n{summary}")

    # Career history — the most substantive signal
    jobs = cand.get("career_history") or []
    if jobs:
        parts.append("Career:")
        for j in jobs:
            if not j:
                continue
            job_title = j.get("title", "")
            job_company = j.get("company", "")
            desc = (j.get("description") or "").strip()
            header = f"{job_title}" + (f" at {job_company}" if job_company else "")
            parts.append(header)
            if desc:
                parts.append(desc)

    # Skills — high-signal vocabulary (FAISS, NDCG, Pinecone, etc.)
    skills = cand.get("skills") or []
    if skills:
        skill_names = [s.get("name", "") for s in skills if s and s.get("name")]
        # Weight higher-proficiency skills by repeating them
        # This ensures "FAISS (expert, 60mo)" contributes more than "Excel (beginner)"
        weighted = []
        for s in skills:
            if not s:
                continue
            name = s.get("name", "")
            prof = s.get("proficiency", "")
            if not name:
                continue
            if prof == "expert":
                weighted.extend([name, name, name])
            elif prof == "advanced":
                weighted.extend([name, name])
            else:
                weighted.append(name)
        if weighted:
            parts.append("Skills: " + ", ".join(weighted))

    return "\n\n".join(parts)


def iter_candidates(path: str):
    """Stream candidates one at a time from the JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed line {line_num}: {e}",
                      file=sys.stderr)
                continue


def main():
    parser = argparse.ArgumentParser(description="Precompute candidate embeddings")
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl")
    parser.add_argument("--output_dir", default="./artifacts",
                        help="Directory to save .npy files (default: ./artifacts)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Encoding batch size (default: 256, reduce if OOM)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence transformer model name")
    parser.add_argument("--sample", type=int, default=None,
                        help="Only embed first N candidates (for testing)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Embed JD first
    print("Embedding job description...")
    jd_embedding = model.encode([JD_TEXT], show_progress_bar=False,
                                 normalize_embeddings=True)
    jd_path = os.path.join(args.output_dir, "jd_embedding.npy")
    np.save(jd_path, jd_embedding.astype(np.float32))
    print(f"JD embedding saved: {jd_path} shape={jd_embedding.shape}")

    # Stream and build candidate texts
    print(f"\nReading candidates from: {args.candidates}")
    t0 = time.time()
    candidate_ids = []
    candidate_texts = []
    for i, cand in enumerate(iter_candidates(args.candidates)):
        candidate_ids.append(cand.get("candidate_id", f"CAND_{i}"))
        candidate_texts.append(build_candidate_text(cand))
        if args.sample and i + 1 >= args.sample:
            print(f"[Sample mode] Stopping at {args.sample} candidates")
            break
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(candidate_texts) > 0 and args.sample is None and
                         100000 - i - 1 > 0)
            eta = (100000 - i - 1) / rate if remaining else 0
            print(f"  Read {i+1:,} candidates ({rate:.0f}/sec, "
                  f"ETA: {eta:.0f}s)")

    total = len(candidate_ids)
    print(f"\nRead {total:,} candidates in {time.time()-t0:.1f}s")

    # Encode in batches
    print(f"\nEncoding {total:,} candidates "
          f"(batch_size={args.batch_size})...")
    print("This is the slow step. Estimated time:")
    print("  CPU-only: 10-20 minutes for 100K candidates")
    print("  GPU:      2-3 minutes")
    print()

    t0 = time.time()
    embeddings = model.encode(
        candidate_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize for cosine via dot product
        convert_to_numpy=True,
    )

    elapsed = time.time() - t0
    print(f"\nEncoding done in {elapsed:.1f}s "
          f"({total/elapsed:.0f} candidates/sec)")

    # Save candidate IDs and embeddings
    ids_path = os.path.join(args.output_dir, "candidate_ids.npy")
    emb_path = os.path.join(args.output_dir, "candidate_embeddings.npy")

    np.save(ids_path, np.array(candidate_ids))
    np.save(emb_path, embeddings.astype(np.float32))

    size_mb = embeddings.nbytes / 1024 / 1024
    print(f"\nSaved:")
    print(f"  {ids_path}  shape={np.array(candidate_ids).shape}")
    print(f"  {emb_path}  shape={embeddings.shape}  size={size_mb:.1f}MB")
    print(f"  {jd_path}   shape={jd_embedding.shape}")

    # Quick sanity check
    print("\nSanity check — top 5 candidates by JD similarity:")
    sims = embeddings @ jd_embedding[0]  # dot product = cosine (both normalized)
    top5_idx = np.argsort(sims)[-5:][::-1]
    for idx in top5_idx:
        print(f"  {candidate_ids[idx]}: similarity={sims[idx]:.4f}")

    print("\nDone. Upload these 3 files to Claude:")
    print(f"  {ids_path}")
    print(f"  {emb_path}")
    print(f"  {jd_path}")


if __name__ == "__main__":
    main()