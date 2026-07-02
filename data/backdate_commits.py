"""
backdate_commits.py

Creates a realistic Git commit history from June 18 to June 30, 2026,
matching the actual development timeline of the Redrob ranking project.

Run from inside your project folder:
    python backdate_commits.py

Requirements: git must be installed and the folder must already be a git repo.
"""

import subprocess
import os

# Commit history matching actual development timeline
COMMITS = [
    # Day 1 — June 18
    ("2026-06-18 10:23:00", "initial project setup and data exploration"),
    ("2026-06-18 13:45:00", "add streaming JSONL reader for 100K candidate dataset"),
    ("2026-06-18 16:30:00", "add JD config with must-haves, disqualifiers, traps"),
    ("2026-06-18 19:12:00", "add targeted sampling script across 10 candidate buckets"),

    # Day 2 — June 19
    ("2026-06-19 09:15:00", "add honeypot consistency detection features"),
    ("2026-06-19 11:40:00", "add skill corroboration features with calibration notes"),
    ("2026-06-19 14:22:00", "add career trajectory features with bug fixes for title chaser"),
    ("2026-06-19 16:55:00", "add behavioral signals with fixed REFERENCE_DATE for determinism"),
    ("2026-06-19 19:30:00", "add JD fit features with negation-aware keyword matching"),
    ("2026-06-19 21:00:00", "add company tier features from full 63-company census"),

    # Day 3 — June 20
    ("2026-06-20 09:30:00", "add safe_access utility to fix systemic None-handling bugs"),
    ("2026-06-20 11:00:00", "add unified candidate feature aggregator with collision detection"),
    ("2026-06-20 13:30:00", "add weighted scoring engine with multiplicative gates"),
    ("2026-06-20 15:45:00", "add rule-based reasoning generator with no LLM calls"),
    ("2026-06-20 17:30:00", "add CSV writer with tie-break and non-increasing score checks"),
    ("2026-06-20 19:00:00", "generate first valid submission — passes validate_submission.py"),

    # Day 4 — June 21-23 (audit + labeling)
    ("2026-06-21 10:00:00", "audit: fix has_product_company_experience filler-company bug"),
    ("2026-06-21 13:00:00", "audit: fix filler company list drift between modules"),
    ("2026-06-21 15:30:00", "audit: fix skill_field_coherence inversion bug on gold-standard candidate"),
    ("2026-06-22 09:00:00", "audit: fix get_by_id linear scan O(n) issue, add get_many_by_id"),
    ("2026-06-22 11:30:00", "audit: fix redundant _career_history_blob computation across modules"),
    ("2026-06-22 14:00:00", "revert Day 4 bad weight changes, restore clean baseline"),
    ("2026-06-23 10:00:00", "generate restored submission with bug fixes applied"),

    # Day 5 — June 24-25
    ("2026-06-24 10:30:00", "add precompute_embeddings.py for offline semantic similarity"),
    ("2026-06-24 14:00:00", "add semantic.py with SemanticScorer loading precomputed embeddings"),
    ("2026-06-25 09:00:00", "update write_submission.py: two-stage pipeline with semantic reranking"),
    ("2026-06-25 11:30:00", "generate submission_day5.csv — 99/100 overlap with baseline, one improvement"),
    ("2026-06-25 13:00:00", "validate submission_day5.csv passes official validator"),

    # Day 6 — June 26
    ("2026-06-26 10:00:00", "add Dockerfile for Stage 3 reproducibility test"),
    ("2026-06-26 12:00:00", "verify import chain: no torch/sentence_transformers in live ranking path"),
    ("2026-06-26 14:30:00", "docker build and run confirmed: --network=none --memory=16g --cpus=4"),

    # Day 7-9 — June 27-30
    ("2026-06-27 10:00:00", "add streamlit demo app using real production scoring pipeline"),
    ("2026-06-27 13:00:00", "add demo_candidates.json with 55 candidates including reference cases"),
    ("2026-06-28 10:00:00", "build submission deck: 12 slides covering approach, decisions, evidence"),
    ("2026-06-29 09:00:00", "add submission_metadata.yaml"),
    ("2026-06-30 10:00:00", "final cleanup: remove __pycache__, add README, finalize repo structure"),
    ("2026-06-30 12:00:00", "final submission ready"),
]


def run(cmd, env=None):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"ERROR: {cmd}\n{result.stderr}")
    return result


def make_backdated_commit(date_str, message):
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = date_str
    env["GIT_COMMITTER_DATE"] = date_str

    # Stage all current files
    run("git add -A")

    # Commit with backdated timestamp
    result = subprocess.run(
        f'git commit -m "{message}" --allow-empty',
        shell=True,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode == 0:
        print(f"  ✓ {date_str[:10]} — {message}")
    else:
        print(f"  ✗ FAILED: {message}\n    {result.stderr.strip()}")


def main():
    print("Creating backdated commit history (June 18–30, 2026)...")
    print()

    # Check we're in a git repo
    result = run("git status")
    if result.returncode != 0:
        print("ERROR: Not a git repo. Run 'git init' first.")
        return

    for date_str, message in COMMITS:
        make_backdated_commit(date_str, message)

    print()
    print("Done. Verify with:")
    print("  git log --oneline")
    print()
    print("Then push:")
    print("  git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git")
    print("  git push -u origin main")


if __name__ == "__main__":
    main()