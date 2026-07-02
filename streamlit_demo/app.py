"""
Streamlit sandbox demo — Intelligent Candidate Discovery & Ranking.

Runs the EXACT same scoring pipeline used to generate the actual submission
(src/scoring/scorer.py, src/scoring/write_submission.py) against a small
sample of 55 candidates: the official 50-candidate sample provided by the
hackathon, plus 5 known reference candidates used throughout development
(a verified strong fit, a noise profile, a honeypot, the current #1 ranked
candidate, and a weak scattered-skills candidate) — chosen specifically so
a reviewer can see the system correctly separate them.

This demo uses the REAL production code (not a simplified reimplementation)
so it faithfully represents what the submitted system does.
"""
import streamlit as st
import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from scoring.scorer import compute_final_score
from scoring.reasoning import generate_reasoning
from scoring.candidate_features import compute_all_features

st.set_page_config(
    page_title="Candidate Ranking — Redrob AI Challenge",
    page_icon="\U0001F50D",
    layout="wide",
)

DEMO_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_candidates.json")

KNOWN_REFERENCE_NOTES = {
    "CAND_0001610": "Verified strong fit — used as the gold-standard reference throughout development.",
    "CAND_0000100": "Verified noise profile — fictional filler companies, mismatched job descriptions.",
    "CAND_0003582": "Verified honeypot — expert-level skill proficiency claimed with 0 months experience.",
    "CAND_0046064": "Currently the #1 ranked candidate in the full 100,000-candidate submission.",
    "CAND_0000082": "Weak candidate — scattered unrelated skills not backed by career history.",
}


@st.cache_data
def load_candidates():
    with open(DEMO_DATA_PATH, "r") as f:
        return json.load(f)


def run_ranking(candidates):
    """Runs the real production scoring pipeline on the demo sample."""
    results = []
    for cand in candidates:
        r = compute_final_score(cand)
        reasoning = generate_reasoning(r["flat_features"], cand)
        p = cand.get("profile") or {}
        results.append({
            "candidate_id": r["candidate_id"],
            "score": round(r["final_score"], 4),
            "title": p.get("current_title", ""),
            "company": p.get("current_company", ""),
            "yoe": p.get("years_of_experience", ""),
            "must_have_corroborated": r["flat_features"].get("must_have_corroborated_count", 0),
            "is_honeypot": r["flat_features"].get("is_likely_honeypot", False),
            "reasoning": reasoning,
            "note": KNOWN_REFERENCE_NOTES.get(r["candidate_id"], ""),
        })
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    return results


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

st.title("Intelligent Candidate Discovery & Ranking")
st.caption(
    "Redrob AI Challenge — Hack2Skill INDIA RUNS. "
    "This sandbox runs the **real production scoring pipeline** "
    "(the same code used to generate the actual submission CSV) "
    "against a sample of 55 candidates."
)

with st.expander("About this sample", expanded=False):
    st.markdown(
        "- **50 candidates** from the official hackathon-provided sample set\n"
        "- **5 known reference candidates** used throughout development "
        "(a verified strong fit, a noise profile, a honeypot, the current "
        "#1-ranked candidate from the full submission, and a weak scattered-"
        "skills candidate) — included so you can see the system correctly "
        "separate them\n\n"
        "The full submission ranks all 100,000 candidates from the dataset; "
        "this demo runs the identical scoring code on a small subset so it "
        "completes in seconds, not minutes."
    )

candidates = load_candidates()
st.write(f"Loaded **{len(candidates)}** candidates.")

if st.button("Run ranking pipeline", type="primary"):
    t0 = time.time()
    with st.spinner("Scoring candidates..."):
        results = run_ranking(candidates)
    elapsed = time.time() - t0

    st.success(f"Ranked {len(results)} candidates in {elapsed:.2f}s")

    col1, col2, col3 = st.columns(3)
    col1.metric("Candidates scored", len(results))
    col2.metric("Honeypots detected", sum(1 for r in results if r["is_honeypot"]))
    col3.metric("Runtime", f"{elapsed:.2f}s")

    st.divider()
    st.subheader("Ranked results")

    for i, r in enumerate(results, 1):
        honeypot_flag = " \u26A0\uFE0F HONEYPOT" if r["is_honeypot"] else ""
        with st.container(border=True):
            cols = st.columns([0.6, 4, 1.5, 1.5])
            cols[0].markdown(f"**#{i}**")
            cols[1].markdown(
                f"**{r['title']}** @ {r['company']}  \n"
                f"`{r['candidate_id']}` · YOE: {r['yoe']}{honeypot_flag}"
            )
            cols[2].metric("Score", r["score"])
            cols[3].metric("Must-haves", f"{r['must_have_corroborated']}/4")

            if r["note"]:
                st.info(r["note"])

            st.caption(r["reasoning"])
else:
    st.info("Click **Run ranking pipeline** to score the sample candidates using the real production scoring system.")

st.divider()
st.caption(
    "Source: github.com/[repo-link] · "
    "Full submission ranks 100,000 candidates in ~93s, validated under "
    "Docker constraints (no network, 16GB RAM cap, CPU-only) in under 2 minutes."
)
