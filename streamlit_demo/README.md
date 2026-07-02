# Candidate Ranking Sandbox Demo

Runs the real production scoring pipeline (`src/scoring/scorer.py`) against
a sample of 55 candidates — the official hackathon-provided 50-candidate
sample, plus 5 reference candidates used during development.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Connect the repo, set main file to `app.py`
4. Deploy — no additional configuration needed (no secrets, no external
   API keys; the demo uses only the bundled `demo_candidates.json` and
   precomputed embeddings in `artifacts/`)

## Note on embeddings

`artifacts/` contains precomputed semantic embeddings (~147MB) used by
the full submission's semantic reranking stage. For this small 55-candidate
demo, semantic reranking only activates for the top-500 candidates in the
FULL ranking — since this demo has only 55 candidates total, all of them
are within that pool, so semantic blending is active here too if the
artifacts are present. If `artifacts/` is omitted (e.g. to keep the repo
small), the pipeline degrades gracefully to rule-only scoring automatically
— no code changes needed.
