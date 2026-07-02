# Dockerfile for Stage 3 reproducibility test.
#
# Builds a minimal container matching the JD's stated constraints:
#   - CPU-only (no GPU dependencies installed)
#   - No network access at runtime (--network=none enforces this)
#   - Should complete in < 5 minutes
#   - Should stay under 16GB RAM
#
# IMPORTANT: this Dockerfile must be built and run on a machine with
# Docker installed. It cannot be tested inside this sandbox (Docker is
# not available here). Run the commands at the bottom of this file on
# your local machine.

FROM python:3.11-slim

WORKDIR /app

# Only numpy is needed at RUNTIME — confirmed via import audit that the
# ranking pipeline never imports torch/sentence_transformers/transformers.
# Those heavy libraries are only used in the OFFLINE precompute step,
# which already ran on Kalpesh's machine and produced the .npy artifacts.
RUN pip install --no-cache-dir numpy==2.4.1

# Copy only what the ranking step needs — not the precompute script,
# not torch, not the labeling tools.
COPY src/ /app/src/
COPY artifacts/ /app/artifacts/
COPY data/candidates.jsonl /app/data/candidates.jsonl

WORKDIR /app/src

# No network access needed at build OR run time after this point.
# The container should be run with: docker run --network=none

ENTRYPOINT ["python3", "scoring/write_submission.py"]

# ============================================================
# BUILD AND RUN INSTRUCTIONS (run on a machine with Docker):
# ============================================================
#
# 1. Build the image:
#    docker build -t redrob-ranker -f docker/Dockerfile .
#
# 2. Run with the EXACT Stage 3 constraints enforced by Docker itself,
#    not just measured after the fact:
#    docker run --rm \
#      --network=none \
#      --memory=16g \
#      --cpus=4 \
#      -v "$(pwd)/output:/app/output" \
#      redrob-ranker
#
# 3. Time it externally (Docker's internal timing isn't enough —
#    measure wall clock from outside):
#    time docker run --rm --network=none --memory=16g --cpus=4 \
#      -v "$(pwd)/output:/app/output" redrob-ranker
#
# 4. If it fails with "no internet" errors, that means something is
#    STILL trying to import torch/sentence_transformers at runtime —
#    go back and check the import chain (see import_audit.py).
#
# 5. If it fails with OOM (out of memory), reduce --memory and re-test
#    to find the actual peak, then confirm it's safely under 16g.
