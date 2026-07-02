"""
Streaming reader for candidates.jsonl.

The file is ~487MB / 100K lines. We never load it all into memory.
Every function here either streams (generator) or processes in a single pass.
"""
import json
import sys
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, Callable, List


def iter_candidates(path: str) -> Iterator[Dict[str, Any]]:
    """
    Stream candidates one at a time from the JSONL file.
    Use this for ANY full-dataset pass — never json.load() the whole file.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                # AUDIT NOTE: routed to stderr, not stdout. If the Stage 3
                # sandboxed scoring run prints progress/results to stdout,
                # a malformed-line warning on stdout could interleave with
                # and corrupt machine-parseable output. Verified the real
                # candidates.jsonl (100,000 lines) produces zero warnings,
                # so this path is untriggered in practice — fixed
                # defensively in case the held-out evaluation file differs.
                print(f"[WARN] Skipping malformed line {line_num}: {e}", file=sys.stderr)
                continue


def count_candidates(path: str) -> int:
    """Quick count without parsing JSON (fast)."""
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def sample_first_n(path: str, n: int = 5) -> List[Dict[str, Any]]:
    """Grab the first N candidates as-is, no filtering."""
    out = []
    for i, cand in enumerate(iter_candidates(path)):
        if i >= n:
            break
        out.append(cand)
    return out


def filter_candidates(
    path: str,
    predicate: Callable[[Dict[str, Any]], bool],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Single streaming pass, keep candidates where predicate(candidate) is True.
    Use this for all the 'pull extremes / suspicious profiles' sampling today.
    """
    out = []
    for cand in iter_candidates(path):
        if predicate(cand):
            out.append(cand)
            if limit is not None and len(out) >= limit:
                break
    return out


def get_by_id(path: str, candidate_id: str) -> Optional[Dict[str, Any]]:
    """
    Find one candidate by ID (streams until found, then stops).

    *** PERFORMANCE WARNING (found via audit, empirically measured) ***
    This is an O(n) LINEAR SCAN from the start of the file every time it's
    called. Measured cost on the real 100K/487MB dataset:
      - lookup near start of file:  ~0.002s
      - lookup near end of file:    ~3.8s
    A single call is fine. Calling this in a loop (e.g. "re-fetch full
    records for the top 100 after an initial scoring pass") is NOT fine —
    100 worst-case calls would cost ~380s, alone exceeding the hard 300s
    Stage 3 reproduction limit. Confirmed by an actual timeout when this
    was tested in a 100x loop during the audit.

    If you need multiple candidates by ID, use get_many_by_id() below
    (single streaming pass, O(n) total regardless of how many IDs requested)
    instead of calling get_by_id() repeatedly.
    """
    for cand in iter_candidates(path):
        if cand.get("candidate_id") == candidate_id:
            return cand
    return None


def get_many_by_id(path: str, candidate_ids) -> Dict[str, Dict[str, Any]]:
    """
    Fetch MULTIPLE candidates by ID in a single streaming pass — O(n) total
    regardless of how many IDs are requested, unlike calling get_by_id() in
    a loop (which is O(n*k) and was measured to risk exceeding the 300s
    hard limit for as few as ~100 lookups near the end of the file).

    Use this whenever you need more than one candidate by ID — e.g. Day 3's
    "re-fetch full records for the top 100 to generate reasoning text" step.
    """
    wanted = set(candidate_ids)
    found = {}
    if not wanted:
        return found
    for cand in iter_candidates(path):
        cid = cand.get("candidate_id")
        if cid in wanted:
            found[cid] = cand
            if len(found) == len(wanted):
                break  # stop early once everything requested has been found
    return found


if __name__ == "__main__":
    # Quick sanity check
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "../data/candidates.jsonl"
    print(f"Counting candidates in {data_path}...")
    n = count_candidates(data_path)
    print(f"Total candidates: {n}")
    print("\nFirst candidate:")
    first = sample_first_n(data_path, 1)[0]
    print(json.dumps(first, indent=2)[:2000])
