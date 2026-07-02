"""
Shared safe accessors for candidate dicts.

FOUND VIA PRODUCTION-READINESS AUDIT: every feature module independently
used the pattern `cand.get("profile", {})`. This pattern has a real bug —
.get(key, default) only supplies the default when the KEY IS ABSENT. If the
key is present but its value is explicitly None (which the JSON schema
technically permits, even though no candidate in the real 100K dataset
actually does this), .get() returns None, and any chained .get() call on
that result raises AttributeError.

Adversarial testing confirmed this crashes 3 of 6 feature modules on inputs
like {"profile": None, ...} or {"career_history": [None], ...}. The real
public dataset never triggers this (verified: 0 exceptions across all
100,000 candidates), but the held-out evaluation set is NOT under our
control, and a single malformed record causing an unhandled exception
mid-run is a real risk to the 5-minute reproduction step. This module
centralizes the fix so it's applied consistently rather than patched
ad-hoc in 21 separate call sites across 6 files (which is itself how the
bug arose in the first place — duplicated patterns drift).
"""
from typing import Dict, Any, List


def safe_profile(cand: Dict[str, Any]) -> Dict[str, Any]:
    return cand.get("profile") or {}


def safe_career_history(cand: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Also filters out any None entries within the list itself — found via
    adversarial testing that [None] as a career_history value crashes
    every downstream .get() call on individual job entries."""
    jobs = cand.get("career_history") or []
    return [j for j in jobs if j is not None]


def safe_skills(cand: Dict[str, Any]) -> List[Dict[str, Any]]:
    skills = cand.get("skills") or []
    return [s for s in skills if s is not None]


def safe_education(cand: Dict[str, Any]) -> List[Dict[str, Any]]:
    edu = cand.get("education") or []
    return [e for e in edu if e is not None]


def safe_redrob_signals(cand: Dict[str, Any]) -> Dict[str, Any]:
    return cand.get("redrob_signals") or {}
