"""Deployable numeric features: IDs, gold, hidden state and future are absent."""

from __future__ import annotations
from ..types import PublicContext, Candidate, STATE_KEYS
from ..retrieval.retriever import tokenize
from ..utils.core import digest

FEATURES = (
    "goal_overlap",
    "semantic_similarity",
    "semantic_available",
    "tool_overlap",
    "precondition_match",
    "precondition_conflict",
    "unknown_fraction",
    "precondition_coverage",
    "schema_compatible",
    "historical_success",
    "normalized_token_ratio",
    "source_reliability",
)
FEATURE_SCHEMA = digest({"version": 1, "features": FEATURES})


def extract(context: PublicContext, candidate: Candidate, budget: int):
    exp = candidate.experience
    a, b = set(tokenize(context.goal)), set(tokenize(exp.goal))
    old_tools = {c["tool"] for c in exp.tool_calls}
    match = conflict = unknown = 0
    evidence = []
    for key, required in exp.preconditions.items():
        if key not in STATE_KEYS or type(required) is not bool:
            raise ValueError("experience has non-public preconditions")
        current = context.visible_state.get(key)
        status = (
            "unknown"
            if current is None
            else "match"
            if current == required
            else "conflict"
        )
        match += status == "match"
        conflict += status == "conflict"
        unknown += status == "unknown"
        evidence.append(
            {"key": key, "required": required, "observed": current, "status": status}
        )
    count = max(len(exp.preconditions), 1)
    values = (
        len(a & b) / max(len(a | b), 1),
        candidate.semantic_score or 0.0,
        float(candidate.semantic_score is not None),
        len(set(context.tools) & old_tools) / max(len(old_tools), 1),
        match / count,
        conflict / count,
        unknown / count if exp.preconditions else 1.0,
        len(exp.preconditions) / len(STATE_KEYS),
        float(context.schema_hash == exp.schema_hash),
        float(exp.feedback.get("success") is True),
        candidate.normalized_tokens / max(budget, 1),
        float(exp.reliability),
    )
    return dict(zip(FEATURES, map(float, values))), evidence
