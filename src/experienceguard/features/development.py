"""Version predicates and unknown-preserving comparison over parsed public evidence."""

from packaging.specifiers import SpecifierSet, InvalidSpecifier
from packaging.version import Version, InvalidVersion
from ..features.state import FEATURES
from ..retrieval.retriever import tokenize
from ..utils.core import digest
from ..domain.development import DOMAIN

FEATURE_SCHEMA = digest(
    {
        "domain": DOMAIN,
        "features": FEATURES,
        "predicate_version": 2,
        "source_quality": "user_attested_capped_0.55",
    }
)


def compare_conditions(facts, conditions):
    result = []
    for key, condition in sorted(conditions.items()):
        current = facts.get(key)
        op = condition["op"]
        expected = condition.get("value")
        status = "unknown"
        reason = "current or source evidence is missing"
        if op != "unknown" and current is not None and expected is not None:
            if op == "eq":
                matched = str(current) == str(expected)
            elif op == "version":
                try:
                    matched = Version(str(current)) in SpecifierSet(str(expected))
                except (InvalidVersion, InvalidSpecifier):
                    result.append(
                        {
                            "key": key,
                            "required": expected,
                            "observed": current,
                            "status": "unknown",
                            "reason": "invalid or unresolved version",
                            "source_evidence": condition.get("evidence_refs", []),
                        }
                    )
                    continue
            else:
                raise ValueError("unsupported condition operator")
            status = "match" if matched else "conflict"
            reason = (
                "public value satisfies predicate"
                if matched
                else "public value violates predicate"
            )
        result.append(
            {
                "key": key,
                "operator": op,
                "required": expected,
                "observed": current,
                "status": status,
                "reason": reason,
                "source_evidence": condition.get("evidence_refs", []),
            }
        )
    return result


def extract(context, candidate, budget):
    exp = candidate.experience
    evidence = compare_conditions(context.visible_state, exp.preconditions)
    a, b = set(tokenize(context.goal)), set(tokenize(exp.goal))
    n = max(len(evidence), 1)
    values = (
        len(a & b) / max(len(a | b), 1),
        candidate.semantic_score or 0.0,
        float(candidate.semantic_score is not None),
        len(set(context.tools) & {c["tool"] for c in exp.tool_calls})
        / max(len({c["tool"] for c in exp.tool_calls}), 1),
        sum(e["status"] == "match" for e in evidence) / n,
        sum(e["status"] == "conflict" for e in evidence) / n,
        sum(e["status"] == "unknown" for e in evidence) / n if evidence else 1.0,
        min(len(evidence) / 6, 1.0),
        float(context.schema_hash == exp.schema_hash),
        float(exp.feedback.get("success") is True),
        candidate.normalized_tokens / max(budget, 1),
        min(exp.reliability, 0.55)
        if exp.provenance == "user_attested"
        else exp.reliability,
    )
    return dict(zip(FEATURES, map(float, values))), evidence
