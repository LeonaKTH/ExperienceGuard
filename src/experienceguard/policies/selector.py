"""One shared candidate pool, one historical experience or an explicit rejection."""

from __future__ import annotations
import math
import numpy as np
from ..types import Decision, UnavailablePolicy
from ..features.state import FEATURES, FEATURE_SCHEMA, extract
from ..utils.core import digest, read_json


class PolicyModel:
    def __init__(self, path, *, allow_synthetic=False, expected_schema=FEATURE_SCHEMA):
        value = read_json(path)
        checksum = value.pop("artifact_sha256", None)
        if (
            checksum != digest(value)
            or value.get("version") != 1
            or value.get("feature_schema") != expected_schema
        ):
            raise ValueError("policy hash/version/feature schema mismatch")
        self.artifact_sha = checksum
        if value.get("status") != "REAL_LABELS_TRAINED" and not (
            allow_synthetic and value.get("status") == "SYNTHETIC_ONLY"
        ):
            raise UnavailablePolicy("UNTRAINED: real validated policy weights required")
        if sorted(value["classes"]) != ["harm", "help", "tie"]:
            raise ValueError("policy must contain help/harm/tie classes")
        self.value = value
        self.mean, self.scale = np.array(value["mean"]), np.array(value["scale"])
        self.coef, self.intercept = (
            np.array(value["coef"]),
            np.array(value["intercept"]),
        )
        n = len(FEATURES)
        if (
            self.mean.shape != (n,)
            or self.scale.shape != (n,)
            or self.coef.shape != (3, n)
            or self.intercept.shape != (3,)
        ):
            raise ValueError("invalid model dimensions")
        if not all(
            np.isfinite(x).all()
            for x in (self.mean, self.scale, self.coef, self.intercept)
        ) or np.any(self.scale <= 0):
            raise ValueError("invalid numeric model parameters")
        if not 0.1 <= value["temperature"] <= 10 or any(
            not math.isfinite(value[k]) or value[k] < 0
            for k in ("rho", "lambda_cost", "threshold")
        ):
            raise ValueError("invalid decision parameters")

    def predict(self, features):
        x = np.array([features[k] for k in FEATURES], dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("non-finite features")
        z = (self.coef @ ((x - self.mean) / self.scale) + self.intercept) / self.value[
            "temperature"
        ]
        p = np.exp(z - z.max())
        p /= p.sum()
        return dict(zip(self.value["classes"], map(float, p)))


class ExperienceSelector:
    version = "state-reuse-v1"

    def __init__(
        self,
        strategy,
        *,
        model=None,
        global_utility=None,
        feature_extractor=extract,
        rule_eligibility=None,
    ):
        self.feature_extractor = feature_extractor
        self.rule_eligibility = rule_eligibility or (lambda candidate: True)
        if strategy not in {"B0", "B1", "B2", "B3", "B4", "B5", "New"}:
            raise ValueError("unknown strategy")
        self.strategy, self.model, self.global_utility = strategy, model, global_utility

    def choose(self, context, candidates, budget):
        name = self.strategy
        representation = "raw" if name == "B1" else "normalized"
        evidence, valid = [], []
        for c in candidates:
            f, ev = self.feature_extractor(context, c, budget)
            tokens = c.raw_tokens if representation == "raw" else c.normalized_tokens
            evidence.append(
                {
                    "experience_id": c.experience.experience_id,
                    "source_task": c.experience.source_task,
                    "source_family": c.experience.source_family,
                    "features": f,
                    "preconditions": ev,
                    "within_budget": tokens <= budget,
                }
            )
            if tokens <= budget:
                valid.append((c, f, tokens))

        def decision(item=None, reason="no eligible candidate", scores=None):
            c, f, tokens = item if item else (None, {}, 0)
            return Decision(
                name,
                self.version,
                c.experience.experience_id if c else None,
                reason,
                f,
                scores or {},
                tokens,
                evidence,
                None if c else reason,
                representation,
            )

        if name == "B0":
            return decision(reason="no-memory control")
        if name == "New" and self.model is None:
            raise UnavailablePolicy(
                "UNTRAINED: New requires validated learned artifact"
            )
        if name == "B5" and not self.global_utility:
            raise UnavailablePolicy(
                "UNAVAILABLE: B5 requires real training-only utility records"
            )
        if name == "B1":
            if candidates and any(c.semantic_score is None for c in candidates):
                raise UnavailablePolicy(
                    "B1 requires a frozen local embedding; lexical scores are not semantic scores"
                )
            valid.sort(
                key=lambda item: (
                    -item[0].semantic_score,
                    item[0].experience.experience_id,
                )
            )
        if name == "B2":
            valid = [
                v for v in valid if v[0].experience.feedback.get("success") is True
            ]
        if name == "B4":
            valid = [
                v
                for v in valid
                if v[1]["schema_compatible"] == 1
                and v[1]["precondition_coverage"] > 0
                and v[1]["precondition_conflict"] == 0
                and v[1]["unknown_fraction"] == 0
                and self.rule_eligibility(v[0])
            ]
        if name == "B5":
            valid = [
                v
                for v in valid
                if self.global_utility.get(v[0].experience.experience_id, 0) > 0
            ]
            valid.sort(
                key=lambda v: (
                    -self.global_utility[v[0].experience.experience_id],
                    v[0].experience.experience_id,
                )
            )
        if name == "New":
            ranked = []
            for item in valid:
                c, f, tokens = item
                if f["schema_compatible"] != 1:
                    continue
                p = self.model.predict(f)
                utility = (
                    p["help"]
                    - self.model.value["rho"] * p["harm"]
                    - self.model.value["lambda_cost"] * tokens / max(budget, 1)
                )
                ranked.append(
                    (
                        utility,
                        c.experience.experience_id,
                        item,
                        {"p_" + k: v for k, v in p.items()} | {"utility": utility},
                    )
                )
            ranked.sort(key=lambda row: (-row[0], row[1]))
            if ranked and ranked[0][0] > self.model.value["threshold"]:
                return decision(
                    ranked[0][2],
                    "positive calibrated risk-adjusted utility",
                    ranked[0][3],
                )
            return decision(
                reason="all learned utilities below validated rejection threshold",
                scores=ranked[0][3] if ranked else {},
            )
        return (
            decision(valid[0], "eligible top-ranked historical experience")
            if valid
            else decision()
        )
