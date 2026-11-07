"""CPU-sized selector training, never LLM fine-tuning. JSON deployment, no pickle."""

from __future__ import annotations
import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from ..features.state import FEATURES, FEATURE_SCHEMA
from ..utils.core import digest, write_json, utc_now


def validate_rows(rows, allow_synthetic=False, domain=None):
    allowed_provenance = (
        {"verified_development_pair"}
        if domain == "development-ci-v1"
        else {"local_llm", "remote_llm"}
    )
    groups = {k: set() for k in ("fit", "calibration", "validation")}
    provenance, identities, source_families = set(), set(), set()
    repository_groups = {s: set() for s in groups}
    root_groups = {s: set() for s in groups}
    source_repositories = set()
    for row in rows:
        split = row.get("split")
        if split not in groups or row.get("label") not in {"help", "harm", "tie"}:
            raise ValueError(
                "only fit/calibration/validation help/harm/tie labels are legal"
            )
        groups[split].add(row["family"])
        provenance.add(row["provenance"])
        identities.add(row.get("experiment_identity"))
        source_families.update(row.get("source_families", []))
        if row.get("source_task") == row.get("task_id") or row["family"] in row.get(
            "source_families", []
        ):
            raise ValueError("source/target leakage")
        if set(row["features"]) != set(FEATURES) or not all(
            math.isfinite(float(x)) for x in row["features"].values()
        ):
            raise ValueError("invalid feature schema")
        if row["provenance"] != "synthetic_unit_test":
            if (
                row["provenance"] not in allowed_provenance
                or not row.get("pair_sha")
                or not row.get("execution_shas")
            ):
                raise ValueError("labels require real paired rollout provenance")
            checked = dict(row)
            checksum = checked.pop("pair_sha")
            if checksum != digest(checked):
                raise ValueError("paired label hash mismatch")
            if domain == "development-ci-v1" and (
                row.get("domain") != domain
                or row.get("input_provenance")
                not in {"user_repository", "public_repository"}
                or not row.get("protocol_approved")
                or row.get("agent_provider") != "remote_llm"
                or len(row.get("agent_execution_shas", [])) != 2
                or not row.get("repository_group")
                or not row.get("root_cause_group")
                or not row.get("source_repository_groups")
            ):
                raise ValueError(
                    "production development training requires reviewed real-repository remote-Agent pairs and repository/root-cause groups; manual-only or Mock feedback is insufficient"
                )
            if domain == "development-ci-v1":
                repository_groups[split].add(row["repository_group"])
                root_groups[split].add(row["root_cause_group"])
                source_repositories.update(row["source_repository_groups"])
    if len(identities) != 1 or None in identities:
        raise ValueError("labels must bind one fixed model/prompt/experiment identity")
    if not all(groups.values()) or any(
        groups[a] & groups[b]
        for a, b in (
            ("fit", "calibration"),
            ("fit", "validation"),
            ("calibration", "validation"),
        )
    ):
        raise ValueError(
            "independent nonempty task families required in all partitions"
        )
    if source_families & set.union(*groups.values()):
        raise ValueError("memory source families overlap target partitions")
    if domain == "development-ci-v1":
        for partition in (repository_groups, root_groups):
            if any(
                partition[a] & partition[b]
                for a, b in (
                    ("fit", "calibration"),
                    ("fit", "validation"),
                    ("calibration", "validation"),
                )
            ):
                raise ValueError("repository or root-cause group overlaps partitions")
        if source_repositories & set.union(*repository_groups.values()):
            raise ValueError("source memory repository overlaps target partitions")
    synthetic = provenance == {"synthetic_unit_test"}
    if (
        synthetic
        and not allow_synthetic
        or not synthetic
        and not provenance <= allowed_provenance
    ):
        raise ValueError("synthetic/mock labels cannot train a production policy")
    return synthetic


def train(
    rows,
    output,
    *,
    allow_synthetic=False,
    seed=42,
    c_grid=(0.1, 1.0, 10.0),
    feature_schema=FEATURE_SCHEMA,
    domain=None,
):
    synthetic = validate_rows(rows, allow_synthetic, domain)
    if domain and any(r.get("feature_schema") != feature_schema for r in rows):
        raise ValueError("development feature schema mismatch")
    parts = {
        s: [r for r in rows if r["split"] == s]
        for s in ("fit", "calibration", "validation")
    }

    def xy(s):
        return np.array(
            [[r["features"][f] for f in FEATURES] for r in parts[s]], dtype=float
        ), np.array([r["label"] for r in parts[s]])

    xf, yf = xy("fit")
    xc, yc = xy("calibration")
    xv, yv = xy("validation")
    if set(yf) != {"help", "harm", "tie"}:
        raise ValueError("fit partition needs all three classes")
    scaler = StandardScaler().fit(xf)
    xf, xc, xv = (scaler.transform(x) for x in (xf, xc, xv))
    choices = []
    for c in c_grid:
        if not math.isfinite(c) or c <= 0:
            raise ValueError("C must be finite and positive")
        model = LogisticRegression(
            C=c, penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed
        ).fit(xf, yf)
        logits = model.decision_function(xc)
        result = minimize_scalar(
            lambda t: log_loss(yc, softmax(logits / t, axis=1), labels=model.classes_),
            bounds=(0.1, 10.0),
            method="bounded",
        )
        temperature = float(result.x)
        probabilities = softmax(model.decision_function(xv) / temperature, axis=1)
        choices.append(
            (
                log_loss(yv, probabilities, labels=model.classes_),
                c,
                model,
                temperature,
                probabilities,
            )
        )
    _, chosen_c, model, temperature, pv = min(choices, key=lambda v: (v[0], v[1]))
    class_index = {v: i for i, v in enumerate(model.classes_)}
    # Validation alone chooses rejection/cost tradeoff; no test-tuned threshold.
    best = None
    for rho in (1.0, 2.0, 4.0):
        for cost in (0.0, 0.05, 0.1):
            for threshold in (0.0, 0.05, 0.1):
                utilities = (
                    pv[:, class_index["help"]]
                    - rho * pv[:, class_index["harm"]]
                    - cost
                    * np.array(
                        [
                            r["features"]["normalized_token_ratio"]
                            for r in parts["validation"]
                        ]
                    )
                )
                chosen = utilities > threshold
                realized = np.array(
                    [1.0 if y == "help" else -rho if y == "harm" else 0.0 for y in yv]
                )
                objective = float(
                    np.mean(
                        chosen
                        * (
                            realized
                            - cost
                            * np.array(
                                [
                                    r["features"]["normalized_token_ratio"]
                                    for r in parts["validation"]
                                ]
                            )
                        )
                    )
                )
                key = (-objective, rho, cost, threshold)
                if best is None or key < best[0]:
                    best = (key, rho, cost, threshold)
    artifact = {
        "version": 1,
        "status": "SYNTHETIC_ONLY" if synthetic else "REAL_LABELS_TRAINED",
        "feature_schema": feature_schema,
        "domain": domain,
        "classes": list(model.classes_),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "temperature": temperature,
        "rho": best[1],
        "lambda_cost": best[2],
        "threshold": best[3],
        "C": chosen_c,
        "training_data_sha": digest(rows),
        "partition_families": {
            s: sorted({r["family"] for r in rs}) for s, rs in parts.items()
        },
        "experiment_identity": rows[0]["experiment_identity"],
        "seed": seed,
        "created_at": utc_now(),
        "validation_log_loss": float(min(v[0] for v in choices)),
        "sklearn_version": __import__("sklearn").__version__,
    }
    artifact["artifact_sha256"] = digest(artifact)
    write_json(output, artifact)
    return artifact
