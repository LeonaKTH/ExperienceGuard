"""Explicit paired human/controlled verification export; never infer utility from one success."""

import re
from ..utils.core import read_json, digest
from ..domain.development import DOMAIN
from ..features.development import FEATURE_SCHEMA


def global_utilities(rows):
    totals = {}
    identities = set()
    for row in rows:
        checked = dict(row)
        sha = checked.pop("pair_sha", None)
        if (
            sha != digest(checked)
            or row.get("domain") != DOMAIN
            or row.get("feature_schema") != FEATURE_SCHEMA
            or row.get("split") != "fit"
            or row.get("provenance") != "verified_development_pair"
            or row.get("input_provenance")
            not in {"user_repository", "public_repository"}
            or not row.get("protocol_approved")
            or row.get("agent_provider") != "remote_llm"
            or len(row.get("agent_execution_shas", [])) != 2
            or row.get("label") not in {"help", "harm", "tie"}
        ):
            raise ValueError(
                "B5 development utility requires reviewed, real-repository fit pairs"
            )
        identities.add(row["experiment_identity"])
        totals.setdefault(row["experience_id"], []).append(
            {"help": 1, "harm": -1, "tie": 0}[row["label"]]
        )
    if len(identities) > 1:
        raise ValueError("mixed development experiment identities")
    return {key: sum(values) / len(values) for key, values in totals.items()}


def load_receipt(app, case_id, receipt_id):
    if not re.fullmatch(r"verification-[a-f0-9]{32}", receipt_id):
        raise ValueError("invalid receipt ID")
    value = read_json(app._case_dir(case_id) / (receipt_id + ".json"))
    check = dict(value)
    sha = check.pop("receipt_sha256", None)
    if sha != digest(check) or value["case_id"] != case_id:
        raise ValueError("receipt hash/identity mismatch")
    return value


def build_pair(
    app,
    control_id,
    memory_id,
    control_receipt,
    memory_receipt,
    *,
    split,
    family,
    protocol_approved=False,
    repository_group=None,
    root_cause_group=None,
    source_repository_groups=(),
):
    if (
        control_id == memory_id
        or split not in {"fit", "calibration", "validation"}
        or not family.strip()
    ):
        raise ValueError(
            "independent paired reports, a legal split and semantic family are required"
        )
    a, b = app.load_case(control_id), app.load_case(memory_id)
    if a.get("implementation") != b.get("implementation") or not a.get(
        "implementation", {}
    ).get("source_tree_sha"):
        raise ValueError("pair diagnosis implementation versions differ or are unbound")
    if (
        a["decision"]["strategy"] != "B0"
        or a["decision"]["experience_id"]
        or not b["decision"]["experience_id"]
    ):
        raise ValueError(
            "pair must compare explicit no-memory and used-memory diagnoses"
        )
    if any(
        a[k] != b[k]
        for k in (
            "log_sha256",
            "inspected_snapshot_sha",
            "facts",
            "candidate_pool_sha",
            "mode",
        )
    ):
        raise ValueError("pair inputs/model mode/frozen candidate pool differ")
    va, vb = (
        load_receipt(app, control_id, control_receipt),
        load_receipt(app, memory_id, memory_receipt),
    )
    if va["outcome"] == "inconclusive" or vb["outcome"] == "inconclusive":
        raise ValueError("inconclusive verification has no utility label")
    selected = app.store.get(b["decision"]["experience_id"])
    if selected.source_task in {control_id, memory_id}:
        raise ValueError("future/self memory leakage")
    source_ids = {v["source_incident"] for v in b["candidates"]}
    if source_ids & {control_id, memory_id}:
        raise ValueError("pair used current target results as memory")

    def success(v):
        return v["outcome"] == "success"

    label = (
        "help"
        if success(vb) and not success(va)
        else "harm"
        if success(va) and not success(vb)
        else "tie"
    )
    from ..features.development import extract

    agent_identity = None
    agent_shas = []
    if va.get("agent_run_id") or vb.get("agent_run_id"):
        from ..agents.development import load_execution, load_plan

        if not va.get("agent_run_id") or not vb.get("agent_run_id"):
            raise ValueError("both paired receipts must bind Agent executions")
        ra = load_execution(app, control_id, va["agent_run_id"])
        rb = load_execution(app, memory_id, vb["agent_run_id"])
        if any(
            r["mode"] != "remote_llm" or r["status"] != "completed" for r in (ra, rb)
        ):
            raise ValueError(
                "Mock or incomplete Agent executions cannot produce production utility labels"
            )
        fields = (
            "model",
            "prompt_sha",
            "tool_schema_sha",
            "seed",
            "limits",
            "implementation",
            "agent_implementation_sha",
            "transport_settings",
            "candidate_pool_sha",
        )
        if (
            any(ra[k] != rb[k] for k in fields)
            or ra["selected_memory"] is not None
            or rb["selected_memory"] != selected.experience_id
        ):
            raise ValueError(
                "Agent pair identity or actual memory intervention differs"
            )
        if (
            va.get("agent_execution_sha") != ra["execution_sha"]
            or vb.get("agent_execution_sha") != rb["execution_sha"]
        ):
            raise ValueError("verification/Agent execution binding mismatch")
        plans = {}
        for cid in (control_id, memory_id):
            for path in app._case_dir(cid).glob("plan-*.json"):
                p = load_plan(app, cid, path.stem)
                plans[p["plan_sha256"]] = p
        if (
            plans[ra["plan_sha256"]]["documents"]
            != plans[rb["plan_sha256"]]["documents"]
        ):
            raise ValueError("paired Agent disclosure scopes differ")
        agent_identity = {k: rb[k] for k in fields if k != "candidate_pool_sha"}
        agent_shas = [ra["execution_sha"], rb["execution_sha"]]
    from ..types import Candidate
    from ..domain.development import DevelopmentContext, DevelopmentNormalizer
    from ..utils.core import token_upper

    norm = DevelopmentNormalizer()
    candidate = Candidate(
        selected,
        0,
        raw_tokens=token_upper(norm.render(selected, "raw")),
        normalized_tokens=token_upper(norm.render(selected)),
    )
    features, _ = extract(DevelopmentContext(**b["visible_context"]), candidate, 24000)
    row = {
        "domain": DOMAIN,
        "feature_schema": FEATURE_SCHEMA,
        "split": split,
        "family": family,
        "task_id": memory_id,
        "source_task": selected.source_task,
        "source_families": [selected.source_family],
        "experience_id": selected.experience_id,
        "features": features,
        "label": label,
        "provenance": "verified_development_pair",
        "input_provenance": b["input_provenance"],
        "protocol_approved": bool(protocol_approved),
        "experiment_identity": digest(
            {
                "mode": b["mode"],
                "domain": DOMAIN,
                "source_pool": b.get("source_pool_sha", b["candidate_pool_sha"]),
                "implementation": b["implementation"],
                "policy_artifact_sha": b.get("policy_artifact_sha"),
                "agent": agent_identity,
            }
        ),
        "execution_shas": [va["receipt_sha256"], vb["receipt_sha256"]],
        "control_report_sha": a["report_sha256"],
        "memory_report_sha": b["report_sha256"],
        "label_basis": "paired recorded verification; human action differences require protocol review, not automatically causal Agent utility",
        "agent_provider": "remote_llm" if agent_identity else None,
        "agent_execution_shas": agent_shas,
        "repository_group": repository_group,
        "root_cause_group": root_cause_group,
        "source_repository_groups": list(source_repository_groups),
        "control_receipt": control_receipt,
        "memory_receipt": memory_receipt,
    }
    row["pair_sha"] = digest(row)
    return row
