"""Small application pilot harness. No model is called without per-run authorization."""

from pathlib import Path
from ..utils.core import digest, write_json
from ..agents.development import prepare_agent, run_remote


def evaluate_agent(
    app,
    incidents,
    output,
    *,
    strategies=("B0", "B1", "B3", "B4", "B5", "New"),
    protocol_only=False,
    allow_remote=False,
    consent_shas=None,
    acknowledge_upload=False,
    confirm_cost=False,
):
    """Use preview-only first; a future paid run requires each prepared scope's consent.

    Existing source memory is frozen; target feedback is never promoted by this runner.
    All strategies keep the application hard gate, so ranking arms are safety-filtered.
    """
    output = Path(output).resolve()
    if any(
        output.is_relative_to(
            Path(app.load_case(item["case_id"])["repository"]["path"]).resolve()
        )
        for item in incidents
    ):
        raise ValueError("pilot output must stay outside imported repositories")
    output.mkdir(parents=True, exist_ok=False)
    if not strategies or len(set(strategies)) != len(strategies):
        raise ValueError("unique nonempty strategies required")
    frozen = tuple(app.store.all())
    original_store = app.store

    class Frozen:
        def all(self):
            return list(frozen)

        def get(self, key):
            return next(e for e in frozen if e.experience_id == key)

        def add(self, item):
            raise ValueError("pilot cannot promote target memory")

    app.store = Frozen()
    rows = []
    try:
        for item in incidents:
            initial = app.load_case(item["case_id"])
            log = (app._case_dir(item["case_id"]) / "raw.log").read_text(
                encoding="utf8"
            )
            shared = None
            for strategy in strategies:
                row = {
                    "input_case": item["case_id"],
                    "strategy": strategy,
                    "effective_policy": strategy
                    if strategy == "B0"
                    else "B4-safety-filtered:" + strategy,
                }
                try:
                    report = app.diagnose(
                        initial["repository"]["path"],
                        log,
                        strategy=strategy,
                        provenance=initial["input_provenance"],
                    )
                    if shared is None:
                        shared = report["candidate_pool_sha"]
                    if report["candidate_pool_sha"] != shared:
                        raise ValueError("candidate pool differs between arms")
                    if report["policy_status"] != "AVAILABLE":
                        row.update(status=report["policy_status"], requests=0)
                    else:
                        plan = prepare_agent(
                            app, report["case_id"], item.get("paths", [])
                        )
                        write_json(output / (plan["plan_id"] + ".json"), plan)
                        if protocol_only:
                            from ..agents.protocol_test import protocol_test

                            run = protocol_test(app, plan)
                        else:
                            run = run_remote(
                                app,
                                plan,
                                consent_sha=(consent_shas or {}).get(
                                    plan["plan_sha256"]
                                ),
                                allow_remote=allow_remote,
                                acknowledge_upload=acknowledge_upload,
                                confirm_cost=confirm_cost,
                            )
                        row.update(
                            status=run["status"],
                            case_id=report["case_id"],
                            run_id=run.get("run_id"),
                            plan_sha=plan["plan_sha256"],
                            candidate_pool_sha=shared,
                            metrics=run.get("metrics", {}),
                            requests=run.get("metrics", {}).get("requests", 0),
                            execution_sha=run.get("execution_sha"),
                            selected=run.get("selected_memory"),
                            mode=run.get("mode"),
                            real_llm_status=run.get("real_llm_status"),
                        )
                except Exception as exc:
                    row.update(
                        status="INFRASTRUCTURE_FAILURE",
                        error=type(exc).__name__ + ": " + str(exc),
                    )
                rows.append(row)
                write_json(output / "assignments.json", rows)
    finally:
        app.store = original_store
    if digest([e.to_dict() for e in original_store.all()]) != digest(
        [e.to_dict() for e in frozen]
    ):
        raise ValueError("pilot modified source memory")
    result = {
        "assigned": len(incidents) * len(strategies),
        "rows": rows,
        "source_pool_sha": digest([e.to_dict() for e in frozen]),
        "real_llm": "NOT VERIFIED"
        if protocol_only or not any(r.get("mode") == "remote_llm" for r in rows)
        else "EXECUTED, quality requires independent scoring",
        "New": "NOT TRAINED unless explicitly supplied validated model",
        "algorithm_gains": "NOT ESTABLISHED",
        "scope": "Tool protocol/availability and costs; no inferred task success labels",
    }
    write_json(output / "summary.json", result)
    return result


def execute_reviewed_plan_batch(
    app,
    prepared_plans,
    output,
    *,
    allow_remote=False,
    acknowledge_upload=False,
    confirm_cost=False,
):
    """Future pilot resumes exact plans already inspected, without regenerating consent IDs."""
    from ..agents.development import load_plan
    from ..agents.providers import RemoteBudgetLedger

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    ledger = RemoteBudgetLedger()
    for item in prepared_plans:
        plan = load_plan(app, item["case_id"], item["plan_id"])
        run = run_remote(
            app,
            plan,
            consent_sha=item["consent_sha"],
            allow_remote=allow_remote,
            acknowledge_upload=acknowledge_upload,
            confirm_cost=confirm_cost,
            ledger=ledger,
        )
        rows.append(run)
        write_json(output / "executions.json", rows)
    return {"assigned": len(prepared_plans), "executions": rows}
