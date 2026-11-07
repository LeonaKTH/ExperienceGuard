"""Application workflow evaluation. Reports coverage/availability, never LLM accuracy."""

from pathlib import Path
from ..utils.core import write_json, digest, utc_now


def evaluate_diagnostics(
    app, inputs, output, strategies=("B0", "B1", "B2", "B3", "B4", "B5", "New")
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    frozen = digest([e.to_dict() for e in app.store.all()])
    for i, item in enumerate(inputs):
        for strategy in strategies:
            row = {"input_index": i, "strategy": strategy}
            try:
                r = app.diagnose(item["repository"], item["log"], strategy=strategy)
                row.update(
                    status=r["policy_status"],
                    case_id=r["case_id"],
                    selected=r["decision"]["experience_id"],
                    report_sha256=r["report_sha256"],
                    candidate_pool_sha=r["candidate_pool_sha"],
                    tool_calls=len(r["tool_trace"]),
                )
            except Exception as exc:
                row.update(
                    status="INFRASTRUCTURE_FAILURE",
                    error=type(exc).__name__ + ": " + str(exc),
                )
            rows.append(row)
            write_json(output / "rows.json", rows)
    if digest([e.to_dict() for e in app.store.all()]) != frozen:
        raise ValueError("evaluation unexpectedly wrote memory")
    result = {
        "assigned": len(inputs) * len(strategies),
        "completed_diagnoses": sum(r.get("case_id") is not None for r in rows),
        "unavailable_policies": sum(
            r["status"] in {"UNAVAILABLE", "UNTRAINED"} for r in rows
        ),
        "infrastructure_failures": sum(
            r["status"] == "INFRASTRUCTURE_FAILURE" for r in rows
        ),
        "frozen_memory_sha": frozen,
        "time": utc_now(),
        "interpretation": "Read-only evidence workflow coverage, NOT real LLM performance or algorithm effect",
        "rows": rows,
    }
    write_json(output / "summary.json", result)
    return result
