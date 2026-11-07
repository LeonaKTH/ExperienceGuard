"""Static public failure audit: seal all predictions, then and only then read gold.

This measures deterministic evidence handling, not build repair or LLM accuracy.
"""

from pathlib import Path
import time
from ..utils.core import read_json, write_json, digest, file_hash, utc_now
from ..workflows.service import DevelopmentApp


def bound(root, relative):
    target = (root / relative).resolve(strict=True)
    if not target.is_relative_to(root.resolve()) or target.is_symlink():
        raise ValueError("dataset path escapes its declared partition")
    return target


def tree(root):
    return {
        p.relative_to(root).as_posix(): file_hash(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def public_input(dataset, case_id):
    dataset = Path(dataset).resolve()
    index = read_json(dataset / "cases.json")
    item = next((c for c in index["cases"] if c["case_id"] == case_id), None)
    if item is None:
        raise ValueError("unknown public case")
    path = bound(
        dataset / "inputs", str(Path(item["input_manifest"]).relative_to("inputs"))
    )
    if file_hash(path) != item["input_manifest_sha256"]:
        raise ValueError("public input manifest changed")
    data = read_json(path)
    for rel, sha in data["files"].items():
        if file_hash(bound(path.parent, rel)) != sha:
            raise ValueError("public input identity mismatch")
    return {
        "case_id": case_id,
        "repository": str(bound(path.parent, data["repository"])),
        "log": bound(path.parent, data["log"]).read_text(encoding="utf8"),
        "source_url": item["issue_url"],
        "scope": data["scope"],
    }


def evaluate_public_cases(dataset, output):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.is_relative_to(dataset) or output.exists():
        raise ValueError("use a new evaluation directory outside the immutable dataset")
    index = read_json(dataset / "cases.json")
    output.mkdir(parents=True)
    app = DevelopmentApp(output / "application", allowed_roots=[dataset / "inputs"])
    before = tree(dataset / "inputs")
    predictions = []
    started = time.perf_counter()
    for item in index["cases"]:
        manifest_path = bound(
            dataset / "inputs", str(Path(item["input_manifest"]).relative_to("inputs"))
        )
        if file_hash(manifest_path) != item["input_manifest_sha256"]:
            raise ValueError("input manifest identity mismatch")
        manifest = read_json(manifest_path)
        for rel, sha in manifest["files"].items():
            if file_hash(bound(manifest_path.parent, rel)) != sha:
                raise ValueError("pre-repair input has changed")
        entry = {"case_id": item["case_id"]}
        try:
            report = app.diagnose(
                bound(manifest_path.parent, manifest["repository"]),
                bound(manifest_path.parent, manifest["log"]).read_text(encoding="utf8"),
                strategy="B0",
                provenance="public_repository",
                public_url=item["issue_url"],
            )
            entry.update(
                status="sealed", report=report, report_sha=report["report_sha256"]
            )
        except Exception as exc:
            entry.update(
                status="infrastructure_failure",
                error=type(exc).__name__ + ": " + str(exc),
            )
        predictions.append(entry)
    # No reference contents have been opened above this line.
    write_json(output / "sealed_predictions.json", predictions)
    seal = {
        "predictions_file_sha256": file_hash(output / "sealed_predictions.json"),
        "sealed_at": utc_now(),
        "count": len(predictions),
    }
    write_json(output / "seal.json", seal)
    if tree(dataset / "inputs") != before or app.store.all():
        raise ValueError("evaluation modified input or promoted current-target memory")
    # The scoring phase verifies the durable seal before loading reference answers.
    if file_hash(output / "sealed_predictions.json") != seal["predictions_file_sha256"]:
        raise ValueError("prediction seal changed before scoring")
    rows = []
    for item, prediction in zip(index["cases"], predictions):
        reference_path = bound(
            dataset / "reference", str(Path(item["reference"]).relative_to("reference"))
        )
        if file_hash(reference_path) != item["reference_sha256"]:
            raise ValueError("reference identity mismatch")
        reference = read_json(reference_path)
        report = prediction.get("report", {})
        facts = report.get("facts", {})
        known = {k: v for k, v in reference["expected_facts"].items() if v is not None}
        unknown = [k for k, v in reference["expected_facts"].items() if v is None]
        ev = {e["id"]: e for e in report.get("evidence", [])}
        refs = [
            r for s in report.get("suggestions", []) for r in s.get("evidence_refs", [])
        ]
        located = {e["path"] for e in ev.values() if e["kind"] == "source_location"}
        row = {
            "case_id": item["case_id"],
            "repository": item["repository"],
            "evidence_grade": reference["evidence_grade"],
            "expected_kind": reference["expected_kind"],
            "predicted_kind": report.get("primary_issue", {}).get("kind"),
            "type_correct": report.get("primary_issue", {}).get("kind")
            == reference["expected_kind"],
            "known_facts_correct": sum(facts.get(k) == v for k, v in known.items()),
            "known_facts_total": len(known),
            "unknown_retained": sum(facts.get(k) is None for k in unknown),
            "unknown_total": len(unknown),
            "expected_source_paths": reference["source_paths"],
            "located_source_paths": sorted(located),
            "source_path_hits": len(set(reference["source_paths"]) & located),
            "source_path_total": len(reference["source_paths"]),
            "citation_ids_valid": sum(r in ev for r in refs),
            "citation_ids_total": len(refs),
            "status": prediction["status"],
            "reference_fix": reference["fix"],
            "root_cause_accuracy": "NOT SCORED: error-family recognition is not root-cause or repair success",
            "report_sha": prediction.get("report_sha"),
        }
        rows.append(row)
    n = len(rows)
    totals = {
        key: sum(r[key] for r in rows)
        for key in [
            "known_facts_correct",
            "known_facts_total",
            "unknown_retained",
            "unknown_total",
            "source_path_hits",
            "source_path_total",
            "citation_ids_valid",
            "citation_ids_total",
        ]
    }
    result = {
        "dataset": index["version"],
        "assigned": n,
        "sealed": sum(r["status"] == "sealed" for r in rows),
        "type_correct": sum(r["type_correct"] for r in rows),
        "type_accuracy": sum(r["type_correct"] for r in rows) / n if n else None,
        **totals,
        "inputs_unchanged": tree(dataset / "inputs") == before,
        "memory_count": len(app.store.all()),
        "external_llm_requests": 0,
        "real_llm": "NOT VERIFIED",
        "New": "NOT TRAINED",
        "algorithm_gains": "NOT ESTABLISHED",
        "seconds": time.perf_counter() - started,
        "seal": seal,
        "source_manifest_sha": file_hash(dataset / "SOURCE_IDENTITIES.json"),
        "interpretation": "Curated development case audit of bounded public pre-repair evidence. Not a full repository build, untouched held-out test, or LLM success benchmark.",
        "rows": rows,
    }
    result["summary_sha"] = digest(result)
    write_json(output / "summary.json", result)
    write_json(
        output / "failures.json",
        [
            r
            for r in rows
            if not r["type_correct"]
            or r["known_facts_correct"] < r["known_facts_total"]
            or r["source_path_hits"] < r["source_path_total"]
        ],
    )
    return result
