import pytest
from pathlib import Path
from experienceguard.workflows.service import DevelopmentApp
from experienceguard.training.development import build_pair
from experienceguard.training.trainer import validate_rows
from experienceguard.domain.development import DOMAIN
from experienceguard.evaluation.development import evaluate_diagnostics


def test_verified_pair_export_and_evaluation_frozen_memory(tmp_path):
    app = DevelopmentApp(tmp_path / "app")
    fixtures = app.initialize()

    def run(name, strategy):
        f = next(x for x in fixtures if x["name"] == name)
        return app.diagnose(
            f["repository"],
            Path(f["log_file"]).read_text(encoding="utf8"),
            strategy=strategy,
        )

    source = run("source", "B4")
    app.verify_and_save(source["case_id"], proposal_id=source["proposals"][0]["id"])
    # Freeze both diagnoses before any target feedback is promoted into memory.
    control = run("compatible", "B0")
    treated = run("compatible", "B4")
    a = app.verify_and_save(
        control["case_id"], proposal_id=control["proposals"][0]["id"]
    )
    b = app.verify_and_save(
        treated["case_id"], proposal_id=treated["proposals"][0]["id"]
    )
    row = build_pair(
        app,
        control["case_id"],
        treated["case_id"],
        a["receipt_id"],
        b["receipt_id"],
        split="fit",
        family="held-out-family",
        protocol_approved=True,
    )
    assert row["label"] == "tie" and row["input_provenance"] == "controlled_fixture"
    with pytest.raises(ValueError):
        validate_rows([row], domain=DOMAIN)
    with pytest.raises(ValueError):
        build_pair(
            app,
            control["case_id"],
            treated["case_id"],
            a["receipt_id"],
            b["receipt_id"],
            split="test",
            family="test",
        )
    f = next(x for x in fixtures if x["name"] == "conflict")
    result = evaluate_diagnostics(
        app,
        [
            {
                "repository": f["repository"],
                "log": Path(f["log_file"]).read_text(encoding="utf8"),
            }
        ],
        tmp_path / "evaluation",
    )
    assert result["assigned"] == 7 and result["completed_diagnoses"] == 7
    assert (
        result["unavailable_policies"] == 3
    )  # B1 without embedding, B5 without evidence, New without weights
