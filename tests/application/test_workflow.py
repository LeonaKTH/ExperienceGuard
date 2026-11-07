from pathlib import Path
import json
import subprocess
import sys
import pytest
from experienceguard.workflows.service import DevelopmentApp
from experienceguard.workflows.fixtures import file_map, validate_proposal
from experienceguard.features.development import compare_conditions
from experienceguard.utils.core import read_json


@pytest.fixture
def app(tmp_path):
    x = DevelopmentApp(tmp_path / "appdata")
    x.initialize()
    return x


def diagnostic(app, name, strategy="B4"):
    f = next(x for x in read_json(app.data_dir / "fixtures.json") if x["name"] == name)
    return app.diagnose(
        f["repository"],
        Path(f["log_file"]).read_text(encoding="utf8"),
        strategy=strategy,
    )


def seed(app):
    r = diagnostic(app, "source")
    result = app.verify_and_save(r["case_id"], proposal_id=r["proposals"][0]["id"])
    return r, result


def test_real_failure_independent_copy_and_readonly(app):
    r = diagnostic(app, "source")
    before = file_map(r["repository"]["path"])
    assert r["repository"]["commit"] and r["input_provenance"] == "controlled_fixture"
    assert r["primary_issue"]["kind"] == "python_import_symbol"
    assert r["facts"]["python"] == sys.version.split()[0]
    assert r["facts"]["declared_python"] == "3.13"
    assert r["proposals"][0]["new_symbol"] == "load_config"
    result = app.verify_and_save(r["case_id"], proposal_id=r["proposals"][0]["id"])
    assert result["validation"]["before"]["exit_code"] != 0
    assert result["validation"]["after"]["exit_code"] == 0
    assert file_map(r["repository"]["path"]) == before
    assert app.store.get(result["experience_id"]).source_task == r["case_id"]
    assert (app._case_dir(r["case_id"]) / "raw.log").read_text(encoding="utf8")


def test_reuse_conflict_unknown_and_fresh_evidence(app):
    _, saved = seed(app)
    good = diagnostic(app, "compatible")
    bad = diagnostic(app, "conflict")
    unknown = diagnostic(app, "unknown")
    assert good["decision"]["experience_id"] == saved["experience_id"]
    assert good["proposals"][0]["path"] == "settings_reader.py"
    assert bad["decision"]["experience_id"] is None and any(
        x["status"] == "conflict" for x in bad["candidates"][0]["conditions"]
    )
    assert bad["proposals"][0]["new_symbol"] == "read_config"
    assert (
        unknown["facts"]["python"] is None
        and unknown["decision"]["experience_id"] is None
    )
    assert any(
        x["key"] == "python" and x["status"] == "unknown"
        for x in unknown["candidates"][0]["conditions"]
    )


def test_cross_process_memory_not_just_object_lifetime(app, tmp_path):
    _, saved = seed(app)
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "experienceguard",
            "diagnose",
            "--data-dir",
            str(app.data_dir),
            "--fixture",
            "compatible",
        ],
        capture_output=True,
        text=True,
        encoding="utf8",
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    r = json.loads(process.stdout)
    assert r["decision"]["experience_id"] == saved["experience_id"]
    assert (
        r["candidates"][0]["content_sha256"]
        == app.store.get(saved["experience_id"]).content_hash
    )


def test_prior_repair_really_fails_after_dependency_change(app):
    _, saved = seed(app)
    r = diagnostic(app, "conflict")
    stale = dict(r["proposals"][0], new_symbol="load_config")
    # Explicit test intervention only; no normal workflow reads a gold repair or fault ID.
    intervention = dict(r, proposals=[stale])
    result = validate_proposal(app.data_dir, intervention, stale)
    assert result["after"]["exit_code"] != 0
    fresh = validate_proposal(app.data_dir, r, r["proposals"][0])
    assert fresh["after"]["exit_code"] == 0 and fresh["repository_unchanged"]


def test_no_model_no_network_and_unavailable_policies(app, monkeypatch):
    import urllib.request

    monkeypatch.setattr(
        urllib.request.OpenerDirector,
        "open",
        lambda *a, **k: pytest.fail("unexpected external request"),
    )
    for strategy, status in [("New", "UNTRAINED"), ("B5", "UNAVAILABLE")]:
        r = diagnostic(app, "source", strategy)
        assert (
            r["policy_status"] == status
            and r["decision"]["experience_id"] is None
            and not r["decision"]["scores"]
        )
        assert r["suggestions"] and r["llm_status"] == "UNAVAILABLE / NOT CALLED"


def test_manual_feedback_preserves_original_and_failed_not_reused(app):
    r = diagnostic(app, "source")
    with pytest.raises(ValueError):
        app.verify_and_save(
            r["case_id"],
            manual={"outcome": "success", "verification_log": ""},
            description="claimed success",
        )
    v = app.verify_and_save(
        r["case_id"],
        manual={
            "outcome": "failure",
            "verification_log": "Actual test still raises ImportError",
        },
        description="Tried a repair; failed",
    )
    exp = app.store.get(v["experience_id"])
    assert not exp.feedback["success"]
    assert diagnostic(app, "compatible")["decision"]["experience_id"] is None
    original = app.load_case(r["case_id"])
    assert original["report_sha256"] == r["report_sha256"]


def test_inconclusive_and_supersession_append_only(app):
    r, saved = seed(app)
    no = app.verify_and_save(
        r["case_id"],
        manual={"outcome": "inconclusive", "verification_log": "Tests incomplete"},
        description="No conclusion",
    )
    assert no["experience_id"] is None
    v = app.verify_and_save(
        r["case_id"],
        proposal_id=r["proposals"][0]["id"],
        supersedes=saved["experience_id"],
    )
    assert (
        app.store.get(saved["experience_id"])
        and app.store.get(v["experience_id"]).supersedes == saved["experience_id"]
    )
    assert (
        diagnostic(app, "compatible")["decision"]["experience_id"] == v["experience_id"]
    )


def test_corrupted_report_and_stale_fixture_fail_closed(app):
    r = diagnostic(app, "source")
    p = Path(r["repository"]["path"]) / "worker.py"
    p.write_text(p.read_text() + "\n# modified\n")
    with pytest.raises(ValueError):
        app.verify_and_save(r["case_id"], proposal_id=r["proposals"][0]["id"])
    path = app._case_dir(r["case_id"]) / "diagnosis.json"
    x = read_json(path)
    x["facts"]["python"] = "0"
    path.write_text(json.dumps(x))
    with pytest.raises(ValueError):
        app.load_case(r["case_id"])


def test_unknown_is_not_false_or_a_guess():
    result = compare_conditions(
        {"python": None}, {"python": {"op": "version", "value": ">=3.12"}}
    )
    assert result[0]["status"] == "unknown"
    assert (
        compare_conditions(
            {"python": "not-a-version"},
            {"python": {"op": "version", "value": ">=3.12"}},
        )[0]["status"]
        == "unknown"
    )


def test_reinitialization_does_not_overwrite_failure_evidence(app, monkeypatch):
    before = {
        p.name: p.read_bytes()
        for p in (app.data_dir / "workspaces/source").glob("*.json")
    }
    log = (app.data_dir / "workspaces/source/failure.log").read_bytes()
    monkeypatch.setattr(
        "experienceguard.workflows.fixtures.test_copy",
        lambda *a: pytest.fail(
            "reinitialization should reuse immutable fixture evidence"
        ),
    )
    app.initialize()
    assert (app.data_dir / "workspaces/source/failure.log").read_bytes() == log
    assert all(
        (app.data_dir / "workspaces/source" / name).read_bytes() == data
        for name, data in before.items()
    )
