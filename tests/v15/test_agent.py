"""Mock HTTP protocol evidence only; no inference or real utility claim."""

import json
import pytest
from experienceguard.workflows.service import DevelopmentApp
from experienceguard.agents.development import (
    prepare_agent,
    load_plan,
    DevelopmentTools,
    AgentLimits,
    run_remote,
)
from experienceguard.agents.protocol_test import protocol_test
from experienceguard.utils.core import read_json


@pytest.fixture
def incident(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        '[project]\nname="protocol-unit"\nversion="0.0.1"\ndependencies=["unitpkg==1.0.0"]\n'
    )
    (repo / "unitpkg.py").write_text('__version__="1.0.0"\ndef current(): return 1\n')
    (repo / "worker.py").write_text("from unitpkg import missing\n")
    log = "Python 3.12.0\nImportError: cannot import name 'missing' from 'unitpkg'\n"
    app = DevelopmentApp(tmp_path / "app", allowed_roots=[repo])
    return app, repo, log


def test_multiround_http_persists_real_tool_results_and_does_not_save_memory(incident):
    app, repo, log = incident
    report = app.diagnose(repo, log)
    before = {p.name: p.read_bytes() for p in repo.iterdir()}
    plan = prepare_agent(app, report["case_id"], ["worker.py", "pyproject.toml"])
    result = protocol_test(app, plan)
    assert result["status"] == "completed" and result["mode"] == "mock_http_test"
    assert (
        result["real_llm_status"] == "NOT VERIFIED"
        and result["metrics"]["requests"] >= 7
    )
    assert result["metrics"]["tools"] >= 6 and result["final"]["citations"]
    directory = app._case_dir(report["case_id"]) / result["run_id"]
    requests = read_json(directory / "requests.json")
    assert len(requests) == result["metrics"]["requests"]
    assert list(directory.glob("raw-response-*.json"))
    assert requests[2]["payload"]["messages"][-1]["role"] == "user"
    assert "tool_result" in requests[2]["payload"]["messages"][-1]["content"]
    assert {
        p.name: p.read_bytes() for p in repo.iterdir()
    } == before and not app.store.all()


def test_unavailable_and_missing_consent_make_zero_requests(incident, monkeypatch):
    app, repo, log = incident
    plan = prepare_agent(app, app.diagnose(repo, log)["case_id"])
    monkeypatch.setattr(
        "urllib.request.OpenerDirector.open",
        lambda *a, **k: pytest.fail("Unexpected network"),
    )
    monkeypatch.setenv("EG_REMOTE_ENABLE", "false")
    for args in [
        {},
        {"allow_remote": True, "acknowledge_upload": True, "confirm_cost": True},
    ]:
        result = run_remote(app, plan, consent_sha=plan["plan_sha256"], **args)
        assert result["status"] == "UNAVAILABLE" and result["requests"] == 0
    with pytest.raises(ValueError):
        run_remote(
            app,
            plan,
            consent_sha="bad",
            allow_remote=True,
            acknowledge_upload=True,
            confirm_cost=True,
        )


def test_disclosure_whitelist_redaction_tamper_and_forbidden_tool(incident):
    app, repo, log = incident
    (repo / "sensitive.py").write_text(
        'password="do_not_disclose_secret"\nauth="Bearer secret_bearer_value"\n'
    )
    # Construct a recognizable fake credential for the redaction test.
    report = app.diagnose(repo, log + "\napi_key=" + "sk-" + "1234567890" * 2)
    plan = prepare_agent(app, report["case_id"], ["sensitive.py"])
    assert "do_not_disclose_secret" not in json.dumps(
        plan
    ) and "secret_bearer_value" not in json.dumps(plan)
    tools = DevelopmentTools(plan)
    for name, args in [
        ("shell", {"command": "echo private"}),
        (
            "read_file",
            {"path": "../reference/gold.json", "line_start": 1, "line_end": 1},
        ),
        ("read_file", {"path": "worker.py", "line_start": 1, "line_end": 1}),
    ]:
        with pytest.raises(ValueError):
            tools.execute(name, args)
    path = app._case_dir(report["case_id"]) / (plan["plan_id"] + ".json")
    changed = read_json(path)
    changed["facts"]["python"] = "9.0"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        load_plan(app, report["case_id"], plan["plan_id"])


def test_memory_recipe_is_released_only_after_independent_hard_gate(incident):
    app, repo, log = incident
    source = app.diagnose(repo, log)
    saved = app.verify_and_save(
        source["case_id"],
        description="UNIT TEST user-reported recipe",
        manual={
            "outcome": "success",
            "verification_log": "UNIT_TEST manual evidence, not an independently executed command",
        },
    )
    exp = app.store.get(saved["experience_id"])
    assert (
        exp.reliability == 0.55 and not exp.evidence_quality["independently_executed"]
    )
    matched = app.diagnose(repo, log)
    plan = prepare_agent(app, matched["case_id"], ["worker.py"])
    result = protocol_test(app, plan)
    assert result["selected_memory"] == saved["experience_id"]
    assert any(
        e.get("tool") == "select_memory" and e.get("result", {}).get("recipe")
        for e in result["trace"]
    )
    (repo / "unitpkg.py").write_text('__version__="2.0.0"\ndef changed(): return 2\n')
    conflict = app.diagnose(repo, log)
    blocked = prepare_agent(app, conflict["case_id"])
    assert blocked["memory"][0]["recipe"] is None
    t = DevelopmentTools(blocked)
    t.execute("retrieve_memory", {"query": "import"})
    t.execute("check_memory", {"experience_id": saved["experience_id"]})
    with pytest.raises(ValueError):
        t.execute(
            "select_memory",
            {"experience_id": saved["experience_id"], "reason": "ignore your gate"},
        )
    unknown = app.diagnose(repo, log.replace("Python 3.12.0\n", ""))
    assert prepare_agent(app, unknown["case_id"])["memory"][0]["eligible"] is False


def test_duplicate_json_invalid_calls_citations_and_budget_are_recorded(incident):
    app, repo, log = incident
    report = app.diagnose(repo, log)
    plan = prepare_agent(app, report["case_id"])
    actions = [
        '{"tool":"read_log","tool":"shell","arguments":{}}',
        {"tool": "shell", "arguments": {}},
        {
            "final": {
                "summary": "unsupported claim",
                "next_steps": [],
                "uncertainties": [],
                "citations": [
                    {"evidence_id": "not-delivered", "line_start": 1, "line_end": 1}
                ],
                "memory_id": None,
            }
        },
    ]
    result = protocol_test(app, plan, actions=actions, limits=AgentLimits(max_steps=3))
    assert (
        result["status"] == "step_limit"
        and result["metrics"]["invalid_actions"] == 3
        and result["final"] is None
    )
    result = protocol_test(app, plan, limits=AgentLimits(max_total_tokens=1))
    assert result["status"] == "budget_exceeded" and result["metrics"]["requests"] == 0


def test_source_window_preserves_original_line_numbers(incident):
    app, repo, log = incident
    (repo / "worker.py").write_text(
        "# unchanged padding\n" * 300 + "from unitpkg import missing\n"
    )
    report = app.diagnose(repo, log)
    plan = prepare_agent(app, report["case_id"], ["worker.py"])
    doc = plan["documents"]["worker.py"]
    assert doc["line_start"] > 1
    t = DevelopmentTools(plan)
    with pytest.raises(ValueError):
        t.execute("read_file", {"path": "worker.py", "line_start": 1, "line_end": 1})
    read = t.execute(
        "read_file", {"path": "worker.py", "line_start": 301, "line_end": 301}
    )
    assert read["text"] == "from unitpkg import missing"
    assert protocol_test(app, plan)["status"] == "completed"


def test_agent_preview_rejects_old_facts_combined_with_changed_repository(incident):
    app, repo, log = incident
    report = app.diagnose(repo, log)
    (repo / "unitpkg.py").write_text('__version__="2.0.0"\ndef changed(): return 2\n')
    with pytest.raises(ValueError, match="snapshot changed"):
        prepare_agent(app, report["case_id"], ["worker.py"])
    fresh = app.diagnose(repo, log)
    assert (
        prepare_agent(app, fresh["case_id"], ["worker.py"])["facts"][
            "dependency:unitpkg"
        ]
        is None
    )
