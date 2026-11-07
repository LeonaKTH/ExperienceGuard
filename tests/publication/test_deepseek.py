"""Contract tests with injected responses: no real inference, credentials or fees."""
import json
from pathlib import Path
import urllib.error
import pytest
from experienceguard.agents.deepseek import Ledger, DeepSeekJSONProvider, configured_provider
from experienceguard.agents.development import prepare_agent, run_remote
from experienceguard.agents.providers import ProviderError
from experienceguard.types import BudgetExceeded
from experienceguard.workflows.service import DevelopmentApp


@pytest.fixture
def setup(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("import missing_dependency\n", encoding="utf8")
    app = DevelopmentApp(tmp_path / "state", allowed_roots=[repo])
    report = app.diagnose(repo, "ModuleNotFoundError: No module named 'missing_dependency'")
    plan = prepare_agent(app, report["case_id"], ["app.py"])
    cfg = json.loads((Path(__file__).parents[2] / "configs/deepseek.example.json").read_text())
    cfg.update(input_usd_per_million=1.0, output_usd_per_million=1.0,
               pricing_reference="UNIT TEST PRICES ONLY", advertised_version="TEST_ONLY")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setenv("EG_DEEPSEEK_CONFIG", str(path))
    monkeypatch.setenv("EG_V2_REMOTE_ENABLE", "true")
    monkeypatch.setenv("EG_V2_API_KEY", "UNIT_ONLY_CREDENTIAL_NO_REAL_AUTH")
    return app, plan, cfg


def test_packaged_application_runs_v2_json_provider_multiturn(setup, monkeypatch):
    app, plan, config = setup
    calls = []
    actions = [
        {"tool": "list_repository", "arguments": {}},
        {"tool": "read_log", "arguments": {"line_start": 1, "line_end": 1}},
        {"tool": "read_file", "arguments": {"path": "app.py", "line_start": 1, "line_end": 1}},
        {"final": {"summary": "TEST ONLY diagnosis", "next_steps": [], "uncertainties": [],
                   "memory_id": None, "citations": [
                       {"evidence_id": plan["documents"][name]["id"], "line_start": 1, "line_end": 1}
                       for name in ["ci.log", "app.py"]]}}
    ]

    def transport(url, payload, headers, timeout):
        assert url == "https://api.deepseek.com/chat/completions"
        assert payload["thinking"] == {"type": "disabled"}
        assert payload["response_format"] == {"type": "json_object"} and "seed" not in payload
        calls.append(payload)
        return {"id": "TEST_ONLY", "model": "deepseek-flash", "usage": {"prompt_tokens": 100, "completion_tokens": 40},
                "choices": [{"message": {"content": json.dumps(actions[len(calls)-1])}}]}

    monkeypatch.setattr("experienceguard.agents.development.http_transport", transport)
    result = run_remote(app, plan, consent_sha=plan["plan_sha256"], allow_remote=True,
                        acknowledge_upload=True, confirm_cost=True)
    assert result["status"] == "completed" and result["metrics"]["requests"] == 4
    assert result["metrics"]["input_tokens"] == 400
    assert "tool_result" in calls[1]["messages"][-1]["content"]
    state = json.loads((app.data_dir / "remote-budget.json").read_text())
    assert len(state["entries"]) == 4 and all(e["status"] == "settled" for e in state["entries"])
    assert not app.store.all()
    assert "UNIT_ONLY_CREDENTIAL_NO_REAL_AUTH" not in json.dumps(result)


def test_missing_key_or_template_prices_never_constructs_requests(setup, monkeypatch):
    app, plan, config = setup
    monkeypatch.setattr("experienceguard.agents.development.http_transport", lambda *a: pytest.fail("Unexpected request"))
    monkeypatch.delenv("EG_V2_API_KEY")
    result = run_remote(app, plan, consent_sha=plan["plan_sha256"], allow_remote=True,
                        acknowledge_upload=True, confirm_cost=True)
    assert result["status"] == "UNAVAILABLE" and result["requests"] == 0
    monkeypatch.setenv("EG_V2_API_KEY", "UNIT_ONLY_CREDENTIAL_NO_REAL_AUTH")
    config["input_usd_per_million"] = 0
    Path(__import__('os').environ["EG_DEEPSEEK_CONFIG"]).write_text(json.dumps(config))
    with pytest.raises(ProviderError):
        configured_provider(app, plan)


def test_credential_echo_blocked_before_raw_response_persistence(setup, monkeypatch):
    app, plan, config = setup
    monkeypatch.setattr("experienceguard.agents.development.http_transport", lambda *a:
                        {"choices": [{"message": {"content": "UNIT_ONLY_CREDENTIAL_NO_REAL_AUTH"}}]})
    result = run_remote(app, plan, consent_sha=plan["plan_sha256"], allow_remote=True,
                        acknowledge_upload=True, confirm_cost=True)
    assert result["status"] == "provider_failure" and result["metrics"]["requests"] == 1
    assert not list(app.data_dir.rglob("raw-response-*.json"))
    assert all(b"UNIT_ONLY_CREDENTIAL_NO_REAL_AUTH" not in p.read_bytes()
               for p in app.data_dir.rglob('*') if p.is_file() and p.suffix != '.lock')


def test_failures_keep_durable_reservations_and_restart_request_cap(setup):
    app, plan, config = setup
    path = app.data_dir / 'ledger-unit.json'
    ledger = Ledger(path, budget=.1, request_cap=1, binding='TEST_ONLY')
    provider = DeepSeekJSONProvider(config, ledger, 'episode', allow_remote=True,
                                   transport=lambda *a: (_ for _ in ()).throw(urllib.error.URLError('TEST_ONLY')))
    with pytest.raises(ProviderError):
        provider.generate([{'role': 'user', 'content': 'JSON test'}], [])
    state = ledger._state()
    assert len(state['entries']) == 1 and state['entries'][0]['charge'] > 0
    with pytest.raises(BudgetExceeded):
        Ledger(path, budget=.1, request_cap=1, binding='TEST_ONLY').reserve(.001, 'episode', .05)
    with pytest.raises(ValueError):
        Ledger(path, budget=.2, request_cap=1, binding='changed')._state()
