import json
import threading
import urllib.request
from pathlib import Path
from experienceguard.workflows.service import DevelopmentApp
from experienceguard.api.server import create_server
from experienceguard.evaluation.agent import evaluate_agent


def test_agent_api_disclosure_mock_and_unavailable_modes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "worker.py").write_text("import absent_module\n")
    app = DevelopmentApp(
        tmp_path / "app",
        allowed_roots=[
            repo,
            Path(__file__).resolve().parents[2] / "datasets/public_failures/inputs",
        ],
    )
    report = app.diagnose(repo, "ModuleNotFoundError: No module named 'absent_module'")
    server = create_server(app, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path, body=None):
        r = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}" + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(r, timeout=20) as response:
            return json.load(response)

    try:
        assert request("/api/state")["modes"]["agent"] == "UNAVAILABLE"
        cases = request("/api/public-cases")
        assert cases == []  # Optional public corpus is not redistributed.
        plan = request(
            "/api/agent/prepare", {"case_id": report["case_id"], "paths": ["worker.py"]}
        )
        assert plan["network_requests"] == 0
        run = request(
            "/api/agent/protocol-test",
            {"case_id": report["case_id"], "plan_id": plan["plan_id"]},
        )
        assert run["mode"] == "mock_http_test" and run["status"] == "completed"
        assert (
            request(
                "/api/agent/result?case_id="
                + report["case_id"]
                + "&run_id="
                + run["run_id"]
            )["execution_sha"]
            == run["execution_sha"]
        )
        blocked = request(
            "/api/agent/run",
            {
                "case_id": report["case_id"],
                "plan_id": plan["plan_id"],
                "consent_sha": plan["plan_sha256"],
                "acknowledge_upload": True,
                "confirm_cost": True,
            },
        )
        assert blocked["status"] == "UNAVAILABLE" and blocked["requests"] == 0
        assert not app.store.all()
        saved = request(
            "/api/verify",
            {
                "case_id": report["case_id"],
                "agent_run_id": run["run_id"],
                "description": "TEST_ONLY actual user feedback path",
                "manual": {
                    "outcome": "inconclusive",
                    "verification_log": "No independent validation has occurred",
                },
            },
        )
        assert (
            saved["experience_id"] is None
            and saved["validation"]["agent_mode"] == "mock_http_test"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_pilot_keeps_all_assignment_rows_and_no_default_remote(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("import absent_module\n")
    app = DevelopmentApp(tmp_path / "app", allowed_roots=[repo])
    case = app.diagnose(repo, "ModuleNotFoundError: No module named 'absent_module'")
    result = evaluate_agent(
        app, [{"case_id": case["case_id"], "paths": ["a.py"]}], tmp_path / "pilot"
    )
    assert result["assigned"] == 6 and len(result["rows"]) == 6
    assert all(r.get("requests", 0) == 0 for r in result["rows"])
    assert (
        any(r["status"] == "UNTRAINED" for r in result["rows"])
        and result["real_llm"] == "NOT VERIFIED"
    )
    assert not app.store.all()
