import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from experienceguard.agents.development import prepare_agent, run_agent, AgentLimits
from experienceguard.agents.protocol_test import ProtocolTestProvider, protocol_test
from experienceguard.agents.providers import HTTPSettings
from experienceguard.workflows.service import DevelopmentApp
from experienceguard.training.development import build_pair
from experienceguard.utils.core import read_json


def test_http_retry_and_failed_attempt_are_counted_and_audited(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("import absent\n")
    app = DevelopmentApp(tmp_path / "app", allowed_roots=[repo])
    case = app.diagnose(repo, "ModuleNotFoundError: No module named 'absent'")
    plan = prepare_agent(app, case["case_id"])
    count = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            count.append(1)
            if len(count) == 1:
                self.send_error(429)
                return
            action = (
                {"tool": "read_log", "arguments": {"line_start": 1, "line_end": 1}}
                if len(count) == 2
                else {
                    "final": {
                        "summary": "TEST_ONLY fixture",
                        "next_steps": [],
                        "uncertainties": ["No real inference"],
                        "memory_id": None,
                        "citations": [
                            {
                                "evidence_id": plan["documents"]["ci.log"]["id"],
                                "line_start": 1,
                                "line_end": 1,
                            }
                        ],
                    }
                }
            )
            body = json.dumps(
                {"choices": [{"message": {"content": json.dumps(action)}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = ProtocolTestProvider(
            HTTPSettings(
                f"http://127.0.0.1:{server.server_port}/v1", "TEST_ONLY", retries=1
            )
        )
        result = run_agent(
            app,
            plan,
            provider,
            consent_sha=plan["plan_sha256"],
            limits=AgentLimits(max_requests=5),
        )
        assert (
            result["status"] == "completed"
            and result["metrics"]["requests"] == 3
            and len(count) == 3
        )
        requests = read_json(
            app._case_dir(case["case_id"]) / result["run_id"] / "requests.json"
        )
        assert requests[0]["failure"] == "HTTPError" and len(requests) == 3
        assert result["metrics"]["cost_usd"] == 0 and result["mode"] == "mock_http_test"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_mock_execution_feedback_cannot_become_production_pair(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("from lib import missing\n")
    (repo / "lib.py").write_text('__version__="1.0.0"\ndef present():return 1\n')
    app = DevelopmentApp(tmp_path / "app", allowed_roots=[repo])
    log = "Python 3.12.0\nImportError: cannot import name 'missing' from 'lib'"
    source = app.diagnose(repo, log)
    app.verify_and_save(
        source["case_id"],
        description="TEST_ONLY prior user feedback",
        manual={"outcome": "success", "verification_log": "TEST_ONLY"},
    )
    a = app.diagnose(repo, log, strategy="B0")
    b = app.diagnose(repo, log, strategy="B4")
    ra = protocol_test(app, prepare_agent(app, a["case_id"]))
    rb = protocol_test(app, prepare_agent(app, b["case_id"]))
    va = app.verify_and_save(
        a["case_id"],
        agent_run_id=ra["run_id"],
        description="TEST_ONLY paired action",
        manual={"outcome": "success", "verification_log": "TEST_ONLY"},
    )
    vb = app.verify_and_save(
        b["case_id"],
        agent_run_id=rb["run_id"],
        description="TEST_ONLY paired action",
        manual={"outcome": "success", "verification_log": "TEST_ONLY"},
    )
    with pytest.raises(ValueError, match="Mock"):
        build_pair(
            app,
            a["case_id"],
            b["case_id"],
            va["receipt_id"],
            vb["receipt_id"],
            split="fit",
            family="separate-test-family",
            protocol_approved=True,
        )
