import json
import threading
import urllib.request
import urllib.error
import pytest
from experienceguard.api.server import create_server
from experienceguard.workflows.service import DevelopmentApp


def test_interactive_http_diagnosis_save_and_reuse(tmp_path):
    app = DevelopmentApp(tmp_path / "app")
    app.initialize()
    server = create_server(app, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def request(path, data=None, origin=None):
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        req = urllib.request.Request(
            base + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)

    try:
        with urllib.request.urlopen(base, timeout=5) as r:
            assert "无 LLM 工程模式" in r.read().decode()
            assert "default-src 'self'" in r.headers["Content-Security-Policy"]
        a = request("/api/diagnose", {"fixture": "source", "strategy": "B4"})
        saved = request(
            "/api/verify",
            {
                "case_id": a["case_id"],
                "proposal_id": a["proposals"][0]["id"],
                "actor": "test developer",
            },
        )
        b = request("/api/diagnose", {"fixture": "compatible", "strategy": "B4"})
        assert b["decision"]["experience_id"] == saved["experience_id"]
        c = request("/api/diagnose", {"fixture": "conflict", "strategy": "B4"})
        assert (
            c["decision"]["experience_id"] is None
            and c["proposals"][0]["new_symbol"] == "read_config"
        )
        assert (
            request("/api/state")["memory"][0]["provenance"]
            == "controlled_fixture_verified"
        )
        with pytest.raises(urllib.error.HTTPError):
            request(
                "/api/diagnose",
                {"fixture": "source", "command": "rm -rf"},
                origin="http://evil.invalid",
            )
        with pytest.raises(urllib.error.HTTPError):
            request("/api/diagnose", {"fixture": "source", "command": "anything"})
        with pytest.raises(urllib.error.HTTPError):
            request("/api/case?id=../../raw.log")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
