"""Loopback application server with real input/diagnosis/verification/persistence routes."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import json
from ..repository_tools.reader import redact
from ..utils.core import read_json, canonical


def public(value):
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [public(v) for v in value]
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items()}
    return value


def create_server(app, port=8766):
    assets = Path(__file__).parent

    class Handler(BaseHTTPRequestHandler):
        def respond(self, value, status=200, kind="application/json; charset=utf-8"):
            data = (
                canonical(public(value)).encode()
                if kind.startswith("application/json")
                else value
            )
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

        def validate_origin(self):
            expected = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            if self.headers.get("Host") not in expected:
                raise ValueError("loopback Host required")
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + h for h in expected}:
                raise ValueError("cross-origin requests refused")

        def do_GET(self):
            try:
                self.validate_origin()
                url = urlsplit(self.path)
                query = parse_qs(url.query)
                if url.path in {"/", "/app.css", "/app.js"}:
                    name = "index.html" if url.path == "/" else url.path[1:]
                    return self.respond(
                        (assets / name).read_bytes(),
                        kind={
                            "index.html": "text/html; charset=utf-8",
                            "app.css": "text/css; charset=utf-8",
                            "app.js": "application/javascript; charset=utf-8",
                        }[name],
                    )
                if url.path == "/api/state":
                    fixtures = (
                        read_json(app.data_dir / "fixtures.json")
                        if (app.data_dir / "fixtures.json").exists()
                        else []
                    )
                    return self.respond(
                        {
                            "mode": "evidence_only",
                            "llm": "UNAVAILABLE / NOT CALLED",
                            "modes": {
                                "evidence": "AVAILABLE",
                                "mock": "LOCAL_HTTP_PROTOCOL_TEST_ONLY",
                                "agent": "REQUIRES_CONFIGURATION_AND_PER_RUN_CONSENT"
                                if getattr(app, "allow_remote", False)
                                else "UNAVAILABLE",
                            },
                            "fixtures": fixtures,
                            "cases": app.list_cases(),
                            "memory": [e.to_dict() for e in app.store.all()],
                            "allowed_roots": [str(p) for p in app.allowed_roots],
                            "data_dir": str(app.data_dir),
                        }
                    )
                if url.path in {"/api/public-cases", "/api/public-case"}:
                    from ..evaluation.public_cases import public_input

                    dataset = (
                        Path(__file__).resolve().parents[3] / "datasets/public_failures"
                    )
                    if not dataset.exists():
                        import sys

                        dataset = (
                            Path(sys.prefix)
                            / "share/experienceguard/datasets/public_failures"
                        )
                    if url.path == "/api/public-cases":
                        cases = (
                            read_json(dataset / "cases.json")["cases"]
                            if (dataset / "cases.json").exists()
                            else []
                        )
                        return self.respond(
                            [
                                {
                                    "case_id": c["case_id"],
                                    "repository": c["repository"],
                                    "revision": c["revision"],
                                    "issue_url": c["issue_url"],
                                    "evidence_grade": c["evidence_grade"],
                                }
                                for c in cases
                            ]
                        )
                    return self.respond(public_input(dataset, query.get("id", [""])[0]))
                if url.path == "/api/fixture":
                    name = query.get("name", [""])[0]
                    f = next(
                        x
                        for x in read_json(app.data_dir / "fixtures.json")
                        if x["name"] == name
                    )
                    return self.respond(
                        {**f, "log": Path(f["log_file"]).read_text(encoding="utf8")}
                    )
                if url.path in {"/api/case", "/api/report"}:
                    return self.respond(app.load_case(query.get("id", [""])[0]))
                if url.path == "/api/agent/result":
                    from ..agents.development import load_execution

                    return self.respond(
                        load_execution(
                            app,
                            query.get("case_id", [""])[0],
                            query.get("run_id", [""])[0],
                        )
                    )
                return self.respond({"error": "not found"}, 404)
            except Exception as exc:
                self.respond({"error": type(exc).__name__ + ": " + str(exc)}, 400)

        def do_POST(self):
            try:
                self.validate_origin()
                if (
                    self.headers.get("Content-Type", "").split(";")[0]
                    != "application/json"
                ):
                    raise ValueError("JSON content type required")
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 500000:
                    raise ValueError("invalid body size")
                body = json.loads(self.rfile.read(n))
                if not isinstance(body, dict):
                    raise ValueError("request must be an object")
                if self.path == "/api/agent/prepare":
                    from ..agents.development import prepare_agent

                    if set(body) - {"case_id", "paths"}:
                        raise ValueError("unsupported Agent preview fields")
                    return self.respond(
                        prepare_agent(app, body["case_id"], body.get("paths", []))
                    )
                if self.path in {"/api/agent/run", "/api/agent/protocol-test"}:
                    from ..agents.development import load_plan, run_remote

                    if set(body) - {
                        "case_id",
                        "plan_id",
                        "consent_sha",
                        "acknowledge_upload",
                        "confirm_cost",
                    }:
                        raise ValueError("unsupported Agent execution fields")
                    plan = load_plan(app, body["case_id"], body["plan_id"])
                    if self.path == "/api/agent/protocol-test":
                        from ..agents.protocol_test import protocol_test

                        return self.respond(protocol_test(app, plan))
                    return self.respond(
                        run_remote(
                            app,
                            plan,
                            consent_sha=body.get("consent_sha"),
                            allow_remote=getattr(app, "allow_remote", False),
                            acknowledge_upload=body.get("acknowledge_upload") is True,
                            confirm_cost=body.get("confirm_cost") is True,
                        )
                    )
                if self.path == "/api/diagnose":
                    if set(body) - {
                        "fixture",
                        "repository",
                        "log",
                        "strategy",
                        "public_case_id",
                    }:
                        raise ValueError("unsupported diagnostic fields")
                    provenance, public_url = "user_repository", None
                    if body.get("public_case_id"):
                        from ..evaluation.public_cases import public_input

                        dataset = (
                            Path(__file__).resolve().parents[3]
                            / "datasets/public_failures"
                        )
                        if not dataset.exists():
                            import sys

                            dataset = (
                                Path(sys.prefix)
                                / "share/experienceguard/datasets/public_failures"
                            )
                        packet = public_input(dataset, body["public_case_id"])
                        if (
                            body.get("repository") != packet["repository"]
                            or body.get("log") != packet["log"]
                        ):
                            raise ValueError(
                                "public input changed; import it as a user repository instead"
                            )
                        repo, log = packet["repository"], packet["log"]
                        provenance, public_url = (
                            "public_repository",
                            packet["source_url"],
                        )
                    elif body.get("fixture"):
                        f = next(
                            x
                            for x in read_json(app.data_dir / "fixtures.json")
                            if x["name"] == body["fixture"]
                        )
                        repo = f["repository"]
                        log = body.get("log") or Path(f["log_file"]).read_text(
                            encoding="utf8"
                        )
                    else:
                        repo = body["repository"]
                        log = body["log"]
                    return self.respond(
                        app.diagnose(
                            repo,
                            log,
                            strategy=body.get("strategy", "B4"),
                            provenance=provenance,
                            public_url=public_url,
                        )
                    )
                if self.path == "/api/verify":
                    if set(body) - {
                        "case_id",
                        "proposal_id",
                        "actor",
                        "description",
                        "manual",
                        "supersedes",
                        "agent_run_id",
                    }:
                        raise ValueError("unsupported verification fields")
                    return self.respond(app.verify_and_save(**body))
                return self.respond({"error": "not found"}, 404)
            except Exception as exc:
                self.respond({"error": type(exc).__name__ + ": " + str(exc)}, 400)

        def log_message(self, *args):
            return

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(app, port=8766):
    server = create_server(app, port)
    print(
        f"ExperienceGuard: http://127.0.0.1:{server.server_port} · Evidence default; Agent requires reviewed authorization",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
