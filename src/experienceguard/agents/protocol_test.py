"""Explicit loopback HTTP protocol self-test. Not a language model or task evaluation."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .providers import LocalOpenAICompatibleProvider, HTTPSettings
from .development import run_agent


class ProtocolTestProvider(LocalOpenAICompatibleProvider):
    kind = "mock_http_test"


def protocol_test(app, plan, *, actions=None, limits=None):
    documents = plan["documents"]
    if actions is None:
        actions = [
            {"tool": "list_repository", "arguments": {}},
            {
                "tool": "read_log",
                "arguments": {
                    "line_start": 1,
                    "line_end": min(200, documents["ci.log"]["lines"]),
                },
            },
        ]
        path = next((p for p in documents if p != "ci.log"), None)
        if path:
            actions.append(
                {
                    "tool": "read_file",
                    "arguments": {
                        "path": path,
                        "line_start": documents[path]["line_start"],
                        "line_end": min(
                            documents[path]["line_start"] + 19,
                            documents[path]["line_end"],
                        ),
                    },
                }
            )
        actions.extend(
            [
                {"tool": "analyze_environment", "arguments": {}},
                {"tool": "retrieve_memory", "arguments": {"query": "software failure"}},
            ]
        )
        selected = None
        if plan["memory"]:
            item = next((m for m in plan["memory"] if m["eligible"]), plan["memory"][0])
            actions.append(
                {
                    "tool": "check_memory",
                    "arguments": {"experience_id": item["experience_id"]},
                }
            )
            if item["eligible"]:
                selected = item["experience_id"]
        actions.append(
            {
                "tool": "select_memory",
                "arguments": {
                    "experience_id": selected,
                    "reason": "TEST_ONLY scripted gate exercise, not autonomous reasoning",
                },
            }
        )
        citations = [
            {"evidence_id": documents["ci.log"]["id"], "line_start": 1, "line_end": 1}
        ]
        if path:
            citations.append(
                {
                    "evidence_id": documents[path]["id"],
                    "line_start": documents[path]["line_start"],
                    "line_end": documents[path]["line_start"],
                }
            )
        actions.append(
            {
                "final": {
                    "summary": "TEST_ONLY：本机假HTTP响应验证了取证、检索和条件门控协议；这不是LLM诊断。",
                    "next_steps": ["用户应独立审阅实际日志和源文件。"],
                    "uncertainties": ["没有真实模型推理，不能推断任务成功率。"],
                    "citations": citations,
                    "memory_id": selected,
                }
            }
        )
    queue = list(actions)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            if n > 500000 or self.path != "/v1/chat/completions":
                self.send_error(400)
                return
            json.loads(self.rfile.read(n))
            action = queue.pop(0) if queue else {"tool": "unknown", "arguments": {}}
            body = json.dumps(
                {
                    "id": "TEST_ONLY",
                    "model": "scripted-http-fixture",
                    "choices": [
                        {
                            "message": {
                                "content": action
                                if isinstance(action, str)
                                else json.dumps(action, ensure_ascii=False)
                            }
                        }
                    ],
                },
                ensure_ascii=False,
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = ProtocolTestProvider(
            HTTPSettings(
                f"http://127.0.0.1:{server.server_port}/v1",
                "scripted-http-fixture",
                retries=0,
                max_tokens=1024,
            )
        )
        return run_agent(
            app, plan, provider, consent_sha=plan["plan_sha256"], limits=limits
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
