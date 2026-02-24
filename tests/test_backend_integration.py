import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from Thinkingmachiene import ThinkingMachine
from perception import backend as perception_backend


class _BackendHandler(BaseHTTPRequestHandler):
    response_payload = {}
    captured = {}

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else ""
        self.__class__.captured = {
            "path": self.path,
            "headers": dict(self.headers),
            "body": body,
        }
        payload = json.dumps(self.__class__.response_payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args, **kwargs):
        return


def _serve_once(response_payload):
    _BackendHandler.response_payload = response_payload
    _BackendHandler.captured = {}
    server = HTTPServer(("127.0.0.1", 0), _BackendHandler)
    thread = threading.Thread(target=server.handle_request)
    thread.daemon = True
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


def test_ollama_backend_networked_local_server(monkeypatch):
    server, thread, url = _serve_once({"response": "{\"has_color\": true}", "done": True})
    machine = ThinkingMachine()

    monkeypatch.setattr(perception_backend, "OLLAMA_URL", url)

    raw, status, done = perception_backend.call_ollama(machine, "sample prompt")

    thread.join(timeout=2)
    server.server_close()

    assert status == 200
    assert done is True
    assert "has_color" in raw
    assert "sample prompt" in _BackendHandler.captured.get("body", "")


def test_groq_backend_networked_with_mocked_secret(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {
                    "content": "{\"is_valid\": true}"
                }
            }
        ]
    }
    server, thread, url = _serve_once(payload)
    machine = ThinkingMachine()

    monkeypatch.setattr(perception_backend, "GROQ_URL", url)
    monkeypatch.setattr(perception_backend, "GROQ_API_KEY", "test_mocked_secret")
    monkeypatch.setattr(perception_backend, "GROQ_MIN_INTERVAL_SEC", 0)

    raw, status, done = perception_backend.call_groq(machine, "sample prompt")

    thread.join(timeout=2)
    server.server_close()

    assert status == 200
    assert done == "completed"
    assert "is_valid" in raw
    assert _BackendHandler.captured["headers"].get("Authorization") == "Bearer test_mocked_secret"
