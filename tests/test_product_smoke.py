"""Dependency-free tests for the managed product-runtime smoke harness."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest

from clozn.cli.commands import smoke
from clozn.runs import store
from clozn.server import app as gateway_app
from clozn.server import static as static_routes


class SmokeGatewayHandler(BaseHTTPRequestHandler):
    run_id = ""

    def log_message(self, *_args):
        pass

    def _send(self, code: int, body, content_type="application/json"):
        raw = body if isinstance(body, bytes) else (
            body.encode("utf-8") if isinstance(body, str) else json.dumps(body).encode("utf-8")
        )
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/healthz":
            self._send(200, {"status": "ok", "service": "clozn"})
        elif self.path == "/readyz":
            self._send(200, {"status": "ok", "active": "engine", "model": "fake.gguf",
                             "mode": "autoregressive", "worker": {"status": "ok"},
                             "queue": {"active": 0, "waiting": 0, "capacity": 32}})
        elif self.path == "/":
            self._send(200, "<!doctype html><title>Clozn</title>", "text/html; charset=utf-8")
        elif self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})
        elif self.path == f"/runs/{self.run_id}":
            self._send(200, store.get_run(self.run_id))
        else:
            self._send(404, {"error": self.path})

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(size) or b"{}")
        if self.path == "/v1/chat/completions":
            if body.get("stream"):
                frames = (
                    'data: {"id":"chatcmpl-fake","object":"chat.completion.chunk",'
                    '"created":1,"model":"fake","choices":[{"index":0,"delta":'
                    '{"role":"assistant"},"finish_reason":null}]}\n\n'
                    'data: {"id":"chatcmpl-fake","object":"chat.completion.chunk",'
                    '"created":1,"model":"fake","choices":[{"index":0,"delta":'
                    '{"content":"ready"},"finish_reason":null}]}\n\n'
                    'data: {"id":"chatcmpl-fake","object":"chat.completion.chunk",'
                    '"created":1,"model":"fake","choices":[{"index":0,"delta":{},'
                    '"finish_reason":"stop"}]}\n\n'
                    "data: [DONE]\n\n"
                )
                self._send(200, frames, "text/event-stream")
                return
            self._send(200, {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": body.get("model", "fake"),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ready"}}],
                "clozn_run_id": self.run_id,
            })
        elif self.path == f"/runs/{self.run_id}/explain":
            self._send(200, {"run_id": self.run_id, "confidence": {"available": True}})
        elif self.path == "/v1/completions":
            frames = (
                'data: {"id":"cmpl-fake","object":"text_completion","choices":'
                '[{"text":"ready","index":0,"finish_reason":null}]}\n\n'
                'data: {"id":"cmpl-fake","object":"text_completion","choices":'
                '[{"text":"","index":0,"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            )
            self._send(200, frames, "text/event-stream")
        elif self.path == "/api/clozn/generate":
            frames = (
                'data: {"type":"tokens_committed","items":[{"piece":"ready"}]}\n\n'
                'data: {"type":"gen_finished","reason":"eos"}\n\n'
                "data: [DONE]\n\n"
            )
            self._send(200, frames, "text/event-stream")
        else:
            self._send(404, {"error": self.path})


class FakeWorkerHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(size) or b"{}")
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        if body.get("stream"):
            raw = (
                'data: {"type":"tokens_committed","items":[{"piece":"ready","id":1,'
                '"pos":0,"conf":0.99}]}\n\n'
                'data: {"type":"step_lens","positions":[0],"pieces":["ready"],'
                '"ids":[1],"probs":[0.99]}\n\n'
                'data: {"type":"gen_finished","reason":"eos"}\n\n'
                'data: {"id":"cmpl-worker","object":"text_completion","choices":'
                '[{"text":"ready","index":0,"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        raw = json.dumps({
            "id": "cmpl-worker",
            "object": "text_completion",
            "choices": [{"text": "ready", "index": 0, "finish_reason": "stop"}],
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class FakeEngine:
    timeout = 5

    def __init__(self, base):
        self.base = base

    def health(self):
        return {"status": "ok", "model": "fake.gguf", "mode": "autoregressive"}


class FakeSteer:
    def active(self):
        return {}


class FakeProductSub:
    steer = FakeSteer()
    brain = None

    def chat(self, messages, max_new=256, sample=False, trace_out=None, mem_out=None, **_kwargs):
        if trace_out is not None:
            trace_out.append({"pos": 0, "token_id": 1, "piece": "ready", "prob": 0.99,
                              "alts": [{"token_id": 2, "piece": "set", "prob": 0.01}]})
        if mem_out is not None:
            mem_out.update(mode="prompt", applied=[], gate=0.0, final_prompt="fake prompt")
        return "ready"

    def last_finish_reason(self):
        return "stop"

    def chat_stream(self, messages, max_new=256, mem_out=None, **_kwargs):
        if mem_out is not None:
            mem_out.update(mode="prompt", applied=[], gate=0.0, final_prompt="fake prompt")
        yield "ready"

    def last_stream_trace(self):
        return [{"pos": 0, "token_id": 1, "piece": "ready", "prob": 0.99}]

    def run_meta(self):
        return {"model_file": "fake.gguf", "mode": "autoregressive"}


class ProductSmokeTests(unittest.TestCase):
    def setUp(self):
        self.old_runs = store.RUNS_DIR
        self.temp = tempfile.TemporaryDirectory(prefix="clozn-smoke-test-")
        store.RUNS_DIR = self.temp.name
        rid = store.record(
            source="openai_api",
            model="fake",
            substrate="engine",
            messages=[{"role": "user", "content": "ready?"}],
            response="ready",
            trace={"tokens": ["ready"], "confidence": [0.99]},
        )
        SmokeGatewayHandler.run_id = rid
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), SmokeGatewayHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        store.RUNS_DIR = self.old_runs
        self.temp.cleanup()

    def test_exercise_crosses_http_and_checks_sqlite_blob(self):
        report = smoke.Report()
        base = f"http://127.0.0.1:{self.server.server_address[1]}"
        rid = smoke._exercise(base, 5, report)
        self.assertEqual(rid, SmokeGatewayHandler.run_id)
        self.assertTrue(report.ok, report.render())
        names = {check.name for check in report.checks}
        self.assertIn("OpenAI completion stream contains only standard chunks", names)
        self.assertIn("native stream preserves typed Clozn events", names)
        self.assertIn("trace stored as a content-addressed blob", names)

    def test_exercise_the_real_gateway_against_a_fake_private_worker(self):
        worker = ThreadingHTTPServer(("127.0.0.1", 0), FakeWorkerHandler)
        worker_thread = threading.Thread(target=worker.serve_forever, daemon=True)
        worker_thread.start()

        studio = os.path.join(self.temp.name, "studio")
        os.makedirs(os.path.join(studio, "heavn"), exist_ok=True)
        with open(os.path.join(studio, "heavn", "index.html"), "w", encoding="utf-8") as handle:
            handle.write("<!doctype html><title>Clozn</title>")

        old = (gateway_app.ENGINE, gateway_app.SUB, gateway_app.SUBNAME, static_routes.DEMO)
        gateway_app.ENGINE = FakeEngine(f"http://127.0.0.1:{worker.server_address[1]}")
        gateway_app.SUB = FakeProductSub()
        gateway_app.SUBNAME = "engine"
        static_routes.DEMO = studio
        gateway = ThreadingHTTPServer(("127.0.0.1", 0), gateway_app.make_handler())
        gateway_thread = threading.Thread(target=gateway.serve_forever, daemon=True)
        gateway_thread.start()
        try:
            report = smoke.Report()
            base = f"http://127.0.0.1:{gateway.server_address[1]}"
            rid = smoke._exercise(base, 5, report)
            self.assertTrue(rid)
            self.assertTrue(report.ok, report.render())
            self.assertEqual(store.get_run(rid)["response"], "ready")
        finally:
            gateway.shutdown()
            gateway.server_close()
            gateway_thread.join(timeout=2)
            worker.shutdown()
            worker.server_close()
            worker_thread.join(timeout=2)
            gateway_app.ENGINE, gateway_app.SUB, gateway_app.SUBNAME, static_routes.DEMO = old

    def test_managed_smoke_owns_restarts_and_cleans_the_real_process_tree(self):
        worker_path = os.path.join(self.temp.name, "fake-cloze-server")
        worker_source = textwrap.dedent(r'''
            #!/usr/bin/env python3
            import argparse
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
            import json

            parser = argparse.ArgumentParser()
            parser.add_argument("model")
            parser.add_argument("--port", type=int, required=True)
            parser.add_argument("--host", default="127.0.0.1")
            parser.add_argument("--gpu-layers")
            parser.add_argument("--mask-token")
            parser.add_argument("--eos")
            args, _ = parser.parse_known_args()

            class Server(ThreadingHTTPServer):
                allow_reuse_address = True

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    pass

                def send_json(self, code, value):
                    raw = json.dumps(value).encode("utf-8")
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)

                def do_GET(self):
                    if self.path == "/health":
                        self.send_json(200, {"status": "ok", "model": args.model,
                                             "mode": "autoregressive", "n_ctx": 2048})
                    else:
                        self.send_json(404, {"error": self.path})

                def do_POST(self):
                    size = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(size) or b"{}")
                    if self.path == "/apply_template":
                        text = "\n".join(str(m.get("content") or "") for m in body.get("messages", []))
                        self.send_json(200, {"prompt": text + "\nassistant:"})
                        return
                    if self.path != "/v1/completions":
                        self.send_json(404, {"error": self.path})
                        return
                    if not body.get("stream"):
                        self.send_json(200, {"id": "cmpl-fake", "object": "text_completion",
                                            "choices": [{"text": "ready", "index": 0,
                                                         "finish_reason": "stop"}]})
                        return
                    raw = (
                        'data: {"type":"tokens_committed","items":[{"piece":"ready",'
                        '"id":1,"pos":0,"conf":0.99}]}\n\n'
                        'data: {"type":"step_lens","positions":[0],"pieces":["ready"],'
                        '"ids":[1],"probs":[0.99]}\n\n'
                        'data: {"type":"gen_finished","reason":"eos"}\n\n'
                        'data: {"id":"cmpl-fake","object":"text_completion","choices":'
                        '[{"text":"ready","index":0,"finish_reason":"stop"}]}\n\n'
                        'data: [DONE]\n\n'
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)

            Server((args.host, args.port), Handler).serve_forever()
        ''').lstrip()
        with open(worker_path, "w", encoding="utf-8") as handle:
            handle.write(worker_source)
        os.chmod(worker_path, 0o755)

        studio = os.path.join(self.temp.name, "managed-studio")
        os.makedirs(os.path.join(studio, "heavn"), exist_ok=True)
        with open(os.path.join(studio, "heavn", "index.html"), "w", encoding="utf-8") as handle:
            handle.write("<!doctype html><title>Clozn</title>")
        model = os.path.join(self.temp.name, "fake.gguf")
        with open(model, "wb") as handle:
            handle.write(b"not-a-real-gguf")
        home = os.path.join(self.temp.name, "managed-home")
        os.makedirs(home, exist_ok=True)

        env = dict(os.environ)
        env.update({
            "HOME": home,
            "USERPROFILE": home,
            "CLOZN_ENGINE_BIN": worker_path,
            "CLOZN_STUDIO_DIR": studio,
            "NO_COLOR": "1",
        })
        result = subprocess.run(
            [sys.executable, "-m", "clozn", "smoke", model, "--json",
             "--timeout", "10", "--startup-timeout", "20"],
            cwd=smoke.REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["ok"], report)
        checks = {row["name"]: row for row in report["checks"]}
        self.assertEqual(checks["worker restart recovery"]["status"], "pass")
        self.assertEqual(checks["generation succeeds after worker restart"]["status"], "pass")
        registry = os.path.join(home, ".clozn", "daemons.json")
        if os.path.isfile(registry):
            with open(registry, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {})

    def test_standard_validator_rejects_native_frame(self):
        result = smoke.SSEResult(
            status=200,
            frames=[{"type": "tokens_committed", "items": [{"piece": "oops"}]}],
            done=True,
        )
        ok, _, detail = smoke._completion_stream_text(result)
        self.assertFalse(ok)
        self.assertIn("native frame leaked", detail)

    def test_base_url_parser_rejects_paths_and_https(self):
        self.assertEqual(smoke._parse_base("127.0.0.1:8080"), ("http://127.0.0.1:8080", 8080))
        with self.assertRaises(ValueError):
            smoke._parse_base("https://127.0.0.1:8080")
        with self.assertRaises(ValueError):
            smoke._parse_base("http://127.0.0.1:8080/v1")


if __name__ == "__main__":
    unittest.main()
