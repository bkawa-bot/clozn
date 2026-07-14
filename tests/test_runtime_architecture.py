"""Dependency-free contract tests for the post-beta runtime boundary."""
from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from clozn.cli import engine_process, runtime_process
from clozn.memory import mode as memory_mode
from clozn.runs import store
from clozn.server import app
from clozn.server import generation_gateway
from clozn.server import sse
from clozn.server.request_gate import RequestGate
from clozn.server.routes import health
from clozn.lab import app as lab_app


class FakeProcess:
    _next_pid = 2000

    def __init__(self, code=None):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.code = code
        self.terminated = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = -15

    def kill(self):
        self.terminated = True
        self.code = -9

    def wait(self, timeout=None):
        return self.code


class SequencedProcess(FakeProcess):
    def __init__(self, codes):
        super().__init__()
        self.codes = list(codes)

    def poll(self):
        if len(self.codes) > 1:
            return self.codes.pop(0)
        return self.codes[0]


class FakeResponse:
    def __init__(self, lines=(), payload=b"{}", status=200):
        self.lines = list(lines)
        self.payload = payload
        self.status = status
        self.closed = False

    def __iter__(self):
        return iter(self.lines)

    def read(self):
        return self.payload

    def close(self):
        self.closed = True


class CaptureHandler:
    def __init__(self):
        self.code = None
        self.headers = {}
        self.sent_headers = []
        self.wfile = io.BytesIO()
        self.json = None

    def send_response(self, code):
        self.code = code

    def send_header(self, key, value):
        self.sent_headers.append((key, value))

    def end_headers(self):
        pass

    def _json(self, code, value, extra_headers=None):
        self.code, self.json = code, value

    def _send(self, code, body, ctype, extra_headers=None):
        self.code = code
        self.wfile.write(body if isinstance(body, bytes) else body.encode("utf-8"))

    def _log_run(self, *_args, **_kwargs):
        return "run_test"


def raw_gateway_request(method: str, *, path="/jlens", body=b"", headers=None):
    """Drive the real gateway handler without opening a socket."""
    handler_type = app.make_handler()
    handler = object.__new__(handler_type)
    handler.path = path
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    handler.headers = {
        "Content-Length": str(len(body)),
        "User-Agent": "unittest",
        **(headers or {}),
    }
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.command = method
    handler.close_connection = False
    getattr(handler, f"do_{method}")()
    head, _, payload = handler.wfile.getvalue().partition(b"\r\n\r\n")
    return head.decode("latin-1"), payload, handler


class RuntimeBoundaryTests(unittest.TestCase):
    def test_force_cpu_refuses_a_gpu_only_build(self):
        with tempfile.TemporaryDirectory(prefix="clozn-engine-select-") as root:
            gpu_root = os.path.join(root, "build-gpu")
            os.makedirs(gpu_root)
            open(os.path.join(gpu_root, "cloze-server.exe"), "wb").close()
            with mock.patch.object(engine_process, "ENGINE_CORE", root):
                with self.assertRaisesRegex(Exception, "no CPU engine build"):
                    engine_process.find_engine(prefer_gpu=False)

    def test_force_cpu_selects_the_cpu_build(self):
        with tempfile.TemporaryDirectory(prefix="clozn-engine-select-") as root:
            for build in ("build-gpu", "build-serve"):
                build_root = os.path.join(root, build)
                os.makedirs(build_root)
                open(os.path.join(build_root, "cloze-server.exe"), "wb").close()
            with mock.patch.object(engine_process, "ENGINE_CORE", root):
                executable, _dlls, gpu = engine_process.find_engine(prefer_gpu=False)
            self.assertFalse(gpu)
            self.assertIn("build-serve", executable)

    def test_runtime_config_copies_and_freezes_flags(self):
        flags = {"mask": 7}
        config = runtime_process.RuntimeConfig(model="m.gguf", public_port=8080, flags=flags)
        flags["mask"] = 8
        self.assertEqual(config.flags["mask"], 7)
        with self.assertRaises(TypeError):
            config.flags["mask"] = 9

    def test_spawn_runtime_starts_private_worker_before_gateway(self):
        originals = (runtime_process.spawn_engine, runtime_process.subprocess.Popen,
                     runtime_process.gateway_health, runtime_process.port_is_open)
        calls = []
        worker = FakeProcess()
        gateway = FakeProcess()

        def fake_spawn(model, port, flags, **kwargs):
            calls.append(("worker", port))
            return worker, {"status": "ok", "mode": "autoregressive"}, True

        def fake_popen(command, **kwargs):
            calls.append(("gateway", command, kwargs))
            return gateway

        runtime_process.spawn_engine = fake_spawn
        runtime_process.subprocess.Popen = fake_popen
        runtime_process.gateway_health = lambda port: {"status": "ok"}
        runtime_process.port_is_open = lambda port: False
        try:
            stack = runtime_process.spawn_runtime(runtime_process.RuntimeConfig(
                model="m.gguf", public_port=8123, worker_port=8456
            ))
        finally:
            (runtime_process.spawn_engine, runtime_process.subprocess.Popen,
             runtime_process.gateway_health, runtime_process.port_is_open) = originals

        self.assertEqual([call[0] for call in calls], ["worker", "gateway"])
        gateway_call = calls[1]
        self.assertEqual(gateway_call[2]["env"]["CLOZN_ENGINE_PORT"], "8456")
        self.assertEqual(gateway_call[2]["env"]["CLOZN_RUNTIME_KIND"], "product")
        self.assertNotIn("--substrate", gateway_call[1])
        self.assertEqual(stack.public_port, 8123)
        self.assertEqual(stack.worker_port, 8456)

    def test_interrupted_worker_boot_terminates_the_child(self):
        originals = (engine_process.find_engine, engine_process.subprocess.Popen, engine_process._health)
        worker = FakeProcess()
        engine_process.find_engine = lambda prefer_gpu=True: ("fake-engine", [], False)
        engine_process.subprocess.Popen = lambda *args, **kwargs: worker
        engine_process._health = lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt())
        try:
            with self.assertRaises(KeyboardInterrupt):
                engine_process.spawn_engine("m.gguf", 9001, {}, boot_timeout=1)
        finally:
            (engine_process.find_engine, engine_process.subprocess.Popen,
             engine_process._health) = originals
        self.assertTrue(worker.terminated)

    def test_interrupted_gateway_boot_terminates_the_worker(self):
        originals = (runtime_process.spawn_engine, runtime_process.subprocess.Popen,
                     runtime_process.port_is_open)
        worker = FakeProcess()
        runtime_process.spawn_engine = lambda *args, **kwargs: (
            worker, {"status": "ok", "mode": "autoregressive"}, False
        )
        runtime_process.subprocess.Popen = lambda *args, **kwargs: (
            (_ for _ in ()).throw(KeyboardInterrupt())
        )
        runtime_process.port_is_open = lambda port: False
        try:
            with self.assertRaises(KeyboardInterrupt):
                runtime_process.spawn_runtime(runtime_process.RuntimeConfig(
                    model="m.gguf", public_port=8123, worker_port=8456
                ))
        finally:
            (runtime_process.spawn_engine, runtime_process.subprocess.Popen,
             runtime_process.port_is_open) = originals
        self.assertTrue(worker.terminated)

    def test_readiness_requires_the_one_worker(self):
        class Worker:
            def health(self):
                return {"status": "ok", "model": "m.gguf", "mode": "autoregressive"}

        old_engine, old_sub = app.ENGINE, app.SUB
        handler = CaptureHandler()
        try:
            app.ENGINE, app.SUB = Worker(), object()
            self.assertTrue(health.try_get(handler, "/readyz"))
        finally:
            app.ENGINE, app.SUB = old_engine, old_sub
        self.assertEqual(handler.code, 200)
        self.assertEqual(handler.json["status"], "ok")
        self.assertEqual(handler.json["active"], "engine")

    def test_supervisor_restarts_an_unexpected_worker_exit(self):
        dead_worker = FakeProcess(code=1)
        replacement = FakeProcess()
        gateway = SequencedProcess([None, 0])
        stack = runtime_process.RuntimeStack(
            config=runtime_process.RuntimeConfig(model="m.gguf", public_port=8080, worker_port=9000),
            worker_port=9000,
            worker=dead_worker,
            gateway=gateway,
            worker_health={"status": "ok", "mode": "autoregressive"},
            gpu=False,
        )
        original = runtime_process.spawn_engine
        runtime_process.spawn_engine = lambda *a, **k: (
            replacement, {"status": "ok", "mode": "autoregressive"}, True
        )
        restarts = []
        try:
            code = stack.wait(on_worker_restart=lambda current: restarts.append(current.worker.pid),
                              poll_interval=0)
        finally:
            runtime_process.spawn_engine = original
        self.assertEqual(code, 0)
        self.assertIs(stack.worker, replacement)
        self.assertEqual(restarts, [replacement.pid])

    def test_product_substrate_switch_is_gone(self):
        handler = CaptureHandler()
        self.assertTrue(health.try_post(handler, "/substrate", {"name": "qwen"}))
        self.assertEqual(handler.code, 410)

    def test_product_memory_is_prompt_only_and_never_loads_prefix_artifacts(self):
        old_kind = app.RUNTIME_KIND
        app.RUNTIME_KIND = "product"
        try:
            with mock.patch.dict(os.environ, {"CLOZN_RUNTIME_KIND": "product"}):
                head, payload, _ = raw_gateway_request("GET", path="/memory/mode")
                self.assertIn(" 200 ", head)
                self.assertEqual(json.loads(payload), {"mode": "prompt", "modes": ["prompt"]})

                body = json.dumps({"mode": "internalized"}).encode("utf-8")
                head, payload, _ = raw_gateway_request("POST", path="/memory/mode", body=body)
                self.assertIn(" 410 ", head)
                self.assertIn("lab-only", json.loads(payload)["error"])
                self.assertEqual(memory_mode.get_mode(), "prompt")
                self.assertFalse(memory_mode.set_mode("internalized"))
            self.assertFalse(hasattr(app, "_disk_memory"))
        finally:
            app.RUNTIME_KIND = old_kind

    def test_lab_retains_internalized_memory_experiments(self):
        old_kind = app.RUNTIME_KIND
        old_path = memory_mode.SETTINGS_PATH
        temp = tempfile.TemporaryDirectory(prefix="clozn-lab-memory-")
        app.RUNTIME_KIND = "lab"
        memory_mode.SETTINGS_PATH = os.path.join(temp.name, "settings.json")
        try:
            with mock.patch.dict(os.environ, {"CLOZN_RUNTIME_KIND": "lab"}):
                self.assertTrue(memory_mode.set_mode("internalized"))
                self.assertEqual(memory_mode.get_mode(), "internalized")
                head, payload, _ = raw_gateway_request("GET", path="/memory/mode")
                self.assertIn(" 200 ", head)
                self.assertEqual(set(json.loads(payload)["modes"]), {"prompt", "internalized"})
        finally:
            memory_mode.SETTINGS_PATH = old_path
            app.RUNTIME_KIND = old_kind
            temp.cleanup()

    def test_openai_stream_filters_native_worker_events(self):
        frames = [
            b'data: {"type":"tokens_committed","items":[{"piece":"hel"}]}\n',
            b'data: {"type":"step_lens","pieces":["x"]}\n',
            b'data: {"type":"tokens_committed","items":[{"piece":"lo"}]}\n',
            b'data: {"type":"gen_finished","reason":"eos"}\n',
            b'data: [DONE]\n',
        ]
        response = FakeResponse(frames)
        original = generation_gateway._request
        generation_gateway._request = lambda body: response
        handler = CaptureHandler()
        try:
            generation_gateway.openai_completion(handler, {"prompt": "hi", "stream": True, "model": "m"})
        finally:
            generation_gateway._request = original
        wire = handler.wfile.getvalue().decode("utf-8")
        self.assertIn('"text": "hel"', wire)
        self.assertIn('"text": "lo"', wire)
        self.assertNotIn("tokens_committed", wire)
        self.assertNotIn("step_lens", wire)
        self.assertTrue(wire.endswith("data: [DONE]\n\n"))
        self.assertTrue(response.closed)

    def test_native_stream_preserves_clozn_events(self):
        frame = b'data: {"type":"tokens_committed","items":[{"piece":"x"}]}\n\n'
        response = FakeResponse([frame])
        original = generation_gateway._request
        generation_gateway._request = lambda body: response
        handler = CaptureHandler()
        try:
            generation_gateway.native_completion(handler, {"prompt": "hi", "stream": True})
        finally:
            generation_gateway._request = original
        self.assertEqual(handler.wfile.getvalue(), frame)
        self.assertTrue(response.closed)

    def test_openai_chat_lens_stays_inside_standard_chunk_envelopes(self):
        class LensSubstrate:
            def chat_stream(self, messages, max_new, mem_out=None, lens=None, on_frame=None):
                on_frame({"type": "jlens_live", "pieces": ["ready"]})
                yield "ready"

            def last_finish_reason(self):
                return "stop"

            def last_stream_trace(self):
                return []

        old_sub = app.SUB
        handler = CaptureHandler()
        app.SUB = LensSubstrate()
        try:
            sse.sse_chat(handler, [{"role": "user", "content": "hi"}], 8, "m",
                         lens={"layer": 8})
        finally:
            app.SUB = old_sub
        frames = []
        for line in handler.wfile.getvalue().decode("utf-8").splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                frames.append(json.loads(line[6:]))
        self.assertTrue(frames)
        self.assertTrue(any("clozn_lens" in frame for frame in frames))
        self.assertTrue(all(frame.get("object") == "chat.completion.chunk" for frame in frames))
        self.assertTrue(all(isinstance(frame.get("created"), int) for frame in frames))
        self.assertTrue(all(isinstance(frame.get("choices"), list) and frame["choices"] for frame in frames))

    def test_lab_workbench_refuses_product_apis(self):
        handler_type = lab_app.make_lab_handler()
        handler = object.__new__(handler_type)
        handler.path = "/v1/chat/completions"
        captured = {}
        handler._json = lambda code, value: captured.update(code=code, value=value)
        handler.do_POST()
        self.assertEqual(captured["code"], 404)
        self.assertIn("does not expose", captured["value"]["error"])


class GatewayHTTPPolicyTests(unittest.TestCase):
    def test_invalid_and_negative_content_lengths_are_rejected(self):
        for value in ("nope", "-1"):
            head, payload, handler = raw_gateway_request(
                "POST", body=b"{}", headers={"Content-Length": value}
            )
            self.assertIn(" 400 ", head)
            self.assertIn("invalid Content-Length", payload.decode("utf-8"))
            self.assertTrue(handler.close_connection)

    def test_request_body_limit_is_checked_before_reading(self):
        with mock.patch.dict(os.environ, {"CLOZN_MAX_REQUEST_BYTES": "4"}):
            head, payload, handler = raw_gateway_request(
                "POST", body=b"{}", headers={"Content-Length": "5"}
            )
        self.assertIn(" 413 ", head)
        self.assertIn("4-byte limit", payload.decode("utf-8"))
        self.assertEqual(handler.rfile.tell(), 0)
        self.assertTrue(handler.close_connection)

    def test_chunked_request_body_fails_closed(self):
        head, payload, handler = raw_gateway_request(
            "POST", body=b"{}", headers={"Transfer-Encoding": "chunked"}
        )
        self.assertIn(" 501 ", head)
        self.assertIn("chunked request bodies", payload.decode("utf-8"))
        self.assertTrue(handler.close_connection)

    def test_cors_preflight_allows_loopback_and_echoes_requested_headers(self):
        head, _, _ = raw_gateway_request("OPTIONS", headers={
            "Origin": "http://127.0.0.1:3000",
            "Access-Control-Request-Headers": "authorization, content-type",
        })
        self.assertIn(" 204 ", head)
        self.assertIn("Access-Control-Allow-Origin: http://127.0.0.1:3000", head)
        self.assertIn("Access-Control-Allow-Headers: authorization, content-type", head)

    def test_cors_preflight_rejects_untrusted_web_origins(self):
        head, _, _ = raw_gateway_request(
            "OPTIONS", headers={"Origin": "https://attacker.example"}
        )
        self.assertIn(" 403 ", head)
        self.assertNotIn("Access-Control-Allow-Origin", head)

    def test_cors_operator_can_add_an_exact_origin(self):
        with mock.patch.dict(os.environ, {"CLOZN_ORIGINS": "https://trusted.example"}):
            head, _, _ = raw_gateway_request(
                "OPTIONS", headers={"Origin": "https://trusted.example"}
            )
        self.assertIn(" 204 ", head)
        self.assertIn("Access-Control-Allow-Origin: https://trusted.example", head)

    def test_untrusted_origin_is_rejected_before_get_or_post_dispatch(self):
        for method, path in (("GET", "/healthz"), ("POST", "/capture/tier")):
            head, payload, _ = raw_gateway_request(
                method, path=path, body=b"{}", headers={"Origin": "https://attacker.example"}
            )
            self.assertIn(" 403 ", head)
            self.assertIn("browser origin is not allowed", payload.decode("utf-8"))

    def test_loopback_origin_reaches_normal_route_and_is_echoed(self):
        head, payload, _ = raw_gateway_request(
            "GET", path="/healthz", headers={"Origin": "http://localhost:3000"}
        )
        self.assertIn(" 200 ", head)
        self.assertIn("Access-Control-Allow-Origin: http://localhost:3000", head)
        self.assertEqual(json.loads(payload)["status"], "ok")


class RequestGateTests(unittest.TestCase):
    def test_gate_serializes_and_bounds_admitted_requests(self):
        gate = RequestGate(capacity=2, wait_timeout=1)
        self.assertIsNone(gate.acquire())
        result = []

        def wait_for_turn():
            result.append(gate.acquire())

        thread = threading.Thread(target=wait_for_turn)
        thread.start()
        deadline = time.monotonic() + 1
        while gate.snapshot()["waiting"] != 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(gate.snapshot()["waiting"], 1)
        self.assertEqual(gate.acquire(), "full")
        gate.release()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [None])
        self.assertEqual(gate.snapshot()["active"], 1)
        gate.release()
        self.assertEqual(gate.snapshot()["active"], 0)

    def test_gate_timeout_releases_its_admission_slot(self):
        gate = RequestGate(capacity=2, wait_timeout=0.01)
        self.assertIsNone(gate.acquire())
        self.assertEqual(gate.acquire(), "timeout")
        gate.release()
        self.assertIsNone(gate.acquire())
        gate.release()


class SQLiteRunStoreTests(unittest.TestCase):
    def setUp(self):
        self.old_dir = store.RUNS_DIR
        self.temp = tempfile.TemporaryDirectory(prefix="clozn-sqlite-test-")
        store.RUNS_DIR = self.temp.name

    def tearDown(self):
        store.RUNS_DIR = self.old_dir
        self.temp.cleanup()

    def test_sqlite_is_authoritative_and_trace_is_content_addressed(self):
        rid = store.record(
            source="cli",
            model="m",
            substrate="engine",
            messages=[{"role": "user", "content": "hi"}],
            response="hello",
            trace={"tokens": ["hello"], "confidence": [0.9]},
        )
        self.assertIsNotNone(rid)
        self.assertTrue(os.path.isfile(os.path.join(self.temp.name, "runs.sqlite3")))
        self.assertEqual(store.get_run(rid)["trace"]["tokens"], ["hello"])
        blobs = []
        for root, _, files in os.walk(os.path.join(self.temp.name, "blobs")):
            blobs.extend(os.path.join(root, name) for name in files if name.endswith(".json"))
        self.assertEqual(len(blobs), 1)
        self.assertFalse(glob_run_json(self.temp.name))

    def test_legacy_json_requires_explicit_import(self):
        legacy = tempfile.TemporaryDirectory(prefix="clozn-json-test-")
        try:
            path = os.path.join(legacy.name, "run_legacy_one.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"id": "run_legacy_one", "source": "legacy", "response": "old"}, handle)
            self.assertIsNone(store.get_run("run_legacy_one"))
            result = store.import_json_dir(legacy.name)
            self.assertEqual(result["imported"], 1)
            self.assertEqual(store.get_run("run_legacy_one")["response"], "old")
        finally:
            legacy.cleanup()


def glob_run_json(root: str) -> bool:
    return any(name.startswith("run_") and name.endswith(".json") for name in os.listdir(root))


if __name__ == "__main__":
    unittest.main()
