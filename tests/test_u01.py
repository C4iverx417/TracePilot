"""U01 tests. A tiny local HTTP server stands in for the real provider, so the
OpenAI-compatible code path is exercised without an API key or network."""

from __future__ import annotations

import io
import json
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from tracepilot import cli
from tracepilot.config import ConfigError, load_settings
from tracepilot.llm import (
    InvalidModelOutputError,
    ModelAuthError,
    ModelTimeoutError,
    OpenAICompatibleClient,
    build_messages,
)

FAKE_KEY = "sk-test-SECRET-should-never-print"


class FakeProvider(BaseHTTPRequestHandler):
    mode = "ok"  # ok | auth | slow | garbage
    last_body = None
    last_auth = None

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        FakeProvider.last_body = json.loads(self.rfile.read(length))
        FakeProvider.last_auth = self.headers.get("Authorization")
        if self.mode == "auth":
            self._send(401, {"error": {"message": "bad key"}})
        elif self.mode == "slow":
            time.sleep(1.0)
            self._send(200, {})
        elif self.mode == "garbage":
            self._send(200, {"unexpected": True})
        else:
            self._send(
                200,
                {
                    "model": "fake-model-001",
                    "choices": [{"message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 1},
                },
            )

    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client already gave up (timeout test)

    def log_message(self, *args):
        pass


class OpenAIClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeProvider)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def settings(self, timeout="5"):
        return load_settings(
            {
                "TRACEPILOT_PROVIDER": "openai",
                "OPENAI_API_KEY": FAKE_KEY,
                "OPENAI_BASE_URL": self.base_url,
                "TRACEPILOT_TIMEOUT_S": timeout,
            }
        )

    def test_success_parses_text_and_usage(self):
        FakeProvider.mode = "ok"
        result = OpenAICompatibleClient(self.settings()).chat(build_messages("ping"))
        self.assertEqual(result.text, "pong")
        self.assertEqual(result.model, "fake-model-001")
        self.assertEqual(result.usage.total_tokens, 13)
        self.assertEqual(result.usage.source, "provider")
        # Request shape: system + user roles, key only in the header.
        self.assertEqual([m["role"] for m in FakeProvider.last_body["messages"]], ["system", "user"])
        self.assertEqual(FakeProvider.last_auth, f"Bearer {FAKE_KEY}")
        self.assertNotIn(FAKE_KEY, json.dumps(FakeProvider.last_body))

    def test_auth_failure_does_not_leak_key(self):
        FakeProvider.mode = "auth"
        with self.assertRaises(ModelAuthError) as ctx:
            OpenAICompatibleClient(self.settings()).chat(build_messages("ping"))
        self.assertNotIn(FAKE_KEY, str(ctx.exception))

    def test_timeout(self):
        FakeProvider.mode = "slow"
        with self.assertRaises(ModelTimeoutError):
            OpenAICompatibleClient(self.settings(timeout="0.2")).chat(build_messages("ping"))

    def test_malformed_response(self):
        FakeProvider.mode = "garbage"
        with self.assertRaises(InvalidModelOutputError):
            OpenAICompatibleClient(self.settings()).chat(build_messages("ping"))


class ConfigAndCliTest(unittest.TestCase):
    def test_missing_key_is_clear_error(self):
        with self.assertRaises(ConfigError) as ctx:
            load_settings({"TRACEPILOT_PROVIDER": "openai"})
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_settings_repr_hides_key(self):
        s = load_settings({"TRACEPILOT_PROVIDER": "openai", "OPENAI_API_KEY": FAKE_KEY})
        self.assertNotIn(FAKE_KEY, repr(s))

    def run_cli(self, argv, env):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("tracepilot.cli.load_settings", lambda: load_settings(env)):
            with redirect_stdout(out), redirect_stderr(err):
                code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_cli_mock_happy_path(self):
        code, out, _ = self.run_cli(["what is RAG?"], {"TRACEPILOT_PROVIDER": "mock"})
        self.assertEqual(code, 0)
        self.assertIn("[mock]", out)
        self.assertIn("(estimated)", out)

    def test_cli_rejects_empty_question(self):
        code, _, err = self.run_cli(["   "], {"TRACEPILOT_PROVIDER": "mock"})
        self.assertEqual(code, 2)
        self.assertIn("INVALID_INPUT", err)

    def test_cli_show_request_never_prints_key(self):
        env = {"TRACEPILOT_PROVIDER": "openai", "OPENAI_API_KEY": FAKE_KEY, "OPENAI_BASE_URL": "http://127.0.0.1:9"}
        code, out, err = self.run_cli(["--show-request", "hi"], env)
        self.assertEqual(code, 1)  # nothing listens on port 9 -> MODEL_UNAVAILABLE
        self.assertIn('"messages"', out)
        self.assertNotIn(FAKE_KEY, out + err)


if __name__ == "__main__":
    unittest.main()
