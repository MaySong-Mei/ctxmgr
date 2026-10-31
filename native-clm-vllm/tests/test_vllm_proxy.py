"""CPU-only transport checks; these do not claim a real vLLM/GPU run."""

import copy
import http.client
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ctxmgr.harness.clm_vllm_proxy import (
    GatewayServer,
    ReasoningAliasError,
    normalize_response,
    upstream_port,
)


def completion(message):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


class NormalizationTests(unittest.TestCase):
    def test_alias_is_renamed_once_without_content_or_usage_changes(self):
        message = {
            "role": "assistant",
            "reasoning": "Think once.",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"printf OK"}',
                    },
                }
            ],
        }
        original = completion(message)
        normalized = json.loads(normalize_response(json.dumps(original).encode()))
        expected = copy.deepcopy(original)
        expected["choices"][0]["message"]["reasoning_content"] = "Think once."
        del expected["choices"][0]["message"]["reasoning"]
        self.assertEqual(normalized, expected)
        self.assertEqual(
            json.loads(normalize_response(json.dumps(normalized).encode())),
            normalized,
        )

    def test_identical_aliases_are_deduplicated(self):
        body = completion({"reasoning": "same", "reasoning_content": "same", "content": "OK"})
        actual = json.loads(normalize_response(json.dumps(body).encode()))
        self.assertEqual(
            actual["choices"][0]["message"],
            {
                "reasoning_content": "same",
                "content": "OK",
            },
        )

    def test_conflicting_aliases_fail_closed(self):
        body = completion({"reasoning": "new", "reasoning_content": "different"})
        with self.assertRaises(ReasoningAliasError):
            normalize_response(json.dumps(body).encode())

    def test_null_current_field_does_not_erase_legacy_reasoning(self):
        body = completion({"reasoning": None, "reasoning_content": "legacy"})
        actual = json.loads(normalize_response(json.dumps(body).encode()))
        self.assertEqual(actual["choices"][0]["message"], {"reasoning_content": "legacy"})

    def test_invalid_responses_fail_closed(self):
        for body in (
            b"not-json",
            b"{}",
            b'{"choices":[null]}',
            b'{"choices":NaN}',
            json.dumps(completion({"reasoning": ["bad"]})).encode(),
        ):
            with self.subTest(body=body), self.assertRaises(ReasoningAliasError):
                normalize_response(body)

    def test_only_literal_loopback_origin_is_accepted(self):
        self.assertEqual(upstream_port("http://127.0.0.1:8123/"), 8123)
        for target in (
            "https://127.0.0.1:8123",
            "http://localhost:8123",
            "http://0.0.0.0:8123",
            "http://example.org:8123",
            "http://127.0.0.1:8123/v1",
            "http://127.0.0.1",
            "http://user:secret@127.0.0.1:8123",
            "http://127.0.0.1:8123?x=1",
            "http://127.0.0.1:8123#fragment",
            "http://127.0.0.1:99999",
        ):
            with self.subTest(target=target), self.assertRaises(ValueError):
                upstream_port(target)


class SyntheticUpstream(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _reply(self, status, body, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Upstream-Test", "preserved")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._reply(200, b"healthy", "text/plain")
        else:
            self._reply(200, b'{"data":[{"id":"synthetic-model"}]}')

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.requests.append(body)
        status, response = self.server.replies.pop(0)
        self._reply(status, response)


class GatewayRoundtripTests(unittest.TestCase):
    def setUp(self):
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), SyntheticUpstream)
        self.upstream.requests = []
        self.upstream.replies = []
        self.upstream_thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.upstream_thread.start()
        self.gateway = GatewayServer(0, f"http://127.0.0.1:{self.upstream.server_port}", timeout=5)
        self.gateway_thread = threading.Thread(target=self.gateway.serve_forever, daemon=True)
        self.gateway_thread.start()

    def tearDown(self):
        for server, thread in (
            (self.gateway, self.gateway_thread),
            (self.upstream, self.upstream_thread),
        ):
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def request(self, method, path, body=None):
        client = http.client.HTTPConnection("127.0.0.1", self.gateway.server_port, timeout=5)
        try:
            client.request(method, path, body=body, headers={"Content-Type": "application/json"})
            response = client.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            client.close()

    def test_synthetic_tool_roundtrip_preserves_parameters_and_history(self):
        tool_message = {
            "role": "assistant",
            "content": None,
            "reasoning": "Need a tool.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"printf OK"}',
                    },
                }
            ],
        }
        self.upstream.replies = [
            (200, json.dumps(completion(tool_message)).encode()),
            (
                200,
                json.dumps(
                    completion({"role": "assistant", "reasoning": "Done.", "content": "OK"})
                ).encode(),
            ),
        ]
        payload = {
            "model": "synthetic-model",
            "messages": [{"role": "user", "content": "Use tool"}],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 8192,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        first_body = json.dumps(payload, indent=2).encode()
        status, headers, body = self.request("POST", "/v1/chat/completions", first_body)
        self.assertEqual(status, 200)
        self.assertEqual(int(headers["Content-Length"]), len(body))
        first = json.loads(body)
        replay = first["choices"][0]["message"]
        self.assertNotIn("reasoning", replay)
        self.assertEqual(replay["reasoning_content"], "Need a tool.")
        payload["messages"].extend(
            [replay, {"role": "tool", "tool_call_id": "call-1", "content": "OK (synthetic result)"}]
        )
        second_body = json.dumps(payload).encode()
        status, _, body = self.request("POST", "/v1/chat/completions", second_body)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "OK")
        self.assertEqual(self.upstream.requests, [first_body, second_body])

    def test_upstream_http_error_status_and_body_are_not_rewritten(self):
        original = b'{"error": {"message": "context too long", "reasoning": "unchanged"}}\n'
        self.upstream.replies = [(400, original)]
        status, headers, body = self.request("POST", "/v1/chat/completions", b"{}")
        self.assertEqual((status, body), (400, original))
        self.assertEqual(headers["X-Upstream-Test"], "preserved")
        self.assertEqual(int(headers["Content-Length"]), len(original))

    def test_conflicting_upstream_aliases_produce_502(self):
        original = completion({"reasoning": "secret A", "reasoning_content": "secret B"})
        self.upstream.replies = [(200, json.dumps(original).encode())]
        status, _, body = self.request("POST", "/v1/chat/completions", b"{}")
        self.assertEqual(status, 502)
        self.assertNotIn(b"secret", body)

    def test_streaming_and_unsupported_routes_never_reach_upstream(self):
        status, _, _ = self.request("POST", "/v1/chat/completions", b'{"stream":true}')
        self.assertEqual(status, 400)
        status, _, _ = self.request("POST", "/v1/other", b"{}")
        self.assertEqual(status, 404)
        self.assertEqual(self.upstream.requests, [])

    def test_health_models_and_loopback_binding(self):
        self.assertEqual(self.gateway.server_address[0], "127.0.0.1")
        status, _, body = self.request("GET", "/health")
        self.assertEqual((status, body), (200, b"healthy"))
        status, _, body = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["data"][0]["id"], "synthetic-model")


if __name__ == "__main__":
    unittest.main()
