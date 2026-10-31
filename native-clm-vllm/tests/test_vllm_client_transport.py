"""Exercise the real frozen clients against a synthetic loopback vLLM response.

These tests require envs/clm/.venv, but no GPU, model weights, or container.
They validate transport, replay, and accounting, not model performance.
"""

import asyncio
import importlib.metadata
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from ctxmgr.harness.clm_vllm_proxy import GatewayServer

MODEL = "Qwen3.5-27B-BF16"
THOUGHT = "SYNTHETIC_REASONING: inspect the supplied operation before using bash."
COMMAND = "printf CLM_TRANSPORT_TEST"


def response_message(*, tool=False):
    message = {"role": "assistant", "content": None if tool else "OK", "reasoning": THOUGHT}
    if tool:
        message["tool_calls"] = [
            {
                "id": "call_transport",
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": COMMAND})},
            }
        ]
    return {
        "id": "chatcmpl-transport-test",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [
            {"index": 0, "message": message, "finish_reason": "tool_calls" if tool else "stop"}
        ],
        "usage": {
            "prompt_tokens": 21,
            "completion_tokens": 13,
            "total_tokens": 34,
            "completion_tokens_details": {"reasoning_tokens": 11},
            "prompt_tokens_details": {"cached_tokens": 3},
        },
    }


class SyntheticVllm(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.requests.append(json.loads(body))
        reply = json.dumps(self.server.replies.pop(0)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)


class FrozenClientTransportTests(unittest.TestCase):
    def setUp(self):
        # Local loopback tests must never traverse an inherited network proxy.
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
        }
        environment.update(
            LITELLM_LOCAL_MODEL_COST_MAP="True",
            OPENAI_API_KEY="EMPTY",
            NO_PROXY="127.0.0.1,localhost",
        )
        self.environment = patch.dict(os.environ, environment, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.assertEqual(importlib.metadata.version("litellm"), "1.105.1")

        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), SyntheticVllm)
        self.upstream.requests = []
        self.upstream.replies = []
        self.gateway = GatewayServer(0, f"http://127.0.0.1:{self.upstream.server_port}")
        self.threads = []
        for server in (self.upstream, self.gateway):
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.threads.append(thread)
        self.base_url = f"http://127.0.0.1:{self.gateway.server_port}/v1"
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def tearDown(self):
        for server, thread in zip((self.upstream, self.gateway), self.threads):
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_native_litellm_tool_replay_snapshot_and_reasoning_accounting(self):
        from clm_harness.clm_agent.harness import ClmAgent, _extract_tool_call
        from clm_harness.context_utils.context_string import rendered_text
        from clm_harness.utils import tokens as tk

        self.upstream.replies = [response_message(tool=True), response_message()]
        agent = ClmAgent(
            logs_dir=Path(self.temp.name),
            model_name=f"openai/{MODEL}",
            api_base=self.base_url,
            max_tokens=8192,
            temperature=0.7,
            top_p=0.95,
            context_budget_tokens=32768,
            enable_thinking=True,
            send_chat_template_kwargs=True,
            emit_ctx_trajectory=False,
            cost_metric="usd",
        )
        messages = [
            {"role": "system", "content": "Synthetic transport test."},
            {"role": "user", "content": "Use the bash tool."},
        ]
        first = asyncio.run(agent._query_with_retry(f"openai/{MODEL}", messages))
        message = first.choices[0].message.model_dump()
        message["content"] = message.get("content") or ""
        action = _extract_tool_call(message)
        self.assertEqual(
            action, {"type": "command", "command": COMMAND, "tool_call_id": "call_transport"}
        )
        self.assertEqual(message["reasoning_content"], THOUGHT)
        self.assertNotIn("reasoning", message)
        self.assertEqual(rendered_text(message).count(THOUGHT), 1)
        self.assertEqual(list(tk._iter_text([message])).count(THOUGHT), 1)

        with_reasoning, method = tk.count_tokens([message])
        stripped = {key: value for key, value in message.items() if key != "reasoning_content"}
        without_reasoning, _ = tk.count_tokens([stripped])
        self.assertEqual(method, "tiktoken:o200k_base")
        self.assertEqual(
            with_reasoning - without_reasoning,
            len(tk.get_encoder().encode(THOUGHT, disallowed_special=())),
        )
        messages.extend(
            [
                message,
                {
                    "role": "tool",
                    "tool_call_id": "call_transport",
                    "content": "CLM_TRANSPORT_TEST (simulated, not executed)",
                },
            ]
        )
        tk.record(self.temp.name, 1, messages)
        snapshot = json.loads(
            (Path(self.temp.name) / "context_snapshots/turn-0001.json").read_text()
        )
        self.assertEqual(snapshot["messages"][2]["reasoning_content"], THOUGHT)
        self.assertEqual(snapshot["tokens"], tk.count_tokens(messages)[0])

        second = asyncio.run(agent._query_with_retry(f"openai/{MODEL}", messages))
        self.assertEqual(second.choices[0].message.content, "OK")
        self.assertEqual(second.choices[0].message.reasoning_content, THOUGHT)
        self.assertEqual(first.usage.completion_tokens, 13)
        self.assertEqual(first.usage.completion_tokens_details.reasoning_tokens, 11)
        self.assertEqual(len(self.upstream.requests), 2)
        for request in self.upstream.requests:
            self.assertEqual(request["model"], MODEL)
            self.assertEqual(request["temperature"], 0.7)
            self.assertEqual(request["top_p"], 0.95)
            self.assertEqual(request["max_tokens"], 8192)
            self.assertEqual(request["chat_template_kwargs"], {"enable_thinking": True})
            self.assertNotIn("seed", request)
            self.assertNotIn("top_k", request)
        replay = self.upstream.requests[1]["messages"][2]
        self.assertEqual(replay["reasoning_content"], THOUGHT)
        self.assertEqual(replay["tool_calls"][0]["id"], "call_transport")
        self.assertEqual(self.upstream.requests[1]["messages"][3]["tool_call_id"], "call_transport")

    def test_frozen_baseline_retains_content_only_behavior(self):
        import openai

        from ctxmgr.harness.agents import NoManagementAgent, Sampling

        self.upstream.replies = [response_message()]
        with openai.OpenAI(base_url=self.base_url, api_key="EMPTY") as client:
            agent = NoManagementAgent(
                client=client,
                model=MODEL,
                sampling=Sampling(temperature=0.7, top_p=0.95, max_tokens=8192, thinking=True),
            )
            agent.reset("Synthetic test.", "Wait for an operation.")
            answer = agent.on_user_turn("Answer OK.")
        self.assertEqual(answer, "OK")
        self.assertEqual(agent.messages[-1], {"role": "assistant", "content": "OK"})
        self.assertEqual(agent.calls[-1]["completion_tokens"], 13)
        self.assertEqual(agent.calls[-1]["reasoning_tokens"], 11)
        self.assertNotIn(THOUGHT, json.dumps(agent.context()))


if __name__ == "__main__":
    unittest.main()
