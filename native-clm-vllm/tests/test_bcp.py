"""BrowseComp-Plus pieces without data, models or containers.

Agent tests need the CLM runtime (envs/clm) and are skipped elsewhere:
    PYTHONPATH=src uv run --project envs/clm python -m unittest tests.test_bcp
"""

import argparse
import base64
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ctxmgr.bcp import data, server

HAVE_CLM = importlib.util.find_spec("clm_harness") is not None


class DataTests(unittest.TestCase):
    def test_decrypt_inverts_the_benchmark_scheme(self):
        plain = "Who wrote it? Ünïcode answer 42"
        raw = plain.encode("utf-8")
        cipher = bytes(a ^ b for a, b in zip(raw, data.derive_key(data.CANARY, len(raw))))
        self.assertEqual(data.decrypt(base64.b64encode(cipher).decode()), plain)

    def test_selection_is_deterministic_and_order_free(self):
        ids = [str(i) for i in range(830)]
        first = data.select(ids, 100)
        self.assertEqual(first, data.select(list(reversed(ids)), 100))
        self.assertEqual(len(set(first)), 100)

    def test_private_file_is_owner_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.jsonl"
            data.write_private(path, "x\n")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)


class StubCorpus:
    docids = ("d1", "d2")

    def search(self, query, k=10):
        return [{"docid": "d1", "score": 1.0, "snippet": f"about {query}"}]

    def document(self, docid):
        return {"docid": docid, "text": "full text"} if docid in self.docids else None


class ServerTests(unittest.TestCase):
    def test_search_and_document_endpoints(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.handler_for(StubCorpus()))
        self.addCleanup(httpd.server_close)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(httpd.shutdown)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        with urllib.request.urlopen(f"{base}/search?q=river+delta") as response:
            self.assertEqual(json.load(response)[0]["snippet"], "about river delta")
        with urllib.request.urlopen(f"{base}/doc?docid=d2") as response:
            self.assertEqual(json.load(response)["text"], "full text")
        with self.assertRaises(urllib.error.HTTPError):
            urllib.request.urlopen(f"{base}/doc?docid=nope")


@unittest.skipUnless(HAVE_CLM, "needs the CLM runtime environment (envs/clm)")
class AgentTests(unittest.TestCase):
    def setUp(self):
        from ctxmgr.bcp import agent

        self.agent = agent
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.logs = Path(self.temp.name)

    def args(self, **extra):
        values = {
            "base_url": "http://127.0.0.1:1/v1",
            "search_url": "http://127.0.0.1:2",
            "model": "Qwen3.5-9B-BF16",
            "manager_model": None,
            "manager_base_url": None,
        }
        values.update(extra)
        return argparse.Namespace(**values)

    def build(self, condition, **extra):
        import importlib

        path, kwargs = self.agent.agent_kwargs(condition, self.args(**extra))
        module, name = path.split(":")
        cls = getattr(importlib.import_module(module), name)
        return cls(logs_dir=self.logs, model_name="openai/Qwen3.5-9B-BF16", **kwargs), kwargs

    def test_every_condition_constructs_with_the_released_settings(self):
        for condition in self.agent.CONDITIONS:
            with self.subTest(condition=condition):
                instance, _ = self.build(condition)
                self.assertEqual(instance.max_steps, 100)
                self.assertEqual(instance._budget.strict_target, 28672 - 2048)
                nudged = condition == "clm_self"
                self.assertEqual(bool(instance.nudge_ratios), nudged)

    def test_trigger_is_sixty_percent_of_the_enforced_budget(self):
        instance, _ = self.build("clm_fixed")
        self.assertEqual(instance.trigger_tokens, round(0.6 * (28672 - 2048)))

    def test_manager_defaults_to_actor_server_and_can_be_separate(self):
        _, same = self.build("mgr_watch")
        self.assertEqual(
            (same["manager_model"], same["manager_api_base"]),
            ("openai/Qwen3.5-9B-BF16", "http://127.0.0.1:1/v1"),
        )
        _, big = self.build(
            "mgr_request", manager_model="Qwen3.5-27B-BF16", manager_base_url="http://h:3/v1"
        )
        self.assertEqual(big["manager_mode"], "request")
        self.assertEqual(big["manager_api_base"], "http://h:3/v1")

    def test_system_prompts_use_the_submit_paragraph(self):
        from clm_harness.clm_agent.harness import _SYSTEM_TEMPLATE
        from clm_harness.utils.finish_policy import FINISH_INSTRUCTIONS

        official = _SYSTEM_TEMPLATE.replace("{{finish_instructions}}", FINISH_INSTRUCTIONS)
        fixed, _ = self.build("clm_fixed")
        self.assertIn("harness decides when you compact", fixed.system_prompt(official))
        managed, _ = self.build("mgr_request")
        text = managed.system_prompt(official)
        self.assertIn("separate manager maintains your context", text)
        self.assertIn("REQUEST_COMPACTION", text)
        self.assertTrue(text.rstrip().endswith(official.split(FINISH_INSTRUCTIONS)[1].rstrip()))

    def test_task_keeps_answers_out_of_the_sandbox(self):
        task = self.agent.build_task("What river?", self.logs / "ep", "/c/x.sif")
        files = sorted(p.relative_to(task).as_posix() for p in task.rglob("*") if p.is_file())
        self.assertEqual(files, ["instruction.md", "task.toml", "tests/test.sh"])
        self.assertIn("What river?", (task / "instruction.md").read_text())
        self.assertIn('docker_image = "/c/x.sif"', (task / "task.toml").read_text())


if __name__ == "__main__":
    unittest.main()
