"""Real pinned CLM/ContextBench integration without a model server or GPU.

Run with the locked client environment on an allocated compute node. This
generates all 24 official instances and validates the 240-episode plan. It does
not pretend to test vLLM initialization or model accuracy.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class PinnedIntegrationTests(unittest.TestCase):
    def test_real_pinned_sources_full_grid_and_27b_cell(self):
        if importlib.util.find_spec("clm_contextbench") is None:
            self.skipTest("Run in envs/clm/.venv to test the real pinned CLM packages")

        from ctxmgr.harness import clm_official, clm_upstream

        verified = clm_upstream.verify_upstream()
        self.assertTrue(verified["verified"])
        self.assertEqual(verified["clm_commit"], "483198229ab7b90c0eed4651306ab3b3505aaf50")
        self.assertEqual(
            verified["contextbench_commit"], "fb533c18c78a82d3f1b2047a9c1700735cf829ae"
        )
        observed_sources = clm_official.verify_team_sources()
        self.assertEqual(observed_sources["commit"], clm_official.TEAM_SOURCE_COMMIT)

        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp) / "shared"
            plan = clm_official.prepare(shared)
            self.assertEqual(len(plan["episodes"]), 240)
            self.assertEqual(plan["config"]["seeds"], [42, 43, 44, 45])
            self.assertEqual(plan["models"][-1], "Qwen3.5-27B-BF16")
            self.assertEqual(
                plan["sampling"],
                {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_tokens": 8192,
                    "enable_thinking": True,
                },
            )
            self.assertEqual(plan["config"]["settings"]["profile_overrides"], {})
            self.assertEqual(len(list((shared / "instances").iterdir())), 24)
            self.assertEqual(clm_official.load_plan(shared), plan)

            cell_root = Path(tmp) / "27b-seed42"
            cell = clm_official.prepare(
                cell_root,
                sizes=["27B"],
                seeds=[42],
                reference_instances=shared / "instances",
            )
            self.assertEqual(len(cell["episodes"]), 12)
            shared_hashes = {row["name"]: row["instance_sha256"] for row in plan["episodes"]}
            for row in cell["episodes"]:
                self.assertEqual(row["instance_sha256"], shared_hashes[row["name"]])
                self.assertEqual(
                    (Path(row["instance"]) / "manifest.json").read_bytes(),
                    (shared / "instances" / row["name"] / "manifest.json").read_bytes(),
                )

            rows = clm_official.summary(cell_root, write_outputs=False)
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(row["status"] == "unrun" for row in rows))
            self.assertTrue(all(row["reward"] is None for row in rows))
            self.assertFalse((cell_root / "results.json").exists())
            self.assertFalse((cell_root / "results.txt").exists())
            clm_official.summary(cell_root)
            self.assertEqual(json.loads((cell_root / "results.json").read_text()), rows)

            original_plan = (shared / "official-run.json").read_bytes()
            with self.assertRaises(FileExistsError):
                clm_official.prepare(shared)
            self.assertEqual((shared / "official-run.json").read_bytes(), original_plan)


if __name__ == "__main__":
    unittest.main()
