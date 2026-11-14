"""Grid and command-line tests for the when-versus-how runner (no servers, no models)."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ctxmgr.harness import whenhow_suite as suite

MODELS = {
    size: {"model_alias": f"Qwen3.5-{size}-BF16", "revision": "r" * 40, "sha256": "a" * 64}
    for size in ("2B", "4B", "9B", "27B")
}
READY = {"image": "/c/contextbench.sif", "server_version": "0.30.0"}


class WhenHowPlanTests(unittest.TestCase):
    def test_grid_sizes(self):
        total = 0
        for seed in suite.SEEDS:
            for actor in suite.SIZES:
                same = suite.plan(actor, "same", seed)
                self.assertEqual(len(same), 5 * 4)
                total += len(same)
            for actor in ("2B", "4B", "9B"):
                big = suite.plan(actor, "27B", seed)
                self.assertEqual({row["manager"] for row in big}, {"27B"})
                self.assertEqual(len(big), 3 * 4)
                total += len(big)
        self.assertEqual(total, 464)

    def test_added_seeds_carry_matched_controls(self):
        rows = suite.plan("4B", "same", 47)
        tags = {row["tag"] for row in rows}
        self.assertIn("none_same_harness", tags)
        self.assertIn("clm_self", tags)
        self.assertEqual(len(rows), 7 * 4)
        self.assertEqual(len(suite.plan("4B", "27B", 47)), 3 * 4)

    def test_invalid_combinations_are_rejected(self):
        with self.assertRaises(ValueError):
            suite.plan("27B", "27B", 42)
        with self.assertRaises(ValueError):
            suite.plan("0.8B", "same", 42)
        with self.assertRaises(ValueError):
            suite.plan("4B", "same", 7)

    def test_same_size_manager_and_output_paths(self):
        rows = {row["tag"] for row in suite.plan("4B", "same", 43)}
        self.assertEqual(
            rows,
            {"clm_self_noskill", "clm_fixed", "mgr_fixed@4B", "mgr_request@4B", "mgr_watch@4B"},
        )
        row = suite.plan("4B", "same", 43)[0]
        self.assertEqual(
            suite.episode_out(Path("/r"), row),
            Path("/r/Qwen3.5-4B-BF16/clm_self_noskill/needle_retention-K4-C24-s43"),
        )

    def test_argv_routes_manager_to_its_own_server(self):
        row = next(r for r in suite.plan("2B", "27B", 44) if r["condition"] == "mgr_watch")
        urls = {"2B": "http://127.0.0.1:1/v1", "27B": "http://127.0.0.1:2/v1"}
        argv = suite.episode_argv(
            row, instance=Path("/i"), out=Path("/o"), ready=READY, models=MODELS, urls=urls
        )
        value = {flag: argv[argv.index(flag) + 1] for flag in argv if flag.startswith("--")}
        self.assertEqual(value["--condition"], "mgr_watch")
        self.assertEqual(value["--base-url"], urls["2B"])
        self.assertEqual(value["--manager-base-url"], urls["27B"])
        self.assertEqual(value["--manager-model"], "Qwen3.5-27B-BF16")
        self.assertEqual(value["--budget-ruler"], "o200k")
        plain = next(r for r in suite.plan("2B", "same", 44) if r["condition"] == "clm_fixed")
        argv = suite.episode_argv(
            plain, instance=Path("/i"), out=Path("/o"), ready=READY, models=MODELS, urls=urls
        )
        self.assertNotIn("--manager-model", argv)


if __name__ == "__main__":
    unittest.main()
