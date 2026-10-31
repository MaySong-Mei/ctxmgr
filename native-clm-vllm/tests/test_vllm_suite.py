"""Grid, scheduling-state and report tests with synthetic saved-result fixtures."""

import copy
import csv
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ctxmgr import harness
from ctxmgr.harness import clm_vllm_suite as suite_module


def blank_rows(root):
    return [
        {**row, "status": "unrun", "reward": None, "termination": None}
        for row in suite_module.expected_rows(root)
    ]


class SuiteGridTests(unittest.TestCase):
    def test_gpu_allocation_checks_model_specific_hardware(self):
        with patch.dict(
            "os.environ",
            {
                "CLM_GPU_4B_PARTITION": "gpu",
                "CLM_GPU_4B_TYPE": "rtx_5000_ada",
                "CLM_GPU_4B_MEM": "64G",
            },
        ):
            self.assertEqual(
                suite_module.gpu_allocation("4B"),
                {"partition": "gpu", "type": "rtx_5000_ada", "host_memory": "64G"},
            )
        observed = "name, memory.total [MiB], driver_version\nNVIDIA RTX 5000 Ada Generation, 32760 MiB, 580.1\n"
        self.assertTrue(suite_module.gpu_name_matches(observed, "rtx_5000_ada"))
        self.assertFalse(suite_module.gpu_name_matches(observed, "h200"))
        with self.assertRaisesRegex(ValueError, "Unknown GPU type"):
            suite_module.gpu_name_matches(observed, "invented")

    def test_full_grid_has_240_distinct_episodes_and_48_for_each_model(self):
        cells = suite_module.cells("/tmp/suite")
        rows = suite_module.expected_rows("/tmp/suite")
        self.assertEqual(len(cells), 20)
        self.assertEqual(len(rows), 240)
        self.assertEqual(len({suite_module.row_key(row) for row in rows}), 240)
        self.assertEqual(sum(row["model"] == "Qwen3.5-9B-BF16" for row in rows), 48)
        self.assertEqual(sum(row["model"] == "Qwen3.5-27B-BF16" for row in rows), 48)
        for cell in cells:
            self.assertEqual(sum(row["cell"] == cell["index"] for row in rows), 12)

    def test_comparisons_use_only_matching_valid_seed_pairs(self):
        rows = blank_rows("/tmp/suite")
        model = "Qwen3.5-0.8B-BF16"
        task = "kv_store:R130"
        selected = {
            (row["seed"], row["condition"]): row
            for row in rows
            if row["model"] == model and row["task"] == task
        }
        selected[42, "none"].update(status="completed", reward=0.0)
        selected[42, "clm_self"].update(status="completed", reward=0.5)
        selected[43, "none"].update(status="completed", reward=1.0)
        selected[44, "clm_self"].update(status="completed", reward=1.0)
        comparison = suite_module.paired_comparison(rows)[0]
        self.assertEqual(
            comparison,
            {
                "model": model,
                "task": task,
                "paired": 1,
                "expected": 4,
                "none": 0.0,
                "clm_self": 0.5,
                "delta": 0.5,
            },
        )
        self.assertIsNone(suite_module.paired_comparison(rows)[1]["none"])

    def test_duplicate_episode_identity_is_rejected(self):
        rows = blank_rows("/tmp/suite")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            suite_module.paired_comparison(rows + [rows[0]])

    def test_tsv_and_markdown_distinguish_missing_from_zero(self):
        data = [{"reward": None, "status": "unrun"}, {"reward": 0.0, "status": "completed"}]
        actual = list(csv.DictReader(io.StringIO(suite_module.table_tsv(data)), delimiter="\t"))
        self.assertEqual(actual[0]["reward"], "")
        self.assertEqual(actual[1]["reward"], "0.0")
        self.assertEqual(suite_module.format_score(None), "—")
        self.assertEqual(suite_module.format_score(0.0), "0.0000")

    def test_server_argv_fixes_precision_context_and_sampling_defaults(self):
        argv = suite_module.server_argv(
            {"model_path": "/cache/snapshot", "model_alias": "Qwen3.5-27B-BF16"},
            "/serve/bin/python",
            43210,
        )
        self.assertEqual(
            argv[:5],
            ["/serve/bin/python", "-m", "vllm.entrypoints.cli.main", "serve", "/cache/snapshot"],
        )
        options = {
            "--dtype": "bfloat16",
            "--tensor-parallel-size": "1",
            "--max-model-len": "81920",
            "--max-num-seqs": "1",
            "--generation-config": "vllm",
            "--reasoning-parser": "qwen3",
            "--tool-call-parser": "qwen3_coder",
            "--seed": "0",
        }
        for flag, value in options.items():
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertIn("--language-model-only", argv)
        self.assertIn("--enable-auto-tool-choice", argv)
        self.assertEqual(
            json.loads(argv[argv.index("--default-chat-template-kwargs") + 1]),
            {"enable_thinking": True},
        )
        self.assertNotIn("--quantization", argv)
        self.assertEqual(
            suite_module.SAMPLING,
            {"temperature": 0.7, "top_p": 0.95, "max_tokens": 8192, "enable_thinking": True},
        )

    def test_scheduler_reads_terminal_jobs_and_ignores_step_rows(self):
        output = "123|CANCELLED by 1000|\n123.batch|CANCELLED|\n456|RUNNING|\n"
        with patch.object(suite_module.subprocess, "check_output", return_value=output):
            states = suite_module.scheduler_states(["123", "456"])
        self.assertEqual(states, {"123": "CANCELLED", "456": "RUNNING"})
        with patch.object(suite_module.subprocess, "check_output", side_effect=FileNotFoundError):
            self.assertEqual(suite_module.scheduler_states(["123"]), {})

    def test_readiness_refuses_foreign_listener_without_contacting_it(self):
        process = Mock(pid=123)
        process.poll.return_value = None
        guard = types.ModuleType("ctxmgr.harness.clm_vllm_process")
        guard.listener_owner = Mock(return_value="foreign")
        with (
            patch.dict(sys.modules, {"ctxmgr.harness.clm_vllm_process": guard}),
            patch.object(suite_module, "urlopen") as request,
            self.assertRaisesRegex(RuntimeError, "different process group"),
        ):
            suite_module.wait_healthy(process, "http://127.0.0.1:8000/health", 1)
        request.assert_not_called()

    def test_readiness_waits_for_owned_listener_and_checks_again(self):
        process = Mock(pid=123)
        process.poll.return_value = None
        guard = types.ModuleType("ctxmgr.harness.clm_vllm_process")
        guard.listener_owner = Mock(side_effect=["free", "owned", "owned"])
        with (
            patch.dict(sys.modules, {"ctxmgr.harness.clm_vllm_process": guard}),
            patch.object(suite_module.time, "sleep"),
            patch.object(suite_module, "urlopen") as request,
        ):
            request.return_value.__enter__.return_value.status = 200
            suite_module.wait_healthy(process, "http://127.0.0.1:8000/health", 5)
        self.assertEqual(guard.listener_owner.call_count, 3)
        request.assert_called_once()

    def test_cleanup_targets_group_even_when_leader_already_exited(self):
        process = Mock(pid=123, returncode=0)
        process.poll.return_value = 0
        with (
            patch.object(suite_module.os, "killpg") as kill_group,
            patch.object(suite_module.time, "monotonic", side_effect=[0, 16]),
        ):
            suite_module.stop_group(process)
        self.assertEqual(
            [call.args for call in kill_group.call_args_list],
            [(123, suite_module.signal.SIGTERM), (123, suite_module.signal.SIGKILL)],
        )
        process.wait.assert_called_once_with(timeout=10)


class SuiteReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.suite = Path(self.temp.name)
        self.selection = suite_module.cells(self.suite)[0]
        self.cell = Path(self.selection["root"])
        self.cell.mkdir(parents=True)
        (self.cell / "official-run.json").write_text("{}")
        self.model_manifests = {}
        for size in suite_module.MODEL_SIZES:
            alias = suite_module.model_identity(size)[0]
            path = self.suite / "models" / alias / "model.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"model_alias": alias, "revision": "r", "sha256": "h"}))
            self.model_manifests[alias] = suite_module.sha256(path)
        cell_manifest = self.cell / "models" / self.selection["model"] / "model.json"
        cell_manifest.parent.mkdir(parents=True)
        cell_manifest.write_bytes(
            (self.suite / "models" / self.selection["model"] / "model.json").read_bytes()
        )
        self.settings = {
            "profile": "official",
            "profile_overrides": {},
            "sampling": suite_module.SAMPLING,
        }
        self.shared = {
            "config": {"settings": self.settings},
            "episodes": [],
            "team_source_commit": "team-pin",
            "team_source_sha256": {"agents.py": "x"},
        }
        for task in suite_module.LEVELS:
            for seed in suite_module.SEEDS:
                name = f"{task}-s{seed}"
                self.shared["episodes"].append({"name": name, "instance_sha256": {"stream": name}})
        self.plan = {
            "config": {
                "sizes": [self.selection["size"]],
                "seeds": [self.selection["seed"]],
                "levels": list(suite_module.LEVELS),
                "conditions": list(suite_module.CONDITIONS),
                "settings": self.settings,
                "served_max_len": 81920,
            },
            "episodes": [item for item in self.shared["episodes"] if item["name"].endswith("s42")],
        }
        self.rows = [row for row in blank_rows(self.suite) if row["cell"] == 0]
        for row in self.rows:
            row.update(ops=None, edits=None, error=None)
        self.ready = {
            "model_manifests": self.model_manifests,
            "backend": "vllm",
            "server_version": "0.30.0",
            "precision": "bfloat16",
            "gpu_allocations": {
                "0.8B": {"partition": "gpu", "type": "a5000", "host_memory": "48G"},
                "2B": {"partition": "gpu", "type": "a5000", "host_memory": "48G"},
                "4B": {"partition": "gpu", "type": "rtx_5000_ada", "host_memory": "64G"},
                "9B": {"partition": "gpu", "type": "a40", "host_memory": "96G"},
                "27B": {"partition": "gpu_h200", "type": "h200", "host_memory": "128G"},
            },
            "upstream": {},
            "image_sha256": "image",
            "source_manifest_sha256": "source",
        }
        self.summary_write_flags = []

        def summary(_root, *, write_outputs=True):
            self.summary_write_flags.append(write_outputs)
            return copy.deepcopy(self.rows)

        fake = types.ModuleType("ctxmgr.harness.clm_official")
        fake.summary = summary
        fake.load_plan = lambda root: self.shared if Path(root).name == "shared" else self.plan
        patches = [
            patch.object(suite_module, "validate_ready", return_value=self.ready),
            patch.object(suite_module, "scheduler_states", return_value={}),
            patch.dict(sys.modules, {"ctxmgr.harness.clm_official": fake}),
            patch.object(harness, "clm_official", fake, create=True),
        ]
        for active in patches:
            active.start()
            self.addCleanup(active.stop)

    def report(self):
        with patch("sys.stdout", new=io.StringIO()):
            return suite_module.report(self.suite)

    def saved_rows(self):
        with (self.suite / "report/results.tsv").open() as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    def test_partial_report_preserves_zero_and_marks_finished_incomplete_episode(self):
        self.rows[0].update(status="completed", reward=0.0, termination="budget_wall")
        self.rows[1].update(status="running")
        (self.cell / "cell-status.json").write_text(
            json.dumps(
                {
                    "job_id": "123",
                    "status": "failed",
                    "finished_at": "2026-10-29T00:00:00Z",
                    "error": "Model server was killed",
                }
            )
        )
        self.assertEqual(self.report(), 1)
        status = suite_module.read_json(self.suite / "report/report-status.json")
        self.assertEqual(status["graded"], 1)
        self.assertEqual(status["failed"], 1)
        self.assertEqual(status["unrun"], 238)
        self.assertEqual(status["running"], 0)
        rows = self.saved_rows()
        self.assertEqual(len(rows), 240)
        self.assertEqual(rows[0]["reward"], "0.0")
        self.assertEqual(rows[1]["reward"], "")
        self.assertEqual(rows[1]["status"], "error")
        self.assertEqual(self.summary_write_flags, [False])
        markdown = (self.suite / "report/report.md").read_text()
        self.assertIn("1/240", markdown)
        self.assertIn("Qwen3.5-27B-BF16", markdown)

    def test_active_partial_episode_remains_running(self):
        self.rows[0].update(status="running")
        self.assertEqual(self.report(), 1)
        status = suite_module.read_json(self.suite / "report/report-status.json")
        self.assertEqual(status["running"], 1)
        self.assertEqual(status["unrun"], 239)

    def test_duplicate_summary_invalidates_cell_before_collecting_grades(self):
        self.rows[0].update(status="completed", reward=1.0, termination="stream_exhausted")
        self.rows[-1] = copy.deepcopy(self.rows[0])
        self.assertEqual(self.report(), 1)
        status = suite_module.read_json(self.suite / "report/report-status.json")
        self.assertEqual(status["graded"], 0)
        self.assertEqual(status["failed"], 12)
        self.assertTrue(all(row["reward"] == "" for row in self.saved_rows()))

    def test_hard_kill_detected_from_accounting_without_finished_receipt(self):
        self.rows[0].update(status="running")
        (self.cell / "cell-status.json").write_text(
            json.dumps({"job_id": "123", "status": "running"})
        )
        with patch.object(suite_module, "scheduler_states", return_value={"123": "OUT_OF_MEMORY"}):
            self.assertEqual(self.report(), 1)
        row = self.saved_rows()[0]
        self.assertEqual(row["status"], "error")
        self.assertIn("OUT_OF_MEMORY", row["error"])


if __name__ == "__main__":
    unittest.main()
