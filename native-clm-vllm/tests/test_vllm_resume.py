"""CPU-only tests for guarded continuation; never contact a real scheduler."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("clm_resume", ROOT / "scripts" / "clm_resume.py")
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)
FAKE_SLURM = '''import json, os, pathlib, sys
p=pathlib.Path(os.environ["FAKE_SLURM_DIR"])
name=pathlib.Path(sys.argv[0]).name
with (p/"calls.jsonl").open("a") as f: f.write(json.dumps({"tool":name,"argv":sys.argv[1:]})+"\\n")
if name=="squeue":
    if os.environ.get("QUEUE_ERROR"): sys.exit(1)
    print(os.environ.get("ACTIVE_JOB", ""))
elif name=="sbatch":
    if "--test-only" in sys.argv: sys.exit(0)
    count=p/"count"
    n=int(count.read_text())+1 if count.exists() else 1
    count.write_text(str(n))
    if str(n)==os.environ.get("FAIL_AT"): sys.exit(1)
    print("uncertain response" if str(n)==os.environ.get("BAD_AT") else str(100+n)+";bouchet")
'''


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.suite = self.root / "suite with spaces"
        (self.suite / "source").mkdir(parents=True)
        (self.suite / "ready.json").write_text("{}")
        (self.suite / "source.sha256").write_text("")
        (self.suite / "jobs.tsv").write_text("role\tjob_id\nsetup\t11\narray_27B\t17\n")
        config = self.suite / "config.env"
        config.write_text("\n".join([
            "CLM_ACCOUNT=testcourse", "CLM_QOS=normal", "CLM_CPU_PARTITION=day",
            "CLM_GPU_27B_PARTITION=gpu_h200", "CLM_GPU_27B_TYPE=h200", "CLM_GPU_27B_MEM=128G",
            "CLM_GPU_CPUS=8", "CLM_REPORT_CPUS=2", "CLM_REPORT_MEM=8G",
            "CLM_GPU_TIME=1-00:00:00", "CLM_REPORT_TIME=01:00:00", ""
        ]))
        (self.suite / "config.sha256").write_text(f"{resume.digest(config)}  config.env\n")
        self.fake = self.root / "fake"
        self.fake.mkdir()
        for name in ("sbatch", "squeue", "scontrol"):
            path = self.fake / name
            path.write_text(f"#!{sys.executable}\n" + FAKE_SLURM)
            path.chmod(0o755)
        self.env = dict(os.environ, PATH=f"{self.fake}:{os.environ['PATH']}",
                        FAKE_SLURM_DIR=str(self.fake), USER="test_user")

    def run_launcher(self, *args, **env):
        return subprocess.run(["bash", str(ROOT / "scripts" / "clm_resume.sh"), str(self.suite), *args],
                              env=dict(self.env, **env), text=True, capture_output=True)

    def calls(self, tool):
        path = self.fake / "calls.jsonl"
        return [x for x in map(json.loads, path.read_text().splitlines()) if x["tool"] == tool] if path.exists() else []

    def test_only_27b_chain_and_duplicate_prevention(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        jobs = [r["argv"] for r in self.calls("sbatch")]
        self.assertEqual(len(jobs), 3)
        self.assertIn("--hold", jobs[0])
        self.assertIn("--array=16-19%1", jobs[1])
        self.assertIn("--gpus-per-node=h200:1", jobs[1])
        self.assertIn("--dependency=afterok:101", jobs[1])
        self.assertIn("--dependency=afterok:101,afterany:102", jobs[2])
        self.assertEqual(self.calls("scontrol"), [{"tool": "scontrol", "argv": ["release", "101"]}])
        record = (self.suite / "jobs.tsv").read_text()
        self.assertIn("resume_27B\t102\n", record)
        self.assertEqual(self.run_launcher().returncode, 0)
        self.assertEqual(len(self.calls("sbatch")), 3)

    def test_scheduler_failure_does_not_submit(self):
        self.assertNotEqual(self.run_launcher(QUEUE_ERROR="1").returncode, 0)
        self.assertEqual(self.calls("sbatch"), [])
        self.assertFalse((self.suite / "continuation-27b-v1").exists())

    def test_active_job_does_not_submit(self):
        self.assertNotEqual(self.run_launcher(ACTIVE_JOB="17").returncode, 0)
        self.assertEqual(self.calls("sbatch"), [])

    def test_existing_cell_is_not_overwritten(self):
        target = self.suite / "cells" / "16-27B-s42"
        target.mkdir(parents=True)
        (target / "sentinel").write_text("preserve")
        self.assertNotEqual(self.run_launcher().returncode, 0)
        self.assertEqual((target / "sentinel").read_text(), "preserve")
        self.assertEqual(self.calls("sbatch"), [])

    def test_uncertain_submission_keeps_preflight_held(self):
        self.assertNotEqual(self.run_launcher(BAD_AT="2").returncode, 0)
        self.assertEqual(self.calls("scontrol"), [])
        self.assertIn("uncertain:resume_27B", (self.suite / "continuation-27b-v1" / "state").read_text())
        self.assertEqual(self.run_launcher().returncode, 0)
        self.assertEqual(len(self.calls("sbatch")), 2)

    def test_resource_check_submits_nothing(self):
        result = self.run_launcher("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all("--test-only" in x["argv"] for x in self.calls("sbatch")))
        self.assertFalse((self.suite / "continuation-27b-v1").exists())


class OverlayTests(unittest.TestCase):
    def test_exact_patch_allows_target_and_rejects_outside_grid(self):
        original = 'def cell(suite, index):\n' + resume.OLD_GUARD + '\n    return cells(suite)[index]\n'
        before, after = resume.patch_cell_source(original)
        self.assertEqual(before, original)
        namespace = {"cells": lambda _: list(range(20))}
        exec(compile(after, "patched-cell.py", "exec"), namespace)
        self.assertEqual(namespace["cell"]("suite", 19), 19)
        for invalid in (-1, 20):
            with self.assertRaises(ValueError):
                namespace["cell"]("suite", invalid)
        with self.assertRaises(ValueError):
            resume.patch_cell_source(original.replace("< 16:", "< 18:"))

    def test_runtime_rewrite_preserves_environment_and_unknown_layout_refused(self):
        source = "export CUSTOM_VARIANT=zh_comments\n" + "\n".join(
            f'"$client_python" {resume.MODULE_ENTRY} {mode} --suite "$suite"'
            for mode in ("setup", "cell", "report"))
        updated = resume.patch_runtime(source)
        self.assertIn("export CUSTOM_VARIANT=zh_comments", updated)
        self.assertNotIn(resume.MODULE_ENTRY, updated)
        self.assertEqual(updated.count("CLM_RESUME_RUNNER"), 3)
        with self.assertRaises(ValueError):
            resume.patch_runtime(source.replace("clm_vllm_suite cell", "another_module cell"))

    def test_rewritten_runtime_executes_cell_and_report_with_spaces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = root / "runner with spaces.py"
            runner.write_text("import os,sys,json\nprint(json.dumps([sys.argv[1:],os.environ['CUSTOM_VARIANT']]))\n")
            runtime = root / "runtime.sh"
            source = ('#!/usr/bin/env bash\nset -euo pipefail\n'
                      f'client_python="{sys.executable}"\n'
                      'export CUSTOM_VARIANT=zh_comments\nmode=$1\nsuite=$2\ncase "$mode" in\n' +
                      "\n".join(f'{mode}) "$client_python" {resume.MODULE_ENTRY} {mode} --suite "$suite" ;;'
                                for mode in ("setup", "cell", "report")) + '\nesac\n')
            runtime.write_text(resume.patch_runtime(source))
            for mode in ("cell", "report"):
                result = subprocess.run(["bash", str(runtime), mode, "/suite with spaces"],
                                        env=dict(os.environ, CLM_RESUME_RUNNER=str(runner)),
                                        text=True, capture_output=True, check=True)
                self.assertEqual(json.loads(result.stdout), [[mode, "--suite", "/suite with spaces"], "zh_comments"])

    def test_missing_grid_requires_192_valid_and_27b_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            selections = [{"index": i, "size": "27B" if i >= 16 else "4B", "seed": 42 + i % 4,
                           "root": str(suite / "cells" / str(i))} for i in range(20)]
            rows = [{"cell": i, "model": "model", "task": f"task-{j}", "seed": 42 + i % 4,
                     "condition": condition, "status": "unrun" if i >= 16 else "completed",
                     "reward": "" if i >= 16 else "0.0"}
                    for i in range(20) for j in range(6) for condition in ("none", "clm_self")]
            module = SimpleNamespace(cells=lambda _: selections, expected_rows=lambda _: rows)
            resume.validate_missing_grid(module, suite, rows)
            rows[0]["status"] = "error"
            with self.assertRaises(ValueError):
                resume.validate_missing_grid(module, suite, rows)
            rows[0]["status"] = "completed"
            Path(selections[16]["root"]).mkdir(parents=True)
            with self.assertRaises(ValueError):
                resume.validate_missing_grid(module, suite, rows)

    def test_provenance_annotation_updates_report_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            out = suite / "report"
            out.mkdir()
            continuation = suite / "continuation-27b-v1"
            continuation.mkdir()
            (continuation / "preflight.json").write_text("{}")
            (out / "report.md").write_text("Original Chinese-variant report\n")
            (out / "provenance.json").write_text('{"variant":"zh-comments"}')
            status = {"graded": 240, "unrun": 0, "failed": 0, "running": 0, "report_sha256": {
                name: resume.digest(out / name) for name in ("report.md", "provenance.json")}}
            resume.write_json(out / "report-status.json", status)
            receipt = {"patch": {"id": "test"}, "overlay_artifacts": {"executed-cell.py": "testhash"}}
            result = resume.annotate_report(suite, continuation, receipt)
            for name, expected in result["report_sha256"].items():
                self.assertEqual(resume.digest(out / name), expected)
            self.assertEqual(resume.read_json(out / "provenance.json")["variant"], "zh-comments")
            self.assertIn("Original Chinese-variant report", (out / "report.md").read_text())

    def test_fingerprint_rejects_changed_existing_grade(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            (suite / "score.json").write_text("old")
            fingerprints = {"score.json": resume.digest(suite / "score.json")}
            resume.check_fingerprints(suite, fingerprints)
            (suite / "score.json").write_text("changed")
            with self.assertRaises(ValueError):
                resume.check_fingerprints(suite, fingerprints)


if __name__ == "__main__":
    unittest.main()
