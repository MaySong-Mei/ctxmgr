"""Exercise login-node scheduling with a fake Slurm service, without GPUs."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAKE_SLURM = r"""
import json, os, pathlib, sys
root = pathlib.Path(os.environ["FAKE_SLURM_DIR"])
name = pathlib.Path(sys.argv[0]).name
record = {"tool": name, "argv": sys.argv[1:]}
with (root / "calls.jsonl").open("a") as handle:
    handle.write(json.dumps(record) + "\n")
if name == "sbatch":
    if "--test-only" in sys.argv:
        print("sbatch: test-only allocation accepted", file=sys.stderr)
        sys.exit(0)
    count_file = root / "count"
    count = int(count_file.read_text()) + 1 if count_file.exists() else 1
    count_file.write_text(str(count))
    if str(count) == os.environ.get("FAKE_FAIL_AT"):
        print("scheduler connection interrupted", file=sys.stderr)
        sys.exit(1)
    if str(count) == os.environ.get("FAKE_MALFORMED_AT"):
        print("unexpected success response")
    else:
        print(f"{100 + count};bouchet")
elif name == "scontrol" and "release" in sys.argv:
    if os.environ.get("FAKE_RELEASE_FAIL"):
        sys.exit(1)
elif name == "scontrol" and "hostnames" in sys.argv:
    print(os.environ.get("FAKE_COMPUTE_HOST", "c1234u01n01"))
elif name == "hostname":
    print(os.environ.get("FAKE_HOSTNAME", "c1234u01n01"))
"""


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "source package"
        self.package.mkdir()
        for name in ("scripts/clm_vllm.sh", "slurm/clm_vllm.sbatch", "configs/clm-vllm.env"):
            destination = self.package / name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        (self.package / "scripts/clm_vllm_runtime.sh").write_text("#!/bin/bash\nexit 0\n")
        (self.package / "README.md").write_text("Frozen source fixture.\n")
        self.fake = self.root / "fake"
        self.fake.mkdir()
        for name in ("sbatch", "scontrol", "sacct", "squeue", "hostname"):
            script = self.fake / name
            script.write_text(f"#!{sys.executable}\n" + FAKE_SLURM)
            script.chmod(0o755)
        self.runs = self.root / "new runs"
        self.config = self.root / "test.env"
        self.config.write_text(
            f"CLM_RUNS_DIR='{self.runs}'\n"
            f"PROJECT_DIR='{self.root / 'project'}'\n"
            f"SCRATCH_DIR='{self.root / 'scratch'}'\n"
            "CLM_PARALLEL=2\n"
        )
        self.env = dict(
            os.environ,
            PATH=f"{self.fake}:{os.environ['PATH']}",
            FAKE_SLURM_DIR=str(self.fake),
            USER="cpsc4880_test",
        )
        self.script = self.package / "scripts/clm_vllm.sh"

    def run_launcher(self, *args, success=True, **environment):
        result = subprocess.run(
            ["bash", str(self.script), *map(str, args)],
            env=dict(self.env, **environment),
            check=False,
            capture_output=True,
            text=True,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def calls(self, tool=None):
        path = self.fake / "calls.jsonl"
        records = (
            [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        )
        return [item for item in records if tool is None or item["tool"] == tool]

    def submit(self, **environment):
        self.run_launcher("submit", "--config", self.config, **environment)
        return next(self.runs.glob("clm-vllm-*"))

    def test_complete_dependency_chain_and_frozen_grid(self):
        suite = self.submit()
        jobs = self.calls("sbatch")
        self.assertEqual(len(jobs), 3)
        setup, array, report = [job["argv"] for job in jobs]
        self.assertIn("--hold", setup)
        self.assertIn("--partition=day", setup)
        self.assertIn("--array=0-19%2", array)
        self.assertIn("--dependency=afterok:101", array)
        self.assertIn("--gpus-per-node=h200:1", array)
        self.assertIn("--partition=gpu_h200", array)
        self.assertIn("--dependency=afterany:102", report)
        self.assertIn("--kill-on-invalid-dep=yes", array)
        self.assertEqual(self.calls()[-1], {"tool": "scontrol", "argv": ["release", "101"]})
        cells = (suite / "cells.tsv").read_text().splitlines()
        self.assertEqual(len(cells), 21)
        self.assertEqual(cells[-1], "19\t27B\t45")
        self.assertEqual(
            (suite / "jobs.tsv").read_text(), "role\tjob_id\nsetup\t101\narray\t102\nreport\t103\n"
        )
        (self.package / "README.md").write_text("Changed after submit.\n")
        self.assertEqual((suite / "source/README.md").read_text(), "Frozen source fixture.\n")
        self.assertEqual((suite / "submission/state").read_text(), "submitted\n")

    def test_check_submits_no_jobs_or_suite(self):
        result = self.run_launcher("check", "--config", self.config)
        self.assertIn("no jobs will be submitted", result.stdout)
        self.assertEqual(len(self.calls("sbatch")), 3)
        self.assertTrue(all("--test-only" in call["argv"] for call in self.calls("sbatch")))
        self.assertFalse(self.runs.exists())
        self.assertFalse((self.fake / "count").exists())

    def test_submission_failure_keeps_setup_held(self):
        self.run_launcher("submit", "--config", self.config, success=False, FAKE_FAIL_AT="2")
        suite = next(self.runs.glob("clm-vllm-*"))
        self.assertEqual(len(self.calls("sbatch")), 2)
        self.assertEqual(self.calls("scontrol"), [])
        self.assertEqual((suite / "jobs.tsv").read_text(), "role\tjob_id\nsetup\t101\n")
        self.assertEqual((suite / "submission/state").read_text(), "uncertain:array\n")

    def test_malformed_response_halts_without_releasing(self):
        self.run_launcher("submit", "--config", self.config, success=False, FAKE_MALFORMED_AT="3")
        suite = next(self.runs.glob("clm-vllm-*"))
        self.assertEqual(self.calls("scontrol"), [])
        self.assertNotIn("report\t", (suite / "jobs.tsv").read_text())
        self.assertEqual((suite / "submission/state").read_text(), "uncertain:report\n")

    def test_release_failure_retains_all_recorded_ids(self):
        self.run_launcher("submit", "--config", self.config, success=False, FAKE_RELEASE_FAIL="1")
        suite = next(self.runs.glob("clm-vllm-*"))
        self.assertIn("report\t103", (suite / "jobs.tsv").read_text())
        self.assertEqual((suite / "submission/state").read_text(), "release-requested:101\n")

    def test_report_only_submits_cpu_snapshot(self):
        suite = self.submit()
        self.run_launcher("report", suite)
        snapshot = self.calls("sbatch")[-1]["argv"]
        self.assertIn("--partition=day", snapshot)
        self.assertFalse(any(arg.startswith("--array=") for arg in snapshot))
        self.assertFalse(any(arg.startswith("--gpus") for arg in snapshot))
        self.assertEqual(snapshot[-2:], ["report", str(suite)])
        self.assertIn("snapshot\t104", (suite / "jobs.tsv").read_text())

    def test_frozen_config_tamper_is_rejected(self):
        suite = self.submit()
        with (suite / "config.env").open("a") as handle:
            handle.write("CLM_PARALLEL=20\n")
        self.run_launcher("report", suite, success=False)
        self.assertEqual(len(self.calls("sbatch")), 3)

    def test_invalid_parallelism_cannot_submit(self):
        with self.config.open("a") as handle:
            handle.write("CLM_PARALLEL=21\n")
        self.run_launcher("submit", "--config", self.config, success=False)
        self.assertEqual(self.calls(), [])

    def test_worker_refuses_login_node(self):
        suite = self.submit()
        result = subprocess.run(
            ["bash", str(suite / "source/slurm/clm_vllm.sbatch"), "setup", str(suite)],
            env=dict(
                self.env,
                SLURM_JOB_ID="101",
                SLURM_JOB_NODELIST="c1234u01n01",
                FAKE_HOSTNAME="login2",
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("login node", result.stderr)

    def test_worker_refuses_host_outside_allocation(self):
        suite = self.submit()
        result = subprocess.run(
            ["bash", str(suite / "source/slurm/clm_vllm.sbatch"), "setup", str(suite)],
            env=dict(
                self.env,
                SLURM_JOB_ID="101",
                SLURM_JOB_NODELIST="c1234u01n01",
                FAKE_HOSTNAME="c5678u02n01",
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside the allocated", result.stderr)

    def test_worker_verifies_source_before_runtime(self):
        suite = self.submit()
        (suite / "source/README.md").write_text("Unexpected source edit.\n")
        result = subprocess.run(
            ["bash", str(suite / "source/slurm/clm_vllm.sbatch"), "setup", str(suite)],
            env=dict(self.env, SLURM_JOB_ID="101", SLURM_JOB_NODELIST="c1234u01n01"),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAILED", result.stdout)


if __name__ == "__main__":
    unittest.main()
