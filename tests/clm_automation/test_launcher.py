"""Exercise the login-node entry point with a fake scheduler."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[2] / "scripts" / "clm_automation.sh"


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "checkout"
        scripts = self.repo / "scripts"
        scripts.mkdir(parents=True)
        self.launcher = scripts / "clm_automation.sh"
        shutil.copyfile(LAUNCHER, self.launcher)
        source = self.repo / "tools" / "clm_automation"
        source.mkdir(parents=True)
        for name in ("controller.py", "github_report.py", "worker.sbatch"):
            (source / name).write_text("# Fixture\n")
        self.recovery = self.root / "recovery with spaces"
        (self.recovery / "tools").mkdir(parents=True)
        (self.recovery / "runtime.env").touch()
        (self.recovery / "plan-ready").touch()
        (self.recovery / "tools" / "recover.sh").write_text("echo 'Recovery status fixture'\n")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.calls = self.root / "calls.jsonl"
        stub = f"#!{sys.executable}\n" + '''import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
with open(os.environ["CALLS"], "a") as stream:
    stream.write(json.dumps([name, sys.argv[1:], os.environ.get("SBATCH_PARTITION")]) + "\\n")
if name == "sbatch":
    if os.environ.get("SUBMIT_FAIL"):
        sys.exit(1)
    if "--test-only" in sys.argv:
        print("Estimated start only")
    else:
        print(os.environ.get("SUBMIT_RESULT", "900001"))
elif name == "sacct":
    job = sys.argv[sys.argv.index("-j") + 1]
    state = os.environ.get("JOB_STATE", "COMPLETED")
    if state:
        print(job + "|" + state + "|")
elif name == "scontrol" and os.environ.get("RELEASE_FAIL"):
    sys.exit(1)
'''
        for name in ("sbatch", "sacct", "scontrol", "gh"):
            p = self.bin / name
            p.write_text(stub)
            p.chmod(0o755)
        self.env = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}", "CALLS": str(self.calls)}

    def run_launcher(self, action, *args, **env):
        return subprocess.run(
            ["bash", str(self.launcher), action, str(self.recovery), *args],
            env={**self.env, **env}, text=True, capture_output=True, timeout=10,
        )

    def events(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_check_does_not_record_or_release_a_job(self):
        result = self.run_launcher("check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--test-only", self.events()[0][1])
        self.assertFalse((self.recovery / "automation" / "controller-jobs.tsv").exists())
        self.assertEqual(len(self.events()), 1)

    def test_start_freezes_code_holds_records_releases(self):
        result = self.run_launcher("start", SBATCH_PARTITION="wrong")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.events()
        self.assertEqual([c[0] for c in calls], ["sbatch", "scontrol"])
        self.assertIn("--hold", calls[0][1])
        self.assertIn("--partition=day", calls[0][1])
        self.assertIsNone(calls[0][2])
        ledger = (self.recovery / "automation" / "controller-jobs.tsv").read_text()
        self.assertIn("900001\t", ledger)
        self.assertFalse((self.recovery / "automation" / "launch-intent.txt").exists())
        self.assertEqual(len(list((self.recovery / "automation").glob("code-*/SHA256SUMS"))), 1)

    def test_second_start_does_not_duplicate_active_controller(self):
        self.assertEqual(self.run_launcher("start").returncode, 0)
        result = self.run_launcher("start", JOB_STATE="RUNNING")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No duplicate", result.stdout)
        self.assertEqual(sum(c[0] == "sbatch" for c in self.events()), 1)

    def test_unknown_accounting_fails_closed(self):
        self.run_launcher("start")
        result = self.run_launcher("start", JOB_STATE="")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(sum(c[0] == "sbatch" for c in self.events()), 1)

    def test_ambiguous_submission_blocks_retry(self):
        result = self.run_launcher("start", SUBMIT_RESULT="unparseable")
        self.assertEqual(result.returncode, 2)
        result = self.run_launcher("start")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(sum(c[0] == "sbatch" for c in self.events()), 1)

    def test_failed_submission_blocks_retry(self):
        self.assertEqual(self.run_launcher("start", SUBMIT_FAIL="1").returncode, 2)
        self.assertEqual(self.run_launcher("start").returncode, 2)
        self.assertEqual(sum(c[0] == "sbatch" for c in self.events()), 1)

    def test_release_failure_retains_job_record(self):
        self.assertEqual(self.run_launcher("start", RELEASE_FAIL="1").returncode, 2)
        self.assertIn("900001", (self.recovery / "automation" / "controller-jobs.tsv").read_text())
        self.assertEqual(self.run_launcher("start", JOB_STATE="PENDING").returncode, 0)
        self.assertEqual(sum(c[0] == "sbatch" for c in self.events()), 1)

    def test_stop_does_not_cancel_experiment(self):
        self.assertEqual(self.run_launcher("stop").returncode, 0)
        self.assertTrue((self.recovery / "automation" / "STOP").exists())
        self.assertEqual(self.events(), [])

    def test_reject_education_controller_to_avoid_queue_deadlock(self):
        self.assertEqual(self.run_launcher("start", "--partition", "education").returncode, 2)
        self.assertEqual(self.events(), [])

    def test_publish_options_forwarded_only_when_explicit(self):
        self.assertEqual(self.run_launcher("start", "--publish").returncode, 0)
        submit = next(c for c in self.events() if c[0] == "sbatch")
        self.assertIn("--publish", submit[1])
        self.assertIn("MaySong-Mei/ctxmgr", submit[1])


if __name__ == "__main__":
    unittest.main()
