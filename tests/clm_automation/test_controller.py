"""Controller regression tests with a synthetic frozen plan and scheduler."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[2] / "tools/clm_automation/controller.py"
spec = importlib.util.spec_from_file_location("clm_controller", MODULE)
c = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = c
spec.loader.exec_module(c)
TASKS = [("kv_store", "R130"), ("kv_store", "R1000"), ("kv_store", "R2000"),
         ("needle_retention", "K4-C4"), ("needle_retention", "K4-C24"), ("needle_retention", "K8-C48")]
MISSING = [("0.8B", 42, 10), ("2B", 43, 10), ("2B", 44, 12), ("2B", 45, 12), ("4B", 42, 12)]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def append(path, values):
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\t".join(map(str, values)) + "\n")


class FakeScheduler:
    def __init__(self, root):
        self.root = root
        self.jobs = {"1": c.Job("1", "COMPLETED", "0:0"), "2": c.Job("2", "FAILED", "1:0")}
        self.calls = []
        self.pending = {}
        self.next_id = 500
        self.raise_before = False
        self.raise_after = False
        self.unavailable = False

    def states(self, ids):
        if self.unavailable:
            raise c.AccountingError("Temporary accounting failure.")
        return {identifier: self.jobs[identifier] for identifier in ids}

    def invoke(self, root, action):
        self.calls.append(action)
        if self.raise_before:
            raise c.ControllerError("Ambiguous submission")
        self.next_id += 1
        identifier = str(self.next_id)
        if action == "report":
            append(root / "cpu-jobs.tsv", ["report", identifier])
        else:
            gpu = c.table(root / "jobs.tsv", ["size", "seed", "job_id"])
            seen = {(row["size"], int(row["seed"])) for row in gpu}
            size, seed, _ = next(cell for cell in MISSING if cell[:2] not in seen)
            append(root / "jobs.tsv", [size, seed, identifier])
        self.jobs[identifier] = c.Job(identifier, "PENDING", "0:0")
        self.pending[identifier] = action
        if self.raise_after:
            raise c.ControllerError("Release failed after ledger append")

    def finish(self):
        for identifier, action in list(self.pending.items()):
            if action == "report":
                status = make_report(self.root)
                complete = status["status"] == "complete"
                self.jobs[identifier] = c.Job(identifier, "COMPLETED" if complete else "FAILED", "0:0" if complete else "1:0")
            else:
                self.jobs[identifier] = c.Job(identifier, "COMPLETED", "0:0")
            del self.pending[identifier]


def fixture(root):
    (root / "tools").mkdir()
    (root / "tools/recover.sh").write_text("#!/bin/bash\nexit 0\n")
    (root / "runtime.env").write_text("export CLM_PYTHON=python\n")
    (root / "original-suite.txt").write_text("/synthetic/original\n")
    (root / "plan-ready").touch()
    cells = []
    for size, seed, count in MISSING:
        episodes = [{"model": f"Qwen3.5-{size}-Q8_0", "family": family, "level": level,
                     "seed": seed, "condition": condition}
                    for family, level in (TASKS[1:] if count == 10 else TASKS)
                    for condition in ("none", "clm_self")]
        cells.append({"size": size, "seed": seed, "count": count, "episodes": episodes})
    manifest = {"tools_sha256": {"recover.sh": c.digest(root / "tools/recover.sh")}}
    write(root / "patch-manifest.json", manifest)
    plan = {"schema_version": 1, "planned": 144, "preserved": 88, "cells": cells,
            "patch_manifest_sha256": c.digest(root / "patch-manifest.json"),
            "runtime_files_sha256": {name: c.digest(root / name) for name in ("runtime.env", "original-suite.txt")}}
    write(root / "plan.json", plan)
    (root / "plan.sha256").write_text(c.digest(root / "plan.json") + "\n")
    (root / "cells.tsv").write_text("".join(f"{size}\t{seed}\t{count}\n" for size, seed, count in MISSING))
    (root / "jobs.tsv").write_text("size\tseed\tjob_id\n")
    (root / "cpu-jobs.tsv").write_text("role\tjob_id\nplan\t1\nreport\t2\n")
    make_report(root)


def add_gpu(root, scheduler, number):
    for index, (size, seed, _) in enumerate(MISSING[:number]):
        identifier = str(100 + index)
        append(root / "jobs.tsv", [size, seed, identifier])
        scheduler.jobs[identifier] = c.Job(identifier, "COMPLETED", "0:0")


def make_report(root):
    gpu = c.table(root / "jobs.tsv", ["size", "seed", "job_id"])
    by_cell = {(row["size"], int(row["seed"])): row["job_id"] for row in gpu}
    plan = c.read_json(root / "plan.json")
    selected = {(ep["model"], ep["family"], ep["level"], ep["seed"], ep["condition"]): (cell["size"], cell["seed"])
                for cell in plan["cells"] for ep in cell["episodes"]}
    rows = []
    for size in ("0.8B", "2B", "4B"):
        model = f"Qwen3.5-{size}-Q8_0"
        for seed in range(42, 46):
            for family, level in TASKS:
                for condition in ("none", "clm_self"):
                    key = model, family, level, seed, condition
                    missing_cell = selected.get(key)
                    job = by_cell.get(missing_cell)
                    rows.append({"model": model, "task": f"{family}:{level}", "seed": str(seed), "condition": condition,
                                 "status": "graded" if missing_cell is None or job else "unrun",
                                 "origin": "original" if missing_cell is None else "recovery" if job else "missing",
                                 "source_job_id": job or "90"})
    recovered = sum(row["origin"] == "recovery" for row in rows)
    status = {"status": "complete" if recovered == 56 else "incomplete", "planned": 144,
              "graded": 88 + recovered, "preserved_original": 88, "recovered": recovered, "failed": 0, "unrun": 56 - recovered}
    write(root / "report/report-status.json", status)
    with (root / "report/results.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    for name in ("summary.tsv", "comparison.tsv", "summary.txt"):
        (root / "report" / name).write_text("Synthetic report\n")
    return status


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixture(self.root)
        self.scheduler = FakeScheduler(self.root)

    def controller(self):
        return c.Controller(self.root, self.scheduler)

    def test_existing_last_gpu_is_awaited_and_never_resubmitted(self):
        add_gpu(self.root, self.scheduler, 5)
        self.scheduler.jobs["104"] = c.Job("104", "RUNNING", "0:0")
        controller = self.controller()
        self.assertEqual(controller.tick(), "waiting")
        self.assertEqual(self.scheduler.calls, [])
        self.scheduler.jobs["104"] = c.Job("104", "COMPLETED", "0:0")
        self.assertEqual(controller.tick(), "advanced")
        self.assertEqual(self.scheduler.calls, ["report"])
        self.assertEqual(controller.tick(), "waiting")
        self.scheduler.finish()
        self.assertEqual(controller.tick(), "complete")
        self.assertEqual(controller.tick(), "complete")
        self.assertEqual(self.scheduler.calls, ["report"])

    def test_partial_report_exit_one_validates_before_next_missing_cell(self):
        add_gpu(self.root, self.scheduler, 4)
        controller = self.controller()
        self.assertEqual(controller.tick(), "advanced")
        self.scheduler.finish()
        self.assertEqual(controller.tick(), "advanced")
        self.assertEqual(self.scheduler.calls, ["report", "next"])
        self.assertEqual(c.table(self.root / "jobs.tsv", ["size", "seed", "job_id"])[-1]["size"], "4B")
        self.assertEqual(controller.tick(), "waiting")
        self.scheduler.finish()
        controller.tick()
        self.scheduler.finish()
        self.assertEqual(controller.tick(), "complete")
        self.assertEqual(self.scheduler.calls, ["report", "next", "report"])

    def test_gpu_failure_never_advances(self):
        add_gpu(self.root, self.scheduler, 4)
        self.scheduler.jobs["103"] = c.Job("103", "FAILED", "0:9")
        with self.assertRaisesRegex(c.ControllerError, "GPU job"):
            self.controller().tick()
        self.assertEqual(self.scheduler.calls, [])

    def test_report_fatal_exit_is_not_incomplete_coverage(self):
        add_gpu(self.root, self.scheduler, 4)
        controller = self.controller()
        controller.tick()
        self.scheduler.jobs["501"] = c.Job("501", "FAILED", "2:0")
        with self.assertRaisesRegex(c.ControllerError, "CPU job"):
            controller.tick()
        self.assertEqual(self.scheduler.calls, ["report"])

    def test_wrong_selected_grade_blocks_next_even_when_count_matches(self):
        add_gpu(self.root, self.scheduler, 4)
        controller = self.controller()
        controller.tick()
        self.scheduler.finish()
        path = self.root / "report/results.tsv"
        text = path.read_text()
        text = text.replace("Qwen3.5-0.8B-Q8_0\tkv_store:R1000\t42", "WrongModel\tkv_store:R1000\t42", 1)
        path.write_text(text)
        with self.assertRaisesRegex(c.ControllerError, "selected grades"):
            controller.tick()
        self.assertEqual(self.scheduler.calls, ["report"])

    def test_report_must_be_rewritten_by_new_job(self):
        add_gpu(self.root, self.scheduler, 4)
        make_report(self.root)
        controller = self.controller()
        controller.tick()
        self.scheduler.jobs["501"] = c.Job("501", "FAILED", "1:0")
        with self.assertRaisesRegex(c.ControllerError, "previous report"):
            controller.tick()

    def test_report_status_missing_grades_blocks_continuation(self):
        add_gpu(self.root, self.scheduler, 4)
        controller = self.controller()
        controller.tick()
        self.scheduler.finish()
        status = c.read_json(self.root / "report/report-status.json")
        status["graded"] -= 1
        write(self.root / "report/report-status.json", status)
        with self.assertRaisesRegex(c.ControllerError, "coverage"):
            controller.tick()

    def test_recorded_submission_reconciles_after_interruption(self):
        add_gpu(self.root, self.scheduler, 5)
        self.scheduler.raise_after = True
        with self.assertRaises(c.ControllerError):
            self.controller().tick()
        self.assertTrue((self.root / "automation/submission-intent.json").exists())
        self.scheduler.raise_after = False
        controller = self.controller()
        self.assertEqual(controller.tick(), "waiting")
        self.assertFalse((self.root / "automation/submission-intent.json").exists())
        self.scheduler.finish()
        self.assertEqual(controller.tick(), "complete")
        self.assertEqual(self.scheduler.calls, ["report"])

    def test_unrecorded_ambiguous_submission_is_not_retried(self):
        self.scheduler.raise_before = True
        with self.assertRaises(c.ControllerError):
            self.controller().tick()
        self.scheduler.raise_before = False
        with self.assertRaisesRegex(c.ControllerError, "Unresolved submission intent"):
            self.controller().tick()
        self.assertEqual(self.scheduler.calls, ["report"])

    def test_unknown_accounting_retries_are_bounded(self):
        self.scheduler.unavailable = True
        waits = []
        with self.assertRaisesRegex(c.ControllerError, "Accounting remained unavailable"):
            self.controller().run(poll_seconds=0, max_accounting_errors=3, sleeper=waits.append)
        self.assertEqual(len(waits), 2)
        self.assertEqual(self.scheduler.calls, [])

    def test_stop_leaves_gpu_jobs_alone(self):
        add_gpu(self.root, self.scheduler, 5)
        self.scheduler.jobs["104"] = c.Job("104", "RUNNING", "0:0")
        controller = self.controller()
        (controller.work / "STOP").touch()
        self.assertEqual(controller.run(), 0)
        self.assertEqual(self.scheduler.jobs["104"].state, "RUNNING")
        self.assertEqual(self.scheduler.calls, [])

    def test_second_controller_does_not_take_lock(self):
        controller = self.controller()
        with (controller.work / "controller.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(c.ControllerError, "Another controller"):
                controller.run()
        self.assertEqual(self.scheduler.calls, [])

    def test_changed_frozen_tool_is_rejected_before_submission(self):
        (self.root / "tools/recover.sh").write_text("changed")
        with self.assertRaisesRegex(c.ControllerError, "helper changed"):
            self.controller()
        self.assertEqual(self.scheduler.calls, [])

    def test_duplicate_gpu_cell_is_rejected(self):
        add_gpu(self.root, self.scheduler, 1)
        append(self.root / "jobs.tsv", ["0.8B", 42, 199])
        with self.assertRaisesRegex(c.ControllerError, "repeated recovery cell"):
            self.controller().tick()

    def test_failed_second_controller_preserves_active_state(self):
        controller = self.controller()
        controller.save(status="active-owner", phase="initial")
        before = controller.state_path.read_bytes()
        with (controller.work / "controller.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.dict(c.os.environ, {"SLURM_JOB_ID": "900"}), patch.object(c.socket, "gethostname", return_value="compute1"):
                self.assertEqual(c.main([str(self.root)]), 2)
        self.assertEqual(controller.state_path.read_bytes(), before)

    def test_controller_reloads_state_after_acquiring_lock(self):
        add_gpu(self.root, self.scheduler, 5)
        earlier = self.controller()
        later = self.controller()
        later.tick()
        self.scheduler.finish()
        module = types.SimpleNamespace(render_report=lambda *args, **kwargs: self.root / "report.md", publish_report=lambda *args: "unused")
        with patch.dict(sys.modules, {"github_report": module}):
            earlier.run(poll_seconds=0, sleeper=lambda _: self.scheduler.finish())
        self.assertEqual(self.scheduler.calls, ["report"])

    def test_complete_report_renders_and_publishes_only_when_requested(self):
        add_gpu(self.root, self.scheduler, 5)
        controller = self.controller()
        calls = []
        module = types.SimpleNamespace(
            render_report=lambda *args, **kwargs: self.root / "automation/public/report.md",
            publish_report=lambda *args: calls.append(args) or "https://github.com/example/project/discussions/6#comment",
        )
        with patch.dict(sys.modules, {"github_report": module}):
            controller.run(poll_seconds=0, sleeper=lambda _: self.scheduler.finish())
            self.assertEqual(calls, [])
            self.controller().run(poll_seconds=0, publish=True, repository="example/project", discussion=6,
                                  sleeper=lambda _: self.scheduler.finish())
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.scheduler.calls, ["report", "report"])


class AccountingTests(unittest.TestCase):
    def test_complete_and_active_states(self):
        result = subprocess.CompletedProcess([], 0, "11|COMPLETED|0:0|\n12|RUNNING|0:0|\n", "")
        with patch.object(subprocess, "run", return_value=result):
            jobs = c.Scheduler().states(["11", "12"])
        self.assertTrue(jobs["11"].success)
        self.assertEqual(jobs["12"].state, "RUNNING")

    def test_absent_or_duplicate_or_truncated_state_is_unknown(self):
        for output in ("11|COMPLETED|0:0|\n", "11|COMPLETED|0:0|\n11|COMPLETED|0:0|\n12|COMPLETED|0:0|\n",
                       "11|OUT_OF_MEM+|0:9|\n12|RUNNING|0:0|\n"):
            with self.subTest(output=output), patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, output, "")):
                with self.assertRaises(c.AccountingError):
                    c.Scheduler().states(["11", "12"])

    def test_timeout_never_means_finished(self):
        with patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("sacct", 40)):
            with self.assertRaises(c.AccountingError):
                c.Scheduler().states(["11"])


if __name__ == "__main__":
    unittest.main()
