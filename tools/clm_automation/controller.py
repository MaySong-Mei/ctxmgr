#!/usr/bin/env python3
"""Advance an existing native CLM recovery on an allocated CPU node.

The frozen recovery launcher remains the only writer of its job ledgers. This
controller never changes its source, selected episodes, scores, or resources.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
from typing import Callable


ACTIVE = {
    "PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
    "RESIZING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "SIGNALING",
    "STAGE_OUT", "STOPPED",
}
TERMINAL = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED", "REVOKED",
}
REPORT_FILES = ("report-status.json", "results.tsv", "summary.tsv", "comparison.tsv", "summary.txt")


class ControllerError(RuntimeError):
    """Stop without creating additional experiment jobs."""


class AccountingError(ControllerError):
    """Accounting may be temporarily unavailable; do not assume completion."""


@dataclass(frozen=True)
class Job:
    job_id: str
    state: str
    exit_code: str

    @property
    def success(self) -> bool:
        return self.state == "COMPLETED" and self.exit_code == "0:0"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ControllerError(f"Expected an object: {path.name}")
    return value


def write_json(path: Path, value: dict) -> None:
    if path.is_symlink():
        raise ControllerError(f"Refusing an output symlink: {path.name}")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def table(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != fields:
            raise ControllerError(f"Unexpected columns in {path.name}")
        rows = list(reader)
    if any(set(row) != set(fields) or any(value is None for value in row.values()) for row in rows):
        raise ControllerError(f"Malformed record in {path.name}")
    return rows


def file_stamp(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return [stat.st_ino, stat.st_mtime_ns, stat.st_size]


class Scheduler:
    def states(self, identifiers: list[str]) -> dict[str, Job]:
        if not identifiers:
            return {}
        try:
            result = subprocess.run(
                ["sacct", "-X", "-n", "-P", "-j", ",".join(identifiers),
                 "--format=JobIDRaw,State%40,ExitCode"],
                capture_output=True, text=True, timeout=40, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AccountingError("Cannot read Slurm accounting.") from exc
        if result.returncode:
            raise AccountingError("Slurm accounting returned an error.")
        jobs: dict[str, Job] = {}
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 3 or parts[0] not in identifiers:
                continue
            identifier, state, code = parts[:3]
            state = state.split()[0] if state else ""
            if identifier in jobs or state not in ACTIVE | TERMINAL or not re.fullmatch(r"\d+:\d+", code):
                raise AccountingError("Ambiguous Slurm accounting response.")
            jobs[identifier] = Job(identifier, state, code)
        if set(jobs) != set(identifiers):
            raise AccountingError("Some recorded jobs are absent from accounting.")
        return jobs

    def invoke(self, recovery: Path, action: str) -> None:
        try:
            result = subprocess.run(
                ["bash", str(recovery / "tools/recover.sh"), action, str(recovery)],
                text=True, check=False, timeout=600,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ControllerError("Recovery command did not confirm success; inspect the submission journal.") from exc
        if result.returncode:
            raise ControllerError(
                "Recovery command failed. A submitted job may still be held; inspect its ledger and queue."
            )


class Controller:
    def __init__(self, recovery: Path, scheduler: Scheduler | None = None):
        self.root = recovery.resolve()
        self.work = self.root / "automation"
        if self.work.is_symlink():
            raise ControllerError("Automation output must not be a symlink.")
        self.work.mkdir(exist_ok=True)
        self.state_path = self.work / "controller-state.json"
        self.intent_path = self.work / "submission-intent.json"
        self.scheduler = scheduler or Scheduler()
        self.state = read_json(self.state_path) if self.state_path.exists() else {"version": 1, "phase": "initial"}
        if self.state.get("version") != 1:
            raise ControllerError("Unsupported controller state version.")
        self.plan = self.verify_plan()
        self.allowed = {(str(cell["size"]), str(cell["seed"])): cell for cell in self.plan["cells"]}

    def verify_plan(self) -> dict:
        if not (self.root / "plan-ready").is_file():
            raise ControllerError("The frozen recovery plan is not ready.")
        expected = (self.root / "plan.sha256").read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[a-f0-9]{64}", expected) or digest(self.root / "plan.json") != expected:
            raise ControllerError("Frozen recovery plan checksum mismatch.")
        plan = read_json(self.root / "plan.json")
        if plan.get("schema_version") != 1 or plan.get("planned") != 144 or plan.get("preserved") != 88:
            raise ControllerError("This controller requires the reviewed 88-of-144 recovery plan.")
        if digest(self.root / "patch-manifest.json") != plan["patch_manifest_sha256"]:
            raise ControllerError("Frozen recovery patch manifest changed.")
        for name, expected_hash in plan["runtime_files_sha256"].items():
            if name not in {"runtime.env", "original-suite.txt"} or digest(self.root / name) != expected_hash:
                raise ControllerError("Frozen runtime files changed.")
        manifest = read_json(self.root / "patch-manifest.json")
        expected_tools = manifest["tools_sha256"]
        if {p.name for p in (self.root / "tools").iterdir() if p.is_file()} != set(expected_tools):
            raise ControllerError("Frozen recovery tool selection changed.")
        for name, wanted in expected_tools.items():
            if Path(name).name != name or digest(self.root / "tools" / name) != wanted:
                raise ControllerError("Frozen recovery helper changed.")
        cells = plan["cells"]
        if len({(c["size"], c["seed"]) for c in cells}) != len(cells):
            raise ControllerError("Duplicate planned recovery cell.")
        if any(type(c["count"]) is not int or c["count"] <= 0 or c["count"] != len(c["episodes"]) for c in cells):
            raise ControllerError("Invalid planned episode count.")
        if sum(c["count"] for c in cells) != 56:
            raise ControllerError("Unexpected missing episode selection.")
        contents = "".join(f"{c['size']}\t{c['seed']}\t{c['count']}\n" for c in cells)
        if (self.root / "cells.tsv").read_text(encoding="utf-8") != contents:
            raise ControllerError("Recovery cell list differs from the frozen plan.")
        return plan

    def save(self, **changes) -> None:
        self.state.update(changes)
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(self.state_path, self.state)

    def ledgers(self) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        gpu = table(self.root / "jobs.tsv", ["size", "seed", "job_id"])
        cpu = table(self.root / "cpu-jobs.tsv", ["role", "job_id"])
        seen_ids: set[str] = set()
        seen_cells: set[tuple[str, str]] = set()
        for row in gpu + cpu:
            identifier = row["job_id"]
            if not re.fullmatch(r"[0-9]+", identifier) or identifier in seen_ids:
                raise ControllerError("Invalid or duplicate job ID in recovery ledgers.")
            seen_ids.add(identifier)
        for row in gpu:
            cell = row["size"], row["seed"]
            if cell not in self.allowed or cell in seen_cells:
                raise ControllerError("Unplanned or repeated recovery cell.")
            seen_cells.add(cell)
        if any(row["role"] not in {"plan", "snapshot", "report"} for row in cpu):
            raise ControllerError("Unknown role in frozen CPU ledger.")
        return gpu, cpu

    @staticmethod
    def gpu_signature(gpu: list[dict[str, str]]) -> str:
        return hashlib.sha256(json.dumps(gpu, sort_keys=True).encode()).hexdigest()

    def reconcile_intent(self) -> None:
        if not self.intent_path.exists():
            return
        intent = read_json(self.intent_path)
        gpu, cpu = self.ledgers()
        before = intent["ledger_before"]
        ledger = cpu if intent["action"] == "report" else gpu
        if ledger[:len(before)] != before or len(ledger) != len(before) + 1:
            raise ControllerError(
                "Unresolved submission intent. Inspect squeue and both ledgers before removing "
                "automation/submission-intent.json; no automatic retry was attempted."
            )
        row = ledger[-1]
        if intent["action"] == "report":
            if row["role"] != "report":
                raise ControllerError("Submission intent does not match the recorded CPU job.")
            self.save(phase="report_wait", report_job_id=row["job_id"],
                      report_stamp_before=intent["report_stamp_before"],
                      report_gpu_signature=intent["gpu_signature"])
        else:
            if [row["size"], row["seed"]] != intent["cell"]:
                raise ControllerError("Submission intent does not match the recorded GPU cell.")
            self.save(phase="gpu_wait", waiting_job_id=row["job_id"])
        self.intent_path.unlink()

    def submit(self, action: str, gpu: list[dict[str, str]], cpu: list[dict[str, str]], cell=None) -> None:
        if self.intent_path.exists():
            raise ControllerError("An earlier submission intent is unresolved.")
        intent = {"version": 1, "action": action, "ledger_before": cpu if action == "report" else gpu,
                  "gpu_signature": self.gpu_signature(gpu), "cell": list(cell) if cell else None,
                  "report_stamp_before": file_stamp(self.root / "report/report-status.json")}
        write_json(self.intent_path, intent)
        # If the process dies or the scheduler response is ambiguous, the intent
        # remains until a matching ledger entry proves which job was created.
        self.scheduler.invoke(self.root, action)
        self.reconcile_intent()

    def check_jobs(self, gpu, cpu, states) -> bool:
        active = False
        for row in gpu:
            job = states[row["job_id"]]
            if job.state in ACTIVE:
                active = True
            elif not job.success:
                raise ControllerError(f"GPU job {job.job_id} ended {job.state} ({job.exit_code}); automatic recovery stopped.")
        for row in cpu:
            job = states[row["job_id"]]
            if job.state in ACTIVE:
                active = True
            elif not job.success and not (
                row["role"] in {"report", "snapshot"} and job.state == "FAILED" and job.exit_code == "1:0"
            ):
                raise ControllerError(f"CPU job {job.job_id} ended {job.state} ({job.exit_code}); inspect its log.")
        return active

    def validate_report(self, gpu, report_job: Job) -> dict:
        path = self.root / "report/report-status.json"
        if file_stamp(path) == self.state.get("report_stamp_before"):
            raise ControllerError("The report job did not replace the previous report status.")
        status = read_json(path)
        expected_recovered = sum(self.allowed[(row["size"], row["seed"])]["count"] for row in gpu)
        expected_graded = 88 + expected_recovered
        expected_status = "complete" if expected_graded == 144 else "incomplete"
        expected = {"status": expected_status, "planned": 144, "graded": expected_graded,
                    "preserved_original": 88, "recovered": expected_recovered,
                    "failed": 0, "unrun": 144 - expected_graded}
        if status != expected:
            raise ControllerError("Report coverage does not match successful selected cells; no more GPU jobs will be submitted.")
        expected_exit = "0:0" if expected_status == "complete" else "1:0"
        expected_state = "COMPLETED" if expected_status == "complete" else "FAILED"
        if report_job.state != expected_state or report_job.exit_code != expected_exit:
            raise ControllerError("Report accounting and validated coverage disagree.")
        with (self.root / "report/results.tsv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        if len(rows) != 144 or sum(row.get("status") == "graded" for row in rows) != expected_graded:
            raise ControllerError("Detailed report does not match coverage.")
        for ledger_row in gpu:
            cell = self.allowed[(ledger_row["size"], ledger_row["seed"]) ]
            selected = {(ep["model"], f"{ep['family']}:{ep['level']}", str(ep["seed"]), ep["condition"])
                        for ep in cell["episodes"]}
            recovered = [row for row in rows if row.get("origin") == "recovery"
                         and row.get("source_job_id") == ledger_row["job_id"]]
            observed = {(row.get("model"), row.get("task"), row.get("seed"), row.get("condition"))
                        for row in recovered if row.get("status") == "graded"}
            if observed != selected or len(recovered) != cell["count"]:
                raise ControllerError("A recorded GPU cell has missing, invalid, or unexpected selected grades.")
        return status

    def tick(self) -> str:
        """Return waiting, advanced, or complete; never guess missing job state."""
        self.verify_plan()
        self.reconcile_intent()
        gpu, cpu = self.ledgers()
        states = self.scheduler.states([row["job_id"] for row in gpu + cpu])
        if self.check_jobs(gpu, cpu, states):
            self.save(status="waiting", active_jobs=[job.job_id for job in states.values() if job.state in ACTIVE])
            return "waiting"
        signature = self.gpu_signature(gpu)
        if self.state.get("phase") == "report_wait":
            report_id = self.state["report_job_id"]
            if report_id not in states:
                raise ControllerError("The submitted report job is missing from the CPU ledger.")
            if signature != self.state["report_gpu_signature"]:
                raise ControllerError("GPU ledger changed during reporting; inspect concurrent manual operations.")
            status = self.validate_report(gpu, states[report_id])
            hashes = {name: digest(self.root / "report" / name) for name in REPORT_FILES}
            self.save(phase="validated", validated_signature=signature, report_hashes=hashes,
                      validated_status=status, active_jobs=[])
        if self.state.get("phase") in {"validated", "complete"} and self.state.get("validated_signature") == signature:
            if any(digest(self.root / "report" / name) != wanted for name, wanted in self.state["report_hashes"].items()):
                raise ControllerError("Validated report files changed. Inspect them before restarting automation.")
            status = self.state["validated_status"]
            if status["status"] == "complete":
                self.save(status="ready_to_publish")
                return "complete"
            seen = {(row["size"], row["seed"]) for row in gpu}
            next_cell = next((cell for cell in self.allowed if cell not in seen), None)
            if next_cell is None:
                raise ControllerError("All cells have recorded attempts but coverage remains incomplete.")
            self.submit("next", gpu, cpu, next_cell)
        else:
            self.submit("report", gpu, cpu)
        return "advanced"

    def run(self, *, poll_seconds: float = 60, max_accounting_errors: int = 5,
            publish: bool = False, repository: str | None = None, discussion: int | None = None,
            sleeper: Callable[[float], None] = time.sleep) -> int:
        lock_path = self.work / "controller.lock"
        if lock_path.is_symlink():
            raise ControllerError("Controller lock must not be a symlink.")
        with lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ControllerError("Another controller owns this recovery; no work was submitted.") from exc
            # A second process can have initialized before acquiring this lock.
            # Reload only after ownership, and never overwrite an active owner's state.
            self.state = read_json(self.state_path) if self.state_path.exists() else {"version": 1, "phase": "initial"}
            if self.state.get("version") != 1:
                raise ControllerError("Unsupported controller state version.")
            try:
                # Each explicit start revalidates raw results before publication.
                # An interrupted submitted report is resumed, not resubmitted.
                if not self.intent_path.exists() and self.state.get("phase") in {"complete", "validated"}:
                    self.save(phase="initial", status="starting")
                return self._run_locked(poll_seconds=poll_seconds, max_accounting_errors=max_accounting_errors,
                                        publish=publish, repository=repository, discussion=discussion, sleeper=sleeper)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                self.save(status="blocked", message=str(exc))
                raise

    def _run_locked(self, *, poll_seconds, max_accounting_errors, publish, repository, discussion, sleeper) -> int:
        errors = 0
        while True:
            if (self.work / "STOP").exists():
                self.save(status="stopped", message="Stop requested; existing Slurm jobs continue.")
                return 0
            try:
                outcome = self.tick()
                errors = 0
            except AccountingError as exc:
                errors += 1
                self.save(status="accounting_wait", accounting_errors=errors, message=str(exc))
                if errors >= max_accounting_errors:
                    raise ControllerError("Accounting remained unavailable; stopped without submitting new work.") from exc
                outcome = "waiting"
            if outcome == "complete":
                from github_report import publish_report, render_report
                markdown = render_report(self.root / "report", self.work / "public", label="Native CLM recovery")
                changes = {"phase": "complete", "status": "complete", "preview": str(markdown)}
                if publish:
                    if not repository or not discussion:
                        raise ControllerError("Publication requires an explicit repository and discussion number.")
                    changes["discussion_url"] = publish_report(
                        markdown, repository, discussion, self.work / "publication-receipt.json"
                    )
                self.save(**changes)
                print(f"Recovery complete: 144/144 validated grades. Preview: {markdown}", flush=True)
                return 0
            sleeper(poll_seconds)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recovery", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--max-accounting-errors", type=int, default=5)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--repository")
    parser.add_argument("--discussion", type=int)
    args = parser.parse_args(argv)
    if args.poll_seconds < 10 or args.max_accounting_errors < 1:
        parser.error("Poll interval must be at least 10 seconds and accounting retries positive.")
    if args.publish and (not args.repository or not args.discussion or args.discussion < 1):
        parser.error("--publish requires --repository and a positive --discussion.")
    if not re.fullmatch(r"\d+", os.environ.get("SLURM_JOB_ID", "")) or socket.gethostname().split(".")[0].startswith("login"):
        parser.error("Run through the allocated worker job; Python must not run on a login node.")
    controller = None
    try:
        controller = Controller(args.recovery)
        return controller.run(poll_seconds=args.poll_seconds, max_accounting_errors=args.max_accounting_errors,
                              publish=args.publish, repository=args.repository, discussion=args.discussion)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
