#!/usr/bin/env python3
"""Compute-node-only continuation for the known missing 27B cell-index bug.

The original source, task settings, and existing grades remain unchanged. The
only executable overlay changes cell's 0..15 guard to the frozen grid length.
"""

import argparse
import ast
import csv
import hashlib
import importlib
import inspect
import json
import os
import shutil
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path


TARGETS = (16, 17, 18, 19)
OLD_GUARD = '    if not 0 <= index < 16:\n        raise ValueError("Cell index must be in 0..15")'
NEW_GUARD = '    if not 0 <= index < len(cells(suite)):\n        raise ValueError("Cell index is outside the frozen cell grid")'
MODULE_ENTRY = "-m ctxmgr.harness.clm_vllm_suite"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    stage = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    stage.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    stage.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def patch_cell_source(source):
    """Only one exact historical guard is accepted; unknown code fails closed."""
    source = textwrap.dedent(source)
    require(source.count(OLD_GUARD) == 1, "Frozen cell guard is not the known 0..15 bug; inspect its logs before recovery.")
    tree = ast.parse(source)
    require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef) and tree.body[0].name == "cell",
            "Expected exactly the frozen cell function.")
    patched = source.replace(OLD_GUARD, NEW_GUARD, 1)
    compile(patched, "continuation-cell.py", "exec")
    return source, patched


def patch_runtime(source):
    # The frozen runtime has setup, cell and report CLI calls. Only the cell and
    # report modes are reachable from this continuation's worker.
    require(source.count(MODULE_ENTRY) == 3,
            "Frozen runtime has an unknown CLI layout; no GPU cells will run.")
    require('"$client_python" -m ctxmgr.harness.clm_vllm_suite cell' in source,
            "Frozen cell invocation differs from the known runtime.")
    require('"$client_python" -m ctxmgr.harness.clm_vllm_suite report' in source,
            "Frozen report invocation differs from the known runtime.")
    return source.replace(MODULE_ENTRY, '"${CLM_RESUME_RUNNER:?Missing continuation runner}"')


def load_frozen_module(suite):
    source_dir = suite / "source" / "src"
    sys.path.insert(0, str(source_dir))
    module = importlib.import_module("ctxmgr.harness.clm_vllm_suite")
    expected = (source_dir / "ctxmgr" / "harness" / "clm_vllm_suite.py").resolve()
    require(Path(module.__file__).resolve() == expected, "Imported suite code is not this run's frozen source.")
    return module


def read_rows(suite):
    with (suite / "report" / "results.tsv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def validate_missing_grid(module, suite, rows):
    selections = module.cells(suite)
    require(len(selections) == 20, "Continuation requires the original 20-cell grid.")
    require(tuple((x["index"], x["size"], x["seed"]) for x in selections[16:]) ==
            tuple((i, "27B", 42 + i - 16) for i in TARGETS), "Frozen 27B cell identities differ.")
    expected = module.expected_rows(suite)
    key = lambda r: (int(r["cell"]), r["model"], r["task"], int(r["seed"]), r["condition"])
    require(len(rows) == len(expected) == 240 and len({key(r) for r in rows}) == 240,
            "Expected exactly 240 unique episode identities.")
    require({key(r) for r in rows} == {key(r) for r in expected}, "Report identities differ from the frozen grid.")
    require(all(r["status"] == "completed" and r["reward"] != "" for r in rows if int(r["cell"]) < 16),
            "The existing 192 grades are not all valid; this targeted continuation will not overwrite them.")
    require(all(r["status"] == "unrun" and r["reward"] == "" for r in rows if int(r["cell"]) in TARGETS),
            "27B has partial, failed or existing grades; inspect them before a separate recovery.")
    for item in selections[16:]:
        require(not Path(item["root"]).exists(), f"Refusing to overwrite existing cell directory: {item['root']}")


def original_grade_fingerprints(module, suite):
    files = set()
    for item in module.cells(suite)[:16]:
        root = Path(item["root"])
        for pattern in ("result.json", "score.json", "official-run.json", "cell-status.json"):
            files.update(root.rglob(pattern))
    require(files, "No original grade artifacts were found.")
    return {str(path.relative_to(suite)): digest(path) for path in sorted(files)}


def check_fingerprints(suite, fingerprints):
    for relative, expected in fingerprints.items():
        path = suite / relative
        require(path.is_file() and digest(path) == expected, f"Original grade artifact changed: {relative}")


def validate_receipt(module, suite, continuation):
    receipt = read_json(continuation / "preflight.json")
    require(receipt["suite"] == str(suite) and receipt["target_cells"] == list(TARGETS), "Continuation receipt belongs to another grid.")
    for relative, expected in receipt["frozen_artifacts"].items():
        require(digest(suite / relative) == expected, f"Frozen preparation changed: {relative}")
    for name, expected in receipt["overlay_artifacts"].items():
        require(digest(continuation / name) == expected, f"Continuation overlay changed: {name}")
    module.validate_ready(suite)
    check_fingerprints(suite, receipt["original_grade_artifacts"])
    return receipt


def preflight(module, suite, continuation):
    require(not (continuation / "preflight.json").exists(), "Successful preflight already exists; do not repeat it.")
    require(tuple(module.MODEL_SIZES) == ("0.8B", "2B", "4B", "9B", "27B"), "Unexpected model grid.")
    require(tuple(module.SEEDS) == (42, 43, 44, 45), "Unexpected task seeds.")
    require(tuple(module.CONDITIONS) == ("none", "clm_self"), "Unexpected experiment conditions.")
    module.validate_ready(suite, verify_image=True)
    original, patched = patch_cell_source(inspect.getsource(module.cell))
    runtime_path = suite / "source" / "scripts" / "clm_vllm_runtime.sh"
    runtime = patch_runtime(runtime_path.read_text(encoding="utf-8"))
    for item in module.cells(suite)[16:]:
        require(not Path(item["root"]).exists(), f"Refusing an existing 27B cell: {item['root']}")
    # Check 27B's actual cached weights on the CPU before waiting for a GPU.
    module.verify_model(suite / "models" / "Qwen3.5-27B-BF16" / "model.json")
    report_dir = suite / "report"
    if report_dir.exists():
        shutil.copytree(report_dir, continuation / "previous-report")
    result = module.report(suite)
    require(result in (0, 1), "Frozen reporter returned an unexpected status.")
    rows = read_rows(suite)
    validate_missing_grid(module, suite, rows)
    status = read_json(report_dir / "report-status.json")
    require((status["graded"], status["unrun"], status["failed"], status["running"]) == (192, 48, 0, 0),
            "Freshly validated coverage is not exactly 192 grades plus 48 unrun episodes.")
    (continuation / "original-cell.py").write_text(original, encoding="utf-8")
    (continuation / "executed-cell.py").write_text(patched, encoding="utf-8")
    (continuation / "runtime.sh").write_text(runtime, encoding="utf-8")
    artifact_names = ("runner.py", "worker.sh", "launcher.sh", "original-cell.py", "executed-cell.py", "runtime.sh")
    overlay = {name: digest(continuation / name) for name in artifact_names}
    (continuation / "generated.sha256").write_text(
        "".join(f"{overlay[name]}  {name}\n" for name in ("original-cell.py", "executed-cell.py", "runtime.sh")), encoding="utf-8")
    receipt = {
        "schema_version": 1, "created_at": timestamp(), "suite": str(suite), "target_cells": list(TARGETS),
        "original_validated_grades": 192, "planned_additional_grades": 48,
        "patch": {"id": "frozen-cell-index-16-to-grid-length-v1", "before": OLD_GUARD, "after": NEW_GUARD,
                  "scope": "Only cell index validation changes. Original prompts, skills, tasks, sampling, environments and scorer are retained."},
        "frozen_artifacts": {str(path.relative_to(suite)): digest(path) for path in (
            suite / "source.sha256", suite / "config.env", suite / "config.sha256", suite / "ready.json",
            suite / "shared" / "official-run.json", Path(module.__file__), runtime_path)},
        "overlay_artifacts": overlay,
        "original_grade_artifacts": original_grade_fingerprints(module, suite),
    }
    write_json(continuation / "preflight.json", receipt)
    print("Preflight passed: 192 existing grades preserved; only missing 27B cells 16–19 may run.", flush=True)
    return 0


def run_cell(module, suite, continuation, index):
    require(index in TARGETS, "This continuation only permits 27B cell indices 16–19.")
    receipt = validate_receipt(module, suite, continuation)
    item = module.cells(suite)[index]
    root = Path(item["root"])
    require(not root.exists(), f"Refusing to overwrite an existing or partially run cell: {root}")
    actual_original, actual_patched = patch_cell_source(inspect.getsource(module.cell))
    require(actual_original == (continuation / "original-cell.py").read_text(encoding="utf-8"), "Original cell function changed.")
    require(actual_patched == (continuation / "executed-cell.py").read_text(encoding="utf-8"), "Executable cell patch differs.")
    exec(compile(actual_patched, str(continuation / "executed-cell.py"), "exec"), module.__dict__)
    event = {"schema_version": 1, "cell": index, "started_at": timestamp(), "job_id": os.environ.get("SLURM_JOB_ID"),
             "patch": receipt["patch"], "preflight_sha256": digest(continuation / "preflight.json"),
             "executed_cell_sha256": receipt["overlay_artifacts"]["executed-cell.py"], "status": "running"}
    event_path = continuation / f"cell-{index}-provenance.json"
    write_json(event_path, event)
    try:
        result = module.cell(suite, index)
        event.update(status="complete" if result == 0 else "incomplete", return_code=result)
        return result
    except BaseException as exc:
        event.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        event["finished_at"] = timestamp()
        write_json(event_path, event)
        if root.exists():
            write_json(root / "continuation-provenance.json", event)
        check_fingerprints(suite, receipt["original_grade_artifacts"])


def annotate_report(suite, continuation, receipt):
    out = suite / "report"
    provenance = {"schema_version": 1, "preflight_sha256": digest(continuation / "preflight.json"),
                  "target_cells": list(TARGETS), "existing_grades_preserved": 192,
                  "patch": receipt["patch"], "overlay_artifacts": receipt["overlay_artifacts"],
                  "cells": [read_json(path) for path in sorted(continuation.glob("cell-*-provenance.json"))]}
    write_json(out / "continuation-provenance.json", provenance)
    path = out / "provenance.json"
    base = read_json(path)
    base["continuation"] = {"file": "continuation-provenance.json", "sha256": digest(out / "continuation-provenance.json"),
                            "target_cells": list(TARGETS), "patch_id": receipt["patch"]["id"]}
    write_json(path, base)
    report = out / "report.md"
    with report.open("a", encoding="utf-8") as stream:
        stream.write("\n## 27B continuation provenance\n\n"
                     "Cells 16–19 were continued with a recorded runtime-only cell-index guard repair. "
                     "The original frozen source files, task settings, prompts, skills and scorer were retained; "
                     "the previous 192 grade artifacts were verified unchanged. "
                     "See `continuation-provenance.json` for the executed patch and affected cells. "
                     "This note records provenance, not a claim that every continued episode succeeded.\n")
    status_path = out / "report-status.json"
    status = read_json(status_path)
    hashes = status["report_sha256"]
    for name in ("report.md", "provenance.json", "continuation-provenance.json"):
        hashes[name] = digest(out / name)
    status["continuation"] = {"file": "continuation-provenance.json", "target_cells": list(TARGETS)}
    write_json(status_path, status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "cell", "report"))
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args(argv)
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_NODELIST"), "Use the Slurm continuation worker.")
    suite = args.suite.resolve(strict=True)
    continuation = (suite / "continuation-27b-v1").resolve(strict=True)
    require(Path(os.environ["CLM_RESUME_DIR"]).resolve() == continuation, "Continuation directory mismatch.")
    module = load_frozen_module(suite)
    if args.mode == "preflight":
        return preflight(module, suite, continuation)
    if args.mode == "cell":
        return run_cell(module, suite, continuation, args.index)
    receipt = validate_receipt(module, suite, continuation)
    result = module.report(suite)
    check_fingerprints(suite, receipt["original_grade_artifacts"])
    status = annotate_report(suite, continuation, receipt)
    complete = result == 0 and (status["graded"], status["unrun"], status["failed"], status["running"]) == (240, 0, 0, 0)
    (continuation / "state").write_text("complete\n" if complete else "incomplete-see-report-and-cell-logs\n", encoding="utf-8")
    print("Continuation complete: 240/240 validated grades." if complete else
          f"Continuation ended with {status['graded']}/240 validated grades; inspect the preserved logs.", flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
