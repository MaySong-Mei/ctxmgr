#!/usr/bin/env python3
"""Render allowlisted native CLM results and explicitly publish to GitHub Discussions.

The frozen recovery validator remains authoritative for episode provenance. This
module checks the exported tables for consistency; it does not regrade snapshots.
GitHub transport uses JSON on stdin, never a shell or command-line credentials.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
from collections import Counter
from contextlib import contextmanager

MODELS = tuple(f"Qwen3.5-{size}-Q8_0" for size in ("0.8B", "2B", "4B"))
TASKS = ("kv_store:R130", "kv_store:R1000", "kv_store:R2000",
         "needle_retention:K4-C4", "needle_retention:K4-C24", "needle_retention:K8-C48")
SEEDS = (42, 43, 44, 45)
CONDITIONS = ("none", "clm_self")
TERMINATIONS = ("stream_exhausted", "budget_wall", "submit", "max_steps")
SAMPLING = {"enable_thinking": True, "max_tokens": 8192, "temperature": 0.7, "top_p": 0.95}
PUBLIC_FIELDS = ("model", "task", "seed", "condition", "status", "reward", "termination",
                 "origin", "original_job_id", "source_job_id", "model_revision", "model_sha256")


class ReportError(ValueError):
    """The supplied report or publication request is unsafe or inconsistent."""


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def table_text(rows: list[dict], fields: tuple | list) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def integer(value, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        raise ReportError(f"Invalid integer in {label}.")
    parsed = int(value)
    if parsed < minimum:
        raise ReportError(f"Invalid integer in {label}.")
    return parsed


def reward(value, *, blank: bool = False) -> float | None:
    if value in ("", None):
        if blank:
            return None
        raise ReportError("A graded episode has no score.")
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ReportError("Invalid reward.") from None
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ReportError("Reward must be finite and between zero and one.")
    return result


def read_table(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ReportError("Missing or duplicate table columns.")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ReportError("Malformed TSV row.")
    return rows


def load_report(report_dir: Path) -> tuple[list[dict], dict, list[dict], str]:
    """Validate the reviewed 144-episode schema without importing private text."""
    status = json.loads((report_dir / "report-status.json").read_text(encoding="utf-8"))
    if status.get("status") not in ("complete", "incomplete"):
        raise ReportError("Recovery validation failed; no publishable report is available.")
    rows = []
    seen = set()
    original_cells = {}
    for source in read_table(report_dir / "results.tsv"):
        if any(name not in source for name in PUBLIC_FIELDS):
            raise ReportError("Results are missing required provenance columns.")
        row = {name: source[name] for name in PUBLIC_FIELDS}
        row["seed"] = integer(row["seed"], "seed")
        key = (row["model"], row["task"], row["seed"], row["condition"])
        if (row["model"] not in MODELS or row["task"] not in TASKS
                or row["seed"] not in SEEDS or row["condition"] not in CONDITIONS):
            raise ReportError("Results do not match the reviewed native CLM experiment grid.")
        if key in seen:
            raise ReportError("Duplicate episode identity.")
        seen.add(key)
        if row["status"] not in ("graded", "failed", "unrun"):
            raise ReportError("Unknown episode status.")
        if row["origin"] not in ("original", "recovery"):
            raise ReportError("Unknown score origin.")
        for name in ("original_job_id", "source_job_id"):
            row[name] = str(integer(row[name], name, minimum=1))
        cell = (row["model"], row["seed"])
        if original_cells.setdefault(cell, row["original_job_id"]) != row["original_job_id"]:
            raise ReportError("Original job identity changes within one model/seed cell.")
        for name, width in (("model_revision", 40), ("model_sha256", 64)):
            if row[name] and not re.fullmatch(rf"[0-9a-f]{{{width}}}", row[name]):
                raise ReportError("Invalid model provenance hash.")
        if row["status"] == "graded":
            row["reward"] = reward(row["reward"])
            if row["termination"] not in TERMINATIONS:
                raise ReportError("Unknown graded termination reason.")
            if not row["model_sha256"] or not row["model_revision"]:
                raise ReportError("A graded episode is missing model provenance.")
        else:
            if row["reward"] not in (None, ""):
                raise ReportError("Missing or failed scores must be blank, never zero.")
            row["reward"] = None
            # Raw failures may contain private paths; no raw error or termination is copied.
            row["termination"] = ""
        rows.append(row)
    expected = {(m, t, s, c) for m in MODELS for t in TASKS for s in SEEDS for c in CONDITIONS}
    if seen != expected:
        raise ReportError("Results must include every planned row, including missing episodes.")
    rows.sort(key=lambda r: (MODELS.index(r["model"]), TASKS.index(r["task"]),
                             r["seed"], CONDITIONS.index(r["condition"])))
    graded = [r for r in rows if r["status"] == "graded"]
    counts = {"planned": len(rows), "graded": len(graded),
              "preserved_original": sum(r["origin"] == "original" for r in graded),
              "recovered": sum(r["origin"] == "recovery" for r in graded),
              "failed": sum(r["status"] == "failed" for r in rows),
              "unrun": sum(r["status"] == "unrun" for r in rows)}
    for name, value in counts.items():
        if integer(status.get(name), name) != value:
            raise ReportError(f"Report status and episode counts disagree for {name}.")
    if counts["preserved_original"] != 88:
        raise ReportError("The reviewed recovery must preserve all 88 original valid grades.")
    complete = counts["graded"] == counts["planned"]
    if status["status"] != ("complete" if complete else "incomplete"):
        raise ReportError("Coverage status disagrees with the episode table.")
    for model in MODELS:
        hashes = {(r["model_revision"], r["model_sha256"]) for r in graded if r["model"] == model}
        if len(hashes) != 1:
            raise ReportError("Model provenance differs across graded episodes.")
    sampling_lines = [line for line in (report_dir / "summary.txt").read_text(encoding="utf-8").splitlines()
                      if line.startswith("Sampling profile:")]
    if len(sampling_lines) != 1 or not sampling_lines[0].startswith("Sampling profile: official; "):
        raise ReportError("The report must identify the reviewed official sampling profile.")
    sampling = json.loads(sampling_lines[0].split("; ", 1)[1])
    if sampling != SAMPLING:
        raise ReportError("Sampling settings differ from the reviewed native CLM recovery.")
    lookup = {(r["model"], r["task"], r["seed"], r["condition"]): r for r in rows}
    comparisons = []
    for model in MODELS:
        for task in TASKS:
            pairs = [(s, lookup[model, task, s, "none"]["reward"],
                      lookup[model, task, s, "clm_self"]["reward"]) for s in SEEDS
                     if all(lookup[model, task, s, c]["status"] == "graded" for c in CONDITIONS)]
            comparisons.append({
                "model": model, "task": task, "expected_pairs": 4,
                "graded_pairs": len(pairs), "excluded_pairs": 4 - len(pairs),
                "paired_task_seeds": ",".join(str(p[0]) for p in pairs),
                "none_mean_paired": statistics.mean(p[1] for p in pairs) if pairs else None,
                "clm_mean_paired": statistics.mean(p[2] for p in pairs) if pairs else None,
                "clm_minus_none_paired": statistics.mean(p[2] - p[1] for p in pairs) if pairs else None,
                "coverage": "complete" if len(pairs) == 4 else "partial" if pairs else "unavailable",
            })
    saved_comparisons = read_table(report_dir / "comparison.tsv")
    if len(saved_comparisons) != len(comparisons):
        raise ReportError("Comparison table does not match the episode grid.")
    saved_by_key = {(r["model"], r["task"]): r for r in saved_comparisons}
    if len(saved_by_key) != len(comparisons):
        raise ReportError("Duplicate comparison row.")
    for calculated in comparisons:
        saved = saved_by_key.get((calculated["model"], calculated["task"]), {})
        for name, value in calculated.items():
            if name.endswith("_paired"):
                raw = saved.get(name)
                if value is None:
                    okay = raw in (None, "")
                else:
                    try:
                        okay = math.isclose(float(raw), value, rel_tol=1e-10, abs_tol=1e-12)
                    except (ValueError, TypeError):
                        okay = False
            else:
                okay = str(saved.get(name)) == str(value)
            if not okay:
                raise ReportError("Comparison table is stale or inconsistent with episode scores.")
    saved_summaries = read_table(report_dir / "summary.tsv")
    summaries_by_key = {(r["model"], r["task"], r["condition"]): r for r in saved_summaries}
    expected_summary_keys = {(m, t, c) for m in MODELS for t in TASKS for c in CONDITIONS}
    if set(summaries_by_key) != expected_summary_keys or len(saved_summaries) != 36:
        raise ReportError("Summary table does not match the episode grid.")
    for group_key, saved in summaries_by_key.items():
        group = [r for r in rows if (r["model"], r["task"], r["condition"]) == group_key]
        scores = [r for r in group if r["status"] == "graded"]
        expected_counts = {
            "expected": 4, "graded": len(scores),
            "failed": sum(r["status"] == "failed" for r in group),
            "unrun": sum(r["status"] == "unrun" for r in group),
            **{t: sum(r["termination"] == t for r in scores) for t in TERMINATIONS[:3]},
        }
        if any(integer(saved.get(k), k) != v for k, v in expected_counts.items()):
            raise ReportError("Summary counts are stale or inconsistent with episode rows.")
        calculated_mean = statistics.mean(r["reward"] for r in scores) if scores else None
        saved_mean = reward(saved.get("mean_reward_graded"), blank=True)
        if ((calculated_mean is None) != (saved_mean is None)
                or calculated_mean is not None and not math.isclose(
                    calculated_mean, saved_mean, rel_tol=1e-10, abs_tol=1e-12)):
            raise ReportError("Summary mean is stale or inconsistent with episode scores.")
    # Original IDs define one experiment and stay stable as new recovery grades arrive.
    identity = sha(json.dumps(sorted((list(k), v) for k, v in original_cells.items())))
    return rows, {"status": status["status"], **counts}, comparisons, identity


def number(value: float | None, signed: bool = False) -> str:
    return "—" if value is None else format(value, "+.4f" if signed else ".4f")


def render_report(report_dir: Path, output_dir: Path, label: str = "Native CLM recovery") -> Path:
    """Write a deterministic public report and manifest; do not contact GitHub."""
    report_dir, output_dir = Path(report_dir), Path(output_dir)
    if output_dir.resolve().is_relative_to(report_dir.resolve()):
        raise ReportError("Public output must be separate from the original recovery report.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._()-]{0,79}", label):
        raise ReportError("Use a short public title containing only letters, digits, spaces, or ._()-.")
    rows, status, comparisons, identity = load_report(report_dir)
    marker = f"<!-- ctxmgr-native-clm-report:v1:{identity} -->"
    lines = [marker, f"<!-- ctxmgr-report-status: {status['status']} -->", "",
             f"### {label}: {status['graded']}/{status['planned']} validated grades", "",
             f"**Coverage: {status['status']}.** Original grades preserved: {status['preserved_original']}; "
             f"recovered grades: {status['recovered']}; failed: {status['failed']}; unrun: {status['unrun']}.", ""]
    if status["status"] != "complete":
        lines += ["This is an intermediate snapshot. Partial seed pairs are not the full-grid comparison.", ""]
    lines += ["Qwen3.5 0.8B/2B/4B, Q8_0; six task levels; task seeds 42–45; "
              "conditions `none` and native `clm_self`. Official sampling: temperature 0.7, "
              "top_p 0.95, max_tokens 8192, thinking enabled.", "",
              "Graded means a valid saved score, not necessarily a completed task stream. "
              "`budget_wall` is early termination; missing scores remain blank rather than zero. "
              "Every mean below uses only matching graded task-seed pairs.", "",
              "| Model | Task | Paired / expected | none | clm_self | clm_self − none |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in comparisons:
        lines.append(f"| {row['model']} | {row['task']} | {row['graded_pairs']}/4 | "
                     f"{number(row['none_mean_paired'])} | {number(row['clm_mean_paired'])} | "
                     f"{number(row['clm_minus_none_paired'], True)} |")
    lines += ["", "<details>", "<summary>Per-seed scores (42, 43, 44, 45)</summary>", "",
              "| Model | Task | Condition | Seed 42 | Seed 43 | Seed 44 | Seed 45 |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    lookup = {(r["model"], r["task"], r["seed"], r["condition"]): r for r in rows}
    for model in MODELS:
        for task in TASKS:
            for condition in CONDITIONS:
                values = " | ".join(number(lookup[model, task, s, condition]["reward"]) for s in SEEDS)
                lines.append(f"| {model} | {task} | {condition} | {values} |")
    lines += ["", "</details>", "", "Termination among graded episodes:", "",
              "| Model | Condition | Graded | stream_exhausted | budget_wall | submit | max_steps |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for model in MODELS:
        for condition in CONDITIONS:
            group = [r for r in rows if r["model"] == model and r["condition"] == condition
                     and r["status"] == "graded"]
            counts = Counter(r["termination"] for r in group)
            values = " | ".join(str(counts[t]) for t in TERMINATIONS)
            lines.append(f"| {model} | {condition} | {len(group)} | {values} |")
    lines += ["", "<details>", "<summary>Model and run provenance</summary>", "",
              "| Model | Model revision | Model SHA-256 |", "| --- | --- | --- |"]
    for model in MODELS:
        row = next(r for r in rows if r["model"] == model and r["status"] == "graded")
        lines.append(f"| {model} | `{row['model_revision']}` | `{row['model_sha256']}` |")
    original_ids = sorted({r["original_job_id"] for r in rows}, key=int)
    recovered_ids = sorted({r["source_job_id"] for r in rows if r["origin"] == "recovery"}, key=int)
    lines += ["", "Original jobs: " + ", ".join(original_ids) + ".",
              "Recovery jobs represented: " + (", ".join(recovered_ids) or "none") + ".",
              "", "</details>", "",
              "Original valid grades take precedence. The frozen recovery validator checks saved metadata "
              "and task identities; this formatter checks table consistency. Neither step regrades "
              "snapshots or proves equality with historical teammate model bytes.", "",
              "Task seeds vary generated tasks, not inference seeds. Shared task structure can correlate "
              "seeds, so no independent-replicate confidence interval or significance test is inferred. "
              "Native CLM prompts, Bash/file tools and budget-final-turn rules differ from the original "
              "team `none` system. Native `clm_self` is also distinct from May's JSON `self` protocol; "
              "these Q8_0 official-sampling results must not be pooled with the vLLM/bf16 runs or "
              "their greedy/recommended-sampling comparisons.", "",
              "Prepared with AI-assisted reporting code. Scientific interpretation and publication "
              "remain subject to human review.", ""]
    markdown = "\n".join(lines)
    if len(markdown.encode("utf-8")) > 60000:
        raise ReportError("Generated discussion report is too large.")
    output_dir.mkdir(parents=True, exist_ok=True)
    public_results = table_text(rows, PUBLIC_FIELDS)
    public_comparisons = table_text(comparisons, list(comparisons[0]))
    atomic_text(output_dir / "public-results.tsv", public_results)
    atomic_text(output_dir / "public-comparison.tsv", public_comparisons)
    atomic_text(output_dir / "report.md", markdown)
    manifest = {"schema": 1, "identity": identity, "marker": marker, "coverage": status,
                "markdown_sha256": sha(markdown), "public_results_sha256": sha(public_results),
                "public_comparison_sha256": sha(public_comparisons)}
    atomic_text(output_dir / "report-manifest.json", json.dumps(manifest, indent=2) + "\n")
    return output_dir / "report.md"


QUERY = """query($owner:String!,$name:String!,$number:Int!,$after:String) {
  viewer { login }
  repository(owner:$owner,name:$name) {
    discussion(number:$number) {
      id url comments(first:100,after:$after) {
        nodes { id body url author { login } }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}"""
ADD = """mutation($id:ID!,$body:String!,$mutation:String!) {
  addDiscussionComment(input:{discussionId:$id,body:$body,clientMutationId:$mutation}) {
    comment { id body url author { login } }
  }
}"""
UPDATE = """mutation($id:ID!,$body:String!,$mutation:String!) {
  updateDiscussionComment(input:{commentId:$id,body:$body,clientMutationId:$mutation}) {
    comment { id body url author { login } }
  }
}"""


def graphql(query: str, variables: dict) -> dict:
    """One request, never automatically retry a potentially completed mutation."""
    try:
        process = subprocess.run(
            ["gh", "api", "graphql", "--hostname", "github.com", "--input", "-"],
            input=json.dumps({"query": query, "variables": variables}), text=True,
            capture_output=True, timeout=45, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ReportError("GitHub request did not confirm success. Re-run publication to reconcile "
                          "the remote marker; do not manually add a duplicate comment.") from None
    if process.returncode:
        raise ReportError("GitHub request failed. Check gh authentication and Discussions permission; "
                          "re-run publication to reconcile any uncertain remote write.")
    try:
        response = json.loads(process.stdout)
    except (ValueError, TypeError):
        raise ReportError("GitHub returned an invalid response; remote write outcome is uncertain.") from None
    if not isinstance(response, dict) or response.get("errors") or not isinstance(response.get("data"), dict):
        raise ReportError("GitHub GraphQL returned errors; no successful publication is confirmed.")
    return response["data"]


@contextmanager
def publication_lock(receipt: Path):
    """Serialize publication sharing one receipt on Linux and Windows."""
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.with_suffix(receipt.suffix + ".lock").open("a+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt
                if stream.tell() == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            raise ReportError("Another publisher is using this receipt. Wait for it to finish.") from None
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def publish_report(markdown_path: Path, repository: str, discussion: int, receipt_path: Path,
                   *, allow_partial: bool = False) -> str:
    """Add or update this experiment's own comment after a full remote marker scan.

    Use one publisher/receipt per experiment. GitHub does not offer a unique-key
    constraint for comments; independent publishers on different hosts can race.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ReportError("Repository must have the form owner/name.")
    discussion = integer(discussion, "discussion", minimum=1)
    markdown_path, receipt_path = Path(markdown_path), Path(receipt_path)
    markdown = markdown_path.read_text(encoding="utf-8")
    manifest = json.loads((markdown_path.parent / "report-manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != 1 or manifest.get("markdown_sha256") != sha(markdown)
            or not re.fullmatch(r"[0-9a-f]{64}", manifest.get("identity", ""))):
        raise ReportError("Public report or manifest changed; regenerate the report before publishing.")
    marker = f"<!-- ctxmgr-native-clm-report:v1:{manifest['identity']} -->"
    if manifest.get("marker") != marker or not markdown.startswith(marker + "\n"):
        raise ReportError("Report marker is invalid.")
    coverage = manifest.get("coverage", {})
    if coverage.get("status") not in ("complete", "incomplete"):
        raise ReportError("Public report has no valid coverage status.")
    planned = integer(coverage.get("planned"), "planned")
    graded_count = integer(coverage.get("graded"), "graded")
    if (planned != 144 or integer(coverage.get("preserved_original"), "preserved_original") != 88
            or not 88 <= graded_count <= planned
            or graded_count != 88 + integer(coverage.get("recovered"), "recovered")
            or planned != graded_count + integer(coverage.get("failed"), "failed")
            + integer(coverage.get("unrun"), "unrun")
            or (coverage["status"] == "complete") != (graded_count == planned)
            or f"<!-- ctxmgr-report-status: {coverage['status']} -->" not in markdown):
        raise ReportError("Public manifest coverage is inconsistent.")
    if coverage["status"] != "complete" and not allow_partial:
        raise ReportError("Partial reports require explicit --allow-partial; nothing was published.")
    owner, name = repository.split("/")
    with publication_lock(receipt_path):
        after = None
        seen_cursors = set()
        matching = []
        viewer = discussion_id = None
        while True:
            data = graphql(QUERY, {"owner": owner, "name": name, "number": discussion, "after": after})
            try:
                page_viewer = data["viewer"]["login"]
                target = data["repository"]["discussion"]
                if not target or not page_viewer:
                    raise KeyError("target")
                if viewer and (page_viewer != viewer or discussion_id != target["id"]):
                    raise ReportError("GitHub identity or discussion changed during pagination.")
                viewer, discussion_id = page_viewer, target["id"]
                comments = target["comments"]
                matching.extend(c for c in comments["nodes"] if marker in c["body"])
                page = comments["pageInfo"]
            except (KeyError, TypeError):
                raise ReportError("Discussion is unavailable or GitHub response is incomplete.") from None
            if not page["hasNextPage"]:
                break
            after = page["endCursor"]
            if not after or after in seen_cursors:
                raise ReportError("GitHub pagination did not advance; nothing was published.")
            seen_cursors.add(after)
        if len(matching) > 1:
            raise ReportError("Multiple matching report comments exist; resolve them before publishing.")
        if matching and (matching[0].get("author") or {}).get("login") != viewer:
            raise ReportError("A different author owns the matching report marker; nothing was changed.")
        if matching and matching[0]["body"] == markdown:
            comment, operation = matching[0], "unchanged"
        else:
            existing = matching[0] if matching else None
            operation = "updated" if existing else "created"
            response = graphql(UPDATE if existing else ADD,
                               {"id": existing["id"] if existing else discussion_id,
                                "body": markdown, "mutation": manifest["markdown_sha256"]})
            try:
                comment = response["updateDiscussionComment" if existing else "addDiscussionComment"]["comment"]
                if (comment["body"] != markdown
                        or (comment.get("author") or {}).get("login") != viewer):
                    raise KeyError("confirmed body")
            except (KeyError, TypeError):
                raise ReportError("GitHub did not confirm the published body. Re-run to reconcile "
                                  "the remote marker before taking further action.") from None
        url = comment.get("url", "")
        if not url.startswith(f"https://github.com/{repository}/discussions/{discussion}#discussioncomment-"):
            raise ReportError("GitHub returned an unexpected comment URL; inspect the discussion.")
        receipt = {"schema": 1, "repository": repository, "discussion": discussion,
                   "identity": manifest["identity"], "markdown_sha256": manifest["markdown_sha256"],
                   "comment_id": comment["id"], "url": url, "operation": operation}
        atomic_text(receipt_path, json.dumps(receipt, indent=2) + "\n")
        return url


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    render = sub.add_parser("render", help="Create a public report without contacting GitHub")
    render.add_argument("--report", required=True, type=Path)
    render.add_argument("--output", required=True, type=Path)
    render.add_argument("--label", default="Native CLM recovery")
    publish = sub.add_parser("publish", help="Explicitly add or update a GitHub Discussion comment")
    publish.add_argument("--markdown", required=True, type=Path)
    publish.add_argument("--repository", required=True)
    publish.add_argument("--discussion", required=True, type=int)
    publish.add_argument("--receipt", required=True, type=Path)
    publish.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "render":
            print(render_report(args.report, args.output, args.label))
        else:
            print(publish_report(args.markdown, args.repository, args.discussion,
                                 args.receipt, allow_partial=args.allow_partial))
    except (ReportError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
