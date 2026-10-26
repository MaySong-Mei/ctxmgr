"""Synthetic table and mocked GitHub tests; never submit jobs or send messages."""
from pathlib import Path
import copy
import importlib.util
import json
import statistics
import subprocess
import tempfile
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools/clm_automation/github_report.py"
spec = importlib.util.spec_from_file_location("github_report_under_test", MODULE_PATH)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write_fixture(root, partial=False):
    rows = []
    for mi, model in enumerate(report.MODELS):
        for ti, task in enumerate(report.TASKS):
            for si, seed in enumerate(report.SEEDS):
                for ci, condition in enumerate(report.CONDITIONS):
                    index = len(rows)
                    missing = partial and index >= 132
                    original = index < 88
                    rows.append({
                        "model": model, "task": task, "seed": seed, "condition": condition,
                        "status": "unrun" if missing else "graded",
                        "reward": "" if missing else (si + ci) / 8,
                        "termination": "" if missing else "budget_wall" if ti % 2 else "stream_exhausted",
                        "origin": "original" if original else "recovery",
                        "original_job_id": str(1000 + mi * 10 + si),
                        "source_job_id": str(1000 + mi * 10 + si if original else 2000 + mi * 10 + si),
                        "model_revision": str(mi + 1) * 40,
                        "model_sha256": str(mi + 1) * 64,
                        "error": "/private/person/secrets/token.txt",
                    })
    write_rows(root, rows)
    return rows


def write_rows(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    (root / "results.tsv").write_text(report.table_text(rows, list(rows[0])), encoding="utf-8")
    graded = [r for r in rows if r["status"] == "graded"]
    status = {"status": "complete" if len(graded) == 144 else "incomplete", "planned": len(rows),
              "graded": len(graded), "preserved_original": sum(r["origin"] == "original" for r in graded),
              "recovered": sum(r["origin"] == "recovery" for r in graded),
              "failed": sum(r["status"] == "failed" for r in rows),
              "unrun": sum(r["status"] == "unrun" for r in rows)}
    (root / "report-status.json").write_text(json.dumps(status), encoding="utf-8")
    (root / "summary.txt").write_text("Sampling profile: official; " + json.dumps(report.SAMPLING) + "\n", encoding="utf-8")
    comparisons = []
    summaries = []
    for model in report.MODELS:
        for task in report.TASKS:
            matched = []
            for seed in report.SEEDS:
                pair = [r for r in rows if r["model"] == model and r["task"] == task and r["seed"] == seed]
                if len(pair) == 2 and all(r["status"] == "graded" for r in pair):
                    matched.append((seed, pair[0]["reward"], pair[1]["reward"]))
            comparisons.append({"model": model, "task": task, "expected_pairs": 4,
                                "graded_pairs": len(matched), "excluded_pairs": 4 - len(matched),
                                "paired_task_seeds": ",".join(str(p[0]) for p in matched),
                                "none_mean_paired": statistics.mean(p[1] for p in matched) if matched else None,
                                "clm_mean_paired": statistics.mean(p[2] for p in matched) if matched else None,
                                "clm_minus_none_paired": statistics.mean(p[2]-p[1] for p in matched) if matched else None,
                                "coverage": "complete" if len(matched) == 4 else "partial" if matched else "unavailable"})
            for condition in report.CONDITIONS:
                group = [r for r in rows if r["model"] == model and r["task"] == task and r["condition"] == condition]
                scores = [r for r in group if r["status"] == "graded"]
                summaries.append({"model": model, "task": task, "condition": condition,
                                  "expected": len(group), "graded": len(scores),
                                  "failed": sum(r["status"] == "failed" for r in group),
                                  "unrun": sum(r["status"] == "unrun" for r in group),
                                  "mean_reward_graded": statistics.mean(r["reward"] for r in scores) if scores else None,
                                  **{t: sum(r["termination"] == t for r in scores) for t in report.TERMINATIONS[:3]}})
    (root / "comparison.tsv").write_text(report.table_text(comparisons, list(comparisons[0])), encoding="utf-8")
    (root / "summary.tsv").write_text(report.table_text(summaries, list(summaries[0])), encoding="utf-8")


def page(comments=(), more=False, cursor=None):
    return {"viewer": {"login": "reviewer"}, "repository": {"discussion": {
        "id": "D_test", "url": "https://github.com/example/project/discussions/6",
        "comments": {"nodes": list(comments), "pageInfo": {"hasNextPage": more, "endCursor": cursor}}}}}


def comment(body, author="reviewer"):
    return {"id": "DC_test", "body": body, "author": {"login": author},
            "url": "https://github.com/example/project/discussions/6#discussioncomment-7"}


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input"
        self.output = self.root / "public"
        self.rows = write_fixture(self.input)

    def render(self):
        return report.render_report(self.input, self.output)

    def test_complete_render_scores_and_no_private_error(self):
        path = self.render()
        body = path.read_text()
        self.assertIn("144/144 validated grades", body)
        self.assertIn("0.1875 | 0.3125 | +0.1250", body)
        self.assertIn("0.0000", body)
        for public in self.output.iterdir():
            self.assertNotIn("/private/person", public.read_text())
        self.assertIn("thinking enabled", body)
        self.assertIn("not necessarily a completed task stream", body)

    def test_partial_coverage_and_blank_not_zero(self):
        write_fixture(self.input, partial=True)
        body = self.render().read_text()
        self.assertIn("132/144", body)
        self.assertIn("Coverage: incomplete", body)
        self.assertIn("—", body)
        with patch.object(report, "graphql") as remote:
            with self.assertRaisesRegex(report.ReportError, "Partial reports"):
                report.publish_report(self.output / "report.md", "example/project", 6, self.root / "receipt.json")
            remote.assert_not_called()

    def test_missing_score_rejected(self):
        self.rows[-1]["reward"] = ""
        # Avoid recomputing aggregates from malformed data.
        (self.input / "results.tsv").write_text(report.table_text(self.rows, list(self.rows[0])))
        with self.assertRaisesRegex(report.ReportError, "no score"):
            self.render()

    def test_duplicate_episode_rejected(self):
        self.rows[-1] = dict(self.rows[0])
        (self.input / "results.tsv").write_text(report.table_text(self.rows, list(self.rows[0])))
        with self.assertRaisesRegex(report.ReportError, "Duplicate episode"):
            self.render()

    def test_stale_status_rejected(self):
        status_path = self.input / "report-status.json"
        status = json.loads(status_path.read_text())
        status["graded"] = 143
        status_path.write_text(json.dumps(status))
        with self.assertRaisesRegex(report.ReportError, "counts disagree"):
            self.render()

    def test_stale_comparison_rejected(self):
        table = report.read_table(self.input / "comparison.tsv")
        table[0]["clm_mean_paired"] = "0.999"
        (self.input / "comparison.tsv").write_text(report.table_text(table, list(table[0])))
        with self.assertRaisesRegex(report.ReportError, "stale or inconsistent"):
            self.render()

    def test_stale_summary_rejected(self):
        table = report.read_table(self.input / "summary.tsv")
        table[0]["mean_reward_graded"] = "0.999"
        (self.input / "summary.tsv").write_text(report.table_text(table, list(table[0])))
        with self.assertRaisesRegex(report.ReportError, "Summary mean is stale"):
            self.render()

    def test_failed_rows_cannot_silently_become_zero(self):
        self.rows[-1]["status"] = "failed"
        self.rows[-1]["reward"] = 0.0
        write_rows(self.input, self.rows)
        with self.assertRaisesRegex(report.ReportError, "must be blank"):
            self.render()

    def test_source_report_is_not_an_output_directory(self):
        with self.assertRaisesRegex(report.ReportError, "separate"):
            report.render_report(self.input, self.input / "public")

    def test_nonfinite_and_private_metadata_rejected(self):
        for field, value in (("reward", "NaN"), ("model_sha256", "/private/path"), ("task", "@someone")):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                rows[0][field] = value
                (self.input / "results.tsv").write_text(report.table_text(rows, list(rows[0])))
                with self.assertRaises(report.ReportError):
                    self.render()

    def test_changed_profile_rejected(self):
        (self.input / "summary.txt").write_text('Sampling profile: official; {"temperature": 0}\n')
        with self.assertRaisesRegex(report.ReportError, "Sampling settings differ"):
            self.render()

    def test_identity_stable_across_recovery(self):
        self.render()
        identity = json.loads((self.output / "report-manifest.json").read_text())["identity"]
        write_fixture(self.input, partial=True)
        self.render()
        self.assertEqual(identity, json.loads((self.output / "report-manifest.json").read_text())["identity"])

    def test_generated_body_tampering_blocks_publication(self):
        path = self.render()
        path.write_text(path.read_text() + "Edited text")
        with patch.object(report, "graphql") as remote:
            with self.assertRaisesRegex(report.ReportError, "changed"):
                report.publish_report(path, "example/project", 6, self.root / "receipt.json")
            remote.assert_not_called()

    def test_publish_create_then_idempotent_remote_reconciliation(self):
        path = self.render()
        body = path.read_text()
        receipt = self.root / "receipt.json"
        with patch.object(report, "graphql", side_effect=[page(), {"addDiscussionComment": {"comment": comment(body)}}]) as remote:
            url = report.publish_report(path, "example/project", 6, receipt)
            self.assertEqual(remote.call_count, 2)
            self.assertIn("#discussioncomment-7", url)
        # The remote marker is sufficient even after losing the local receipt.
        receipt.unlink()
        with patch.object(report, "graphql", return_value=page([comment(body)])) as remote:
            report.publish_report(path, "example/project", 6, receipt)
            self.assertEqual(remote.call_count, 1)
        self.assertEqual(json.loads(receipt.read_text())["operation"], "unchanged")

    def test_publish_finds_later_page_and_updates_own_comment(self):
        path = self.render()
        body = path.read_text()
        old = body.replace("144/144 validated grades", "132/144 validated grades")
        responses = [page([], True, "cursor2"), page([comment(old)]),
                     {"updateDiscussionComment": {"comment": comment(body)}}]
        with patch.object(report, "graphql", side_effect=responses) as remote:
            report.publish_report(path, "example/project", 6, self.root / "receipt.json")
            self.assertEqual(remote.call_args_list[1].args[1]["after"], "cursor2")
            self.assertEqual(remote.call_args_list[2].args[0], report.UPDATE)

    def test_foreign_marker_and_duplicate_markers_block_writes(self):
        path = self.render()
        for comments in ([comment(path.read_text(), "other")], [comment(path.read_text())] * 2):
            with patch.object(report, "graphql", return_value=page(comments)) as remote:
                with self.assertRaises(report.ReportError):
                    report.publish_report(path, "example/project", 6, self.root / "receipt.json")
                self.assertEqual(remote.call_count, 1)

    def test_lost_mutation_response_does_not_retry_write(self):
        path = self.render()
        receipt = self.root / "receipt.json"
        with patch.object(report, "graphql", side_effect=[page(), report.ReportError("Uncertain write")]) as remote:
            with self.assertRaises(report.ReportError):
                report.publish_report(path, "example/project", 6, receipt)
            self.assertEqual(remote.call_count, 2)
            self.assertFalse(receipt.exists())
        with patch.object(report, "graphql", return_value=page([comment(path.read_text())])) as remote:
            report.publish_report(path, "example/project", 6, receipt)
            self.assertEqual(remote.call_count, 1)

    def test_graphql_errors_even_with_zero_process_exit(self):
        process = subprocess.CompletedProcess([], 0, json.dumps({"errors": [{"message": "Denied"}], "data": {}}), "")
        with patch.object(report.subprocess, "run", return_value=process) as run:
            with self.assertRaisesRegex(report.ReportError, "GraphQL returned errors"):
                report.graphql(report.QUERY, {"owner": "example"})
            args, kwargs = run.call_args
            self.assertEqual(args[0], ["gh", "api", "graphql", "--hostname", "github.com", "--input", "-"])
            self.assertEqual(json.loads(kwargs["input"])["variables"], {"owner": "example"})
            self.assertNotIn("shell", kwargs)

    def test_timeout_has_no_automatic_retry(self):
        with patch.object(report.subprocess, "run", side_effect=subprocess.TimeoutExpired("gh", 45)) as run:
            with self.assertRaisesRegex(report.ReportError, "did not confirm success"):
                report.graphql(report.ADD, {})
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
