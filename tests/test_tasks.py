"""Integration tests for the pinned benchmark and our handoff contract."""

import json

import pytest
from contextbench import load_instance, run
from contextbench.oracle import OracleAgent
from contextbench.tokens import record_snapshot

from ctxmgr.tasks import generate_instance, paper_config, score_run, validate_instance

LEVELS = [
    (family, entry["name"])
    for family, spec in paper_config()["families"].items()
    if family in ("kv_store", "needle_retention")
    for entry in spec["levels"]
    if entry.get("main_figure")
]


@pytest.mark.parametrize("family,level", LEVELS)
def test_official_levels_feasible_and_oracle_passes(tmp_path, family, level):
    instance_dir = tmp_path / "instance"
    manifest = generate_instance(family, level, 42, instance_dir)
    assert manifest["token_method"] == "tiktoken:o200k_base"
    assert all(check["pass"] for check in manifest["checks"].values())
    assert validate_instance(instance_dir)["feasible"]
    result = run(OracleAgent(instance_dir), instance_dir, out_dir=tmp_path / "run")
    assert result["reward"] == 1
    graded = score_run(instance_dir, tmp_path / "run")
    assert graded["official"]["reward"] == 1
    assert not graded["diagnostics"]["snapshot_budget_exceeded"]
    if family == "needle_retention":
        assert graded["diagnostics"]["required_info_loss"] == 0
    else:
        assert graded["diagnostics"]["strict_exact_accuracy"] == 1


def test_reproducibility_and_public_stream(tmp_path):
    a, b, c = (tmp_path / name for name in ("a", "b", "c"))
    first = generate_instance("kv_store", "R130", 42, a)
    second = generate_instance("kv_store", "R130", 42, b)
    third = generate_instance("kv_store", "R130", 43, c)
    assert first["sha256"] == second["sha256"]
    assert first["sha256"]["chunks.jsonl"] != third["sha256"]["chunks.jsonl"]
    public = json.loads((a / "public/stream.json").read_text())
    assert set(public) == {"family", "level", "seed", "instruction", "operations"}
    assert len(public["operations"]) == 26
    assert public["operations"][0].startswith("=== set batch")
    assert public["operations"][-1].startswith("=== get 23 (0 op(s) remaining")
    with pytest.raises(FileExistsError):
        generate_instance("kv_store", "R130", 42, a)


def test_artifact_tampering_detected(tmp_path):
    generate_instance("kv_store", "R130", 42, tmp_path / "instance")
    p = tmp_path / "instance/gold.json"
    p.write_text(p.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_instance(p.parent)


def test_fallback_tokenizer_refused(monkeypatch):
    from ctxmgr.tasks import core

    monkeypatch.setattr(core, "encoder_name", lambda: "approx_chars3")
    with pytest.raises(RuntimeError, match="o200k"):
        core.require_exact_tokens()


def test_snapshot_source_required_and_corruption_refused(tmp_path):
    d, r = tmp_path / "instance", tmp_path / "run"
    generate_instance("kv_store", "R130", 42, d)
    r.mkdir()
    (r / "final_report.txt").write_text("a report cannot replace snapshots")
    with pytest.raises(ValueError, match="snapshots required"):
        score_run(d, r)
    p = record_snapshot(r, 1, [{"role": "assistant", "content": "nothing retained"}])
    assert score_run(d, r)["official"]["reward"] == 0
    snapshot = json.loads(p.read_text())
    snapshot["tokens"] += 1
    p.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="token count"):
        score_run(d, r)


def test_kv_process_metric_and_strict_diagnostic(tmp_path):
    d, r = tmp_path / "instance", tmp_path / "run"
    generate_instance("kv_store", "R130", 42, d)
    queries = load_instance(d).gold["queries"]
    text = "\n".join(
        f"<<<ANSWER key={q['key']}>>>\nextra {q['value']} extra\n<<<ANSWER END>>>" for q in queries
    )
    record_snapshot(r, 1, [{"role": "assistant", "content": text}])
    record_snapshot(r, 2, [{"role": "assistant", "content": "answers compacted away"}])
    result = score_run(d, r)
    # Official KV metric credits a correct substring in ANY turn, even if later removed.
    assert result["official"]["reward"] == 1
    assert result["diagnostics"]["strict_exact_accuracy"] == 0


def test_needles_only_final_context_counts(tmp_path):
    d, r = tmp_path / "instance", tmp_path / "run"
    generate_instance("needle_retention", "K4-C4", 42, d)
    lines = load_instance(d).gold["answer_lines"]
    record_snapshot(r, 1, [{"role": "assistant", "content": "\n".join(lines)}])
    record_snapshot(r, 2, [{"role": "assistant", "content": lines[0]}])
    result = score_run(d, r)
    assert result["official"]["reward"] == 1 / len(lines)
    assert result["diagnostics"]["required_info_loss"] == 1 - 1 / len(lines)


def test_cli(tmp_path, monkeypatch, capsys):
    import sys

    from ctxmgr.tasks.__main__ import main

    d, r = tmp_path / "instance", tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        ["tasks", "generate", "kv_store", "--level", "R130", "--seed", "42", "--out", str(d)],
    )
    main()
    assert json.loads(capsys.readouterr().out)["family"] == "kv_store"
    monkeypatch.setattr(sys, "argv", ["tasks", "oracle", str(d), "--run-dir", str(r)])
    main()
    assert json.loads(capsys.readouterr().out)["official"]["reward"] == 1


def test_gold_matches_inputs_without_using_oracle(tmp_path):
    import re

    for family, level in (("kv_store", "R130"), ("needle_retention", "K4-C4")):
        d = tmp_path / family
        generate_instance(family, level, 43, d)
        instance = load_instance(d)
        public = json.loads((d / "public/stream.json").read_text())
        if family == "kv_store":
            store = {}
            query_keys = []
            for operation in public["operations"]:
                store.update(re.findall(r"^SET (K\d+) = (.+)$", operation, re.MULTILINE))
                query_keys.extend(re.findall(r"^GET (K\d+)$", operation, re.MULTILINE))
            assert len(store) == 130
            assert query_keys == [q["key"] for q in instance.gold["queries"]]
            assert all(store[q["key"]] == q["value"] for q in instance.gold["queries"])
        else:
            lines = [
                line
                for operation in public["operations"]
                for line in operation.splitlines()
                if line.startswith("[n")
            ]
            assert lines == instance.gold["answer_lines"]


def test_wrong_kv_value_gets_no_credit(tmp_path):
    d, r = tmp_path / "instance", tmp_path / "run"
    generate_instance("kv_store", "R130", 42, d)
    query = load_instance(d).gold["queries"][0]
    record_snapshot(
        r,
        1,
        [
            {
                "role": "assistant",
                "content": f"<<<ANSWER key={query['key']}>>>\nincorrect\n<<<ANSWER END>>>",
            }
        ],
    )
    result = score_run(d, r)
    assert result["official"]["reward"] == 0
    assert result["diagnostics"]["strict_exact_accuracy"] == 0
