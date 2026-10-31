"""Adapter contract tests with fake task/client dependencies; no model inference.

These tests exercise grid/provenance failures without downloading ContextBench,
CLM, or model weights. Production setup additionally verifies pinned upstream
packages and the real worker runs a model protocol probe.
"""

import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def module(name, **values):
    result = types.ModuleType(name)
    result.__dict__.update(values)
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


PAPER_KWARGS = {
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 8192,
    "enable_thinking": True,
    "test_native_setting": "unchanged",
}
LEVELS = {
    "kv_store": ("R130", "R1000", "R2000"),
    "needle_retention": ("K4-C4", "K4-C24", "K8-C48"),
}


class NativeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"

        def generate(family, level, seed, out):
            record = {
                "family": family,
                "level": level,
                "seed": seed,
                "budget": 32768,
                "reserve": 2048,
                "knobs": {"test_level": level},
                "sha256": {"chunks.jsonl": f"{seed:064x}"},
            }
            write_json(Path(out) / "manifest.json", record)
            return record

        dependencies = {
            "ctxmgr.tasks": module(
                "ctxmgr.tasks", generate_instance=generate, validate_instance=lambda _: None
            ),
            "ctxmgr.tasks.core": module(
                "ctxmgr.tasks.core",
                BUDGET=32768,
                RESERVE=2048,
                FAMILIES=tuple(LEVELS),
                paper_config=dict,
            ),
            "ctxmgr.harness.clm_upstream": module(
                "ctxmgr.harness.clm_upstream", AGENT_IMPORT="clm_contextbench:ClmStream"
            ),
            "ctxmgr.harness.clm_grid": module(
                "ctxmgr.harness.clm_grid",
                LEVELS=LEVELS,
                checked_config=dict,
                write_json=write_json,
            ),
            "contextbench.cli": module(
                "contextbench.cli",
                resolve_level=lambda _, family, level: {"test_level": level},
            ),
            "clm_contextbench": module(
                "clm_contextbench",
                PAPER_KWARGS=PAPER_KWARGS,
                PAPER_SKILL_SHA16={family: "1" * 16 for family in LEVELS},
            ),
        }
        self.module_patch = patch.dict(sys.modules, dependencies)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        spec = importlib.util.spec_from_file_location(
            "ctxmgr.harness._adapter_test",
            ROOT / "src/ctxmgr/harness/clm_official.py",
        )
        self.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter)

    def prepare(self, **kwargs):
        return self.adapter.prepare(self.root, **kwargs)

    def model_manifest(self, size="27B"):
        alias, repository, revision = self.adapter.model_identity(size)
        return {
            "model_alias": alias,
            "repository": repository,
            "revision": revision,
            "sha256": "a" * 64,
            "model_sha256_kind": self.adapter.MODEL_SHA256_KIND,
            "backend": "vllm",
            "server_version": "0.30.0",
            "precision": "bfloat16",
        }

    def save_grade(self, plan, row, *, reward=0.75, reason="budget_wall"):
        model = self.model_manifest(row["size"])
        write_json(self.root / "models" / row["model"] / "model.json", model)
        settings = plan["config"]["settings"]
        episode = {
            key: plan[key]
            for key in (
                "backend",
                "server_version",
                "precision",
                "model_sha256_kind",
                "client_transport",
                "profile",
                "budget",
                "reserve",
                "budget_ruler",
                "team_source_commit",
                "team_source_sha256",
            )
        }
        episode.update(
            {key: row[key] for key in ("family", "level", "seed", "condition", "instance_sha256")}
        )
        episode.update(
            actor_model=row["model"],
            served_max_len=plan["config"]["served_max_len"],
            model_revision=model["revision"],
            model_sha256=model["sha256"],
            terminated_by=reason,
            error=None,
        )
        if row["condition"] == "none":
            episode.update(
                implementation="ctxmgr.harness.agents:NoManagementAgent",
                sampling={
                    ("thinking" if key == "enable_thinking" else key): value
                    for key, value in settings["sampling"].items()
                },
            )
        else:
            episode.update(
                implementation="clm_contextbench:ClmStream",
                profile_overrides=settings["profile_overrides"],
                upstream_kwargs=settings["agent_kwargs"],
                sampling={**settings["sampling"], "thinking": True},
            )
            write_json(
                Path(row["out"]) / "harbor/trial/agent/clm_stream.json",
                {
                    "kwargs": {**settings["agent_kwargs"], "model_name": f"openai/{row['model']}"},
                    "skill": "clm",
                    "skill_sha16": "1" * 16,
                },
            )
        write_json(Path(row["out"]) / "episode.json", episode)
        write_json(Path(row["out"]) / "result.json", {"terminated_by": reason})
        write_json(Path(row["out"]) / "score.json", {"official": {"reward": reward}})
        return episode

    def test_full_grid_has_240_fresh_bf16_episodes(self):
        plan = self.prepare()
        self.assertEqual(len(plan["episodes"]), 240)
        self.assertEqual(len(list((self.root / "instances").iterdir())), 24)
        self.assertEqual(plan["models"][-1], "Qwen3.5-27B-BF16")
        self.assertEqual(plan["profile"], "official")
        self.assertEqual(plan["config"]["settings"]["profile_overrides"], {})
        self.assertEqual(plan["agent_kwargs"], PAPER_KWARGS)
        self.assertEqual(self.adapter.load_plan(self.root), plan)
        self.assertEqual(plan["schema_version"], 3)

    def test_defaults_and_27b_request_preserve_native_profile(self):
        parsed = self.adapter.argument_parser().parse_args(["prepare", "--root", str(self.root)])
        self.assertEqual(parsed.sizes, ["0.8B", "2B", "4B", "9B", "27B"])
        self.assertEqual(parsed.seeds, [42, 43, 44, 45])
        self.assertEqual(parsed.profile, "official")
        plan = self.prepare(sizes=["27B"], seeds=[42])
        argv = self.adapter.episode_argv(
            plan["episodes"][1],
            config=plan["config"],
            base_url="http://127.0.0.1:1234/v1",
            image="/test/image.sif",
            revision="a" * 40,
            sha256="b" * 64,
        )
        for flag, value in (
            ("--backend", "vllm"),
            ("--server-version", "0.30.0"),
            ("--precision", "bfloat16"),
            ("--profile", "official"),
        ):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertNotIn("--temperature", argv)
        self.assertNotIn("--no-thinking", argv)

    def test_plan_rejects_old_backend_and_unrecognized_source(self):
        plan = self.prepare(sizes=["27B"], seeds=[42])
        for key, value in (
            ("backend", "llama.cpp"),
            ("server_version", "0.29.0"),
            ("precision", "fp8"),
            ("model_sha256_kind", "gguf-file-sha256"),
            ("team_source_commit", None),
        ):
            damaged = copy.deepcopy(plan)
            damaged[key] = value
            write_json(self.root / "official-run.json", damaged)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.adapter.load_plan(self.root)
        write_json(self.root / "official-run.json", plan)

    def test_existing_run_cannot_be_prepared_again(self):
        self.prepare(sizes=["27B"], seeds=[42])
        before = (self.root / "official-run.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare(sizes=["27B"], seeds=[42])
        self.assertEqual((self.root / "official-run.json").read_bytes(), before)

    def test_model_manifest_rejects_other_precision_revision_or_hash(self):
        manifest = self.model_manifest()
        self.adapter.validate_model_manifest(manifest)
        for key, value in (
            ("revision", "b" * 40),
            ("sha256", "bad"),
            ("precision", "fp8"),
            ("backend", "llama.cpp"),
        ):
            changed = {**manifest, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.adapter.validate_model_manifest(changed)

    def test_summary_counts_early_termination_and_blanks_invalid_scores(self):
        plan = self.prepare(sizes=["27B"], seeds=[42], levels=["kv_store:R130"])
        none, native = plan["episodes"]
        self.save_grade(plan, none, reward=0.0)
        self.save_grade(plan, native, reason="max_steps")
        rows = self.adapter.summary(self.root)
        self.assertEqual([row["status"] for row in rows], ["completed", "completed"])
        self.assertEqual(rows[0]["reward"], 0.0)
        self.assertEqual(rows[1]["termination"], "max_steps")
        episode_path = Path(native["out"]) / "episode.json"
        episode = json.loads(episode_path.read_text())
        episode["precision"] = "fp8"
        write_json(episode_path, episode)
        rows = self.adapter.summary(self.root)
        self.assertEqual(rows[1]["status"], "error")
        self.assertIsNone(rows[1]["reward"])
        self.assertIn("precision", rows[1]["error"])
        raw_score = json.loads((Path(native["out"]) / "score.json").read_text())
        self.assertEqual(raw_score["official"]["reward"], 0.75)

    def test_orphan_score_is_not_a_valid_grade(self):
        plan = self.prepare(sizes=["27B"], seeds=[42], levels=["kv_store:R130"])
        write_json(Path(plan["episodes"][0]["out"]) / "score.json", {"official": {"reward": 1.0}})
        rows = self.adapter.summary(self.root)
        self.assertEqual(rows[0]["status"], "error")
        self.assertIsNone(rows[0]["reward"])
        self.assertEqual(rows[1]["status"], "unrun")
        self.assertIsNone(rows[1]["reward"])

    def test_baseline_keeps_original_agent_with_official_sampling_and_bf16_metadata(self):
        observed = {}

        class Sampling:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class Agent:
            def __init__(self, **kwargs):
                observed["agent"] = kwargs
                self.calls = []

        class Client:
            def __init__(self, **kwargs):
                observed["client"] = kwargs

            def close(self):
                observed["closed"] = True

        def run_episode(agent, instance, out, **kwargs):
            observed["run"] = kwargs
            return {
                "episode": {"terminated_by": "budget_wall", "error": None},
                "score": {"official": {"reward": 0.0}},
            }

        fake_episode = module("ctxmgr.harness.episode", run_episode=run_episode)
        dependencies = {
            "ctxmgr.harness.clm_official": self.adapter,
            "ctxmgr.harness.episode": fake_episode,
            "ctxmgr.harness.agents": module(
                "ctxmgr.harness.agents", NoManagementAgent=Agent, Sampling=Sampling
            ),
            "openai": module("openai", OpenAI=Client),
        }
        import ctxmgr.harness

        with (
            patch.dict(sys.modules, dependencies),
            patch.object(ctxmgr.harness, "episode", fake_episode, create=True),
        ):
            spec = importlib.util.spec_from_file_location(
                "ctxmgr.harness._baseline_test",
                ROOT / "src/ctxmgr/harness/clm_baseline_eval.py",
            )
            baseline = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(baseline)
            out = self.root / "baseline"
            result = baseline.run_baseline(
                self.root / "instance",
                out,
                model=self.model_manifest(),
                base_url="http://127.0.0.1:1234/v1",
            )
        self.assertEqual(result, 0)
        self.assertEqual(
            vars(observed["agent"]["sampling"]),
            {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 8192,
                "thinking": True,
            },
        )
        self.assertEqual(observed["run"]["condition"], "none")
        self.assertEqual(observed["run"]["precision"], "bfloat16")
        metadata = json.loads((out / "episode.json").read_text())
        self.assertEqual(metadata["model_sha256_kind"], "hf-snapshot-tree-sha256")
        self.assertEqual(metadata["backend"], "vllm")
        self.assertEqual(metadata["server_version"], "0.30.0")
        self.assertTrue(observed["closed"])


if __name__ == "__main__":
    unittest.main()
