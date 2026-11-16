"""Pinned ContextBench adapter with exact tokenization and auditable artifacts."""

import hashlib
import importlib
import json
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from contextbench import instruction_for, load_instance, render_op
from contextbench.cli import level_entry, resolve_level
from contextbench.feasibility import evaluate, write_receipt
from contextbench.levels import level_regime
from contextbench.tokens import count_text, encoder_name

UPSTREAM_COMMIT = "fb533c18c78a82d3f1b2047a9c1700735cf829ae"
FAMILIES = ("kv_store", "needle_retention")
BUDGET = 32768
RESERVE = 2048


def require_exact_tokens():
    if encoder_name() != "tiktoken:o200k_base":
        raise RuntimeError(
            "Exact o200k_base tokenizer required; no character/cl100k fallback allowed"
        )


def paper_config() -> dict:
    return yaml.safe_load(files("contextbench").joinpath("configs/paper.yaml").read_text())


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_instance(family: str, level: str, seed: int, out_dir: Path | str) -> dict:
    """Generate official named levels; refuse overwrite and failed feasibility gates."""
    require_exact_tokens()
    if family not in FAMILIES:
        raise ValueError(f"Track 1 supports {FAMILIES}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    config = paper_config()
    knobs = resolve_level(config, family, level)
    regime = level_regime(level_entry(config, family, level))
    module = importlib.import_module(f"contextbench.tasks.{family}.generate")
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Generate and check before publishing a usable artifact directory.
    with TemporaryDirectory(dir=out.parent, prefix=".ctxmgr-build-") as temporary:
        stage = Path(temporary) / "instance"
        module.write_outputs(stage, module.generate(seed=seed, **knobs))
        verdict = evaluate(stage, BUDGET, RESERVE, regime=regime)
        if not verdict["feasible"]:
            raise ValueError(f"Official instance failed feasibility: {verdict['checks']}")
        write_receipt(stage, verdict)
        instance = load_instance(stage)
        instruction = instruction_for(instance)
        operations = [
            render_op(op, remaining=instance.n_ops - i - 1) for i, op in enumerate(instance.ops)
        ]
        public = stage / "public"
        public.mkdir()
        (public / "stream.json").write_text(
            json.dumps(
                {
                    "family": family,
                    "level": level,
                    "seed": seed,
                    "instruction": instruction,
                    "operations": operations,
                },
                indent=2,
            )
            + "\n"
        )
        report = {
            "family": family,
            "level": level,
            "seed": seed,
            "upstream_commit": UPSTREAM_COMMIT,
            "knobs": knobs,
            "token_method": encoder_name(),
            "budget": BUDGET,
            "reserve": RESERVE,
            "regime": regime,
            "gate_pressure": verdict["P"],
            "environment_input_tokens": count_text(instruction) + sum(map(count_text, operations)),
            "environment_pressure": (count_text(instruction) + sum(map(count_text, operations)))
            / BUDGET,
            "max_operation_tokens": verdict["max_op_tokens"],
            "retained_tokens": verdict["retained_tokens"],
            "n_ops": instance.n_ops,
            "checks": verdict["checks"],
            "sha256": {
                name: _digest(stage / name)
                for name in ("chunks.jsonl", "gold.json", "meta.json", "public/stream.json")
            },
        }
        (stage / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        # Destination was absent; never replace existing experiment artifacts.
        out.mkdir()
        for item in stage.iterdir():
            item.rename(out / item.name)
    return report


def validate_instance(instance_dir: Path | str) -> dict:
    require_exact_tokens()
    root = Path(instance_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError("Instance uses a different upstream revision")
    for name in ("chunks.jsonl", "gold.json", "meta.json", "public/stream.json"):
        if manifest["sha256"][name] != _digest(root / name):
            raise ValueError(f"Artifact hash mismatch: {name}")
    verdict = evaluate(root, BUDGET, RESERVE, regime=manifest["regime"])
    if not verdict["feasible"]:
        raise ValueError(f"Infeasible instance: {verdict['checks']}")
    return verdict
