"""CPU-only audit across official core levels: python -m ctxmgr.tasks.audit --out ..."""

import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from contextbench import run
from contextbench.oracle import OracleAgent

from .core import FAMILIES, generate_instance, paper_config
from .scoring import score_run


def audit(seeds: list[int]) -> list[dict]:
    rows = []
    for family in FAMILIES:
        for entry in paper_config()["families"][family]["levels"]:
            if not entry.get("main_figure"):
                continue
            for seed in seeds:
                with TemporaryDirectory(prefix="ctxmgr-audit-") as temporary:
                    root = Path(temporary)
                    manifest = generate_instance(family, entry["name"], seed, root / "instance")
                    result = run(
                        OracleAgent(root / "instance"), root / "instance", out_dir=root / "run"
                    )
                    graded = score_run(root / "instance", root / "run")
                    if result["reward"] != 1 or graded["diagnostics"]["snapshot_budget_exceeded"]:
                        raise RuntimeError(f"Oracle audit failed: {family} {entry['name']} {seed}")
                    rows.append(
                        {
                            key: manifest[key]
                            for key in (
                                "family",
                                "level",
                                "seed",
                                "token_method",
                                "gate_pressure",
                                "environment_pressure",
                                "max_operation_tokens",
                                "retained_tokens",
                                "n_ops",
                            )
                        }
                        | {
                            "oracle_reward": result["reward"],
                            "oracle_peak_tokens": graded["diagnostics"]["observed_peak_tokens"],
                        }
                    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = audit(args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"PASS: {len(rows)} official task instances; all oracle scores 1.0; {args.out}")


if __name__ == "__main__":
    main()
