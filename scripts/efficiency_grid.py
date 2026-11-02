"""Small, cache-reset, serial timing pilot using the existing benchmark and edit protocol."""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def reset(url):
    request = urllib.request.Request(url.removesuffix("/v1") + "/reset_prefix_cache", method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError("Prefix-cache reset failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--manager-url")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--flipped-manager",
        help="flipped roles: smaller manager size (e.g. 2B) for a 27B actor; runs peer_small only",
    )
    args = parser.parse_args()
    if args.flipped_manager and (args.actor != "27B" or not args.manager_url):
        parser.error("--flipped-manager needs --actor 27B and --manager-url")
    seeds = [50] if args.smoke else [50, 51, 52, 53]
    levels = [("needle_retention", "K4-C24")]
    if (args.actor in ("9B", "27B") and not args.smoke) or args.flipped_manager:
        levels.append(("kv_store", "R1000"))
    for family, level in levels:
        for seed in seeds:
            name = f"{family}-{level}-s{seed}"
            instance = args.out / "instances" / name
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ctxmgr.tasks",
                    "generate",
                    family,
                    "--level",
                    level,
                    "--seed",
                    str(seed),
                    "--out",
                    str(instance),
                ],
                check=True,
            )
            if family == "needle_retention":
                conditions = (
                    ["none", "self", "peer_same"]
                    if args.actor == "27B"
                    else ["none", "self", "peer_27b"]
                )
                if args.smoke:
                    conditions = ["none", "peer_27b"]
            else:
                conditions = ["none_r", "self_arc", "mask_arc"]
                if args.actor != "27B":
                    conditions.append("peer_27b_arc")
            if args.flipped_manager:
                conditions = ["peer_small" if family == "needle_retention" else "peer_small_arc"]
            random.Random(f"{family}:{seed}").shuffle(conditions)
            for cond in conditions:
                for url in set(
                    [args.base_url, args.manager_url] if args.manager_url else [args.base_url]
                ):
                    reset(url)
                arm = f"{cond}-{args.flipped_manager}" if args.flipped_manager else cond
                out = args.out / f"Qwen3.5-{args.actor}" / arm / name
                command = [
                    sys.executable,
                    "-m",
                    "ctxmgr.harness",
                    "run",
                    str(instance),
                    "--out",
                    str(out),
                    "--condition",
                    cond,
                    "--base-url",
                    args.base_url,
                    "--model",
                    f"Qwen/Qwen3.5-{args.actor}",
                    "--served-max-len",
                    "81920",
                    "--precision",
                    "bf16",
                    "--trigger",
                    "18432",
                    "--manager-max-tokens",
                    "4096",
                    "--max-turns-per-op",
                    "2",
                ]
                if cond.startswith(("peer_27b", "peer_small")):
                    manager = args.flipped_manager if cond.startswith("peer_small") else "27B"
                    command += [
                        "--manager-base-url",
                        args.manager_url,
                        "--manager-model",
                        f"Qwen/Qwen3.5-{manager}",
                    ]
                started = time.time()
                subprocess.run(command, check=True)
                (out / "timing.json").write_text(
                    json.dumps(
                        {
                            "started_unix_s": started,
                            "finished_unix_s": time.time(),
                            "job_id": os.environ.get("SLURM_JOB_ID"),
                            "cache_policy": "reset before episode; within-episode prefix reuse enabled",
                            "subprocess_includes_evaluation": True,
                        },
                        indent=2,
                    )
                )


if __name__ == "__main__":
    main()
