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
    args = parser.parse_args()
    seeds = [50] if args.smoke else [50, 51, 52, 53]
    levels = [("needle_retention", "K4-C24")]
    if args.actor in ("9B", "27B") and not args.smoke:
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
            random.Random(seed + family).shuffle(conditions)
            for cond in conditions:
                for url in set(
                    [args.base_url, args.manager_url] if args.manager_url else [args.base_url]
                ):
                    reset(url)
                out = args.out / f"Qwen3.5-{args.actor}" / cond / name
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
                if cond.startswith("peer_27b"):
                    command += [
                        "--manager-base-url",
                        args.manager_url,
                        "--manager-model",
                        "Qwen/Qwen3.5-27B",
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
