"""CLI for synthetic data augmentation.

Example:
    python -m dqe.augment.cli --dataset piper_dataset_1100 --out out \
        --curated out/curated/keep_episodes.json --per-episode 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from . import synthesize


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Data Q Engine — synthetic augmentation")
    p.add_argument("--dataset", required=True, help="LeRobot dataset root")
    p.add_argument("--out", default="out", help="output directory")
    p.add_argument("--curated", default=None,
                   help="keep_episodes.json — only augment these episodes")
    p.add_argument("--per-episode", type=int, default=2,
                   help="transforms sampled per source episode")
    p.add_argument("--threshold", type=float, default=0.65,
                   help="keep synthetic variants scoring >= this")
    p.add_argument("--transforms", default=None,
                   help="comma list: jitter,time_warp,crop_idle,gripper_jitter")
    p.add_argument("--task-backend", choices=["offline", "claude"], default="offline",
                   help="instruction paraphrase backend")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    keep_list = None
    if args.curated:
        with open(args.curated, "r", encoding="utf-8") as f:
            keep_list = json.load(f).get("kept_episodes")

    transforms = args.transforms.split(",") if args.transforms else None
    t0 = time.time()
    summary = synthesize.run(
        args.dataset, args.out, transforms=transforms,
        per_episode=args.per_episode, threshold=args.threshold, seed=args.seed,
        keep_list=keep_list, task_backend=args.task_backend,
        verbose=not args.quiet)
    if not args.quiet:
        print("Generated %d synthetic episodes (%.0f%% of %d attempts) in %.1fs" % (
            summary["kept"], 100 * summary["accept_rate"],
            summary["attempted"], time.time() - t0))
        print("Output: %s" % summary["out_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
