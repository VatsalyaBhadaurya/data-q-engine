"""Command-line entry point for the Data Q Engine.

Example:
    python -m dqe.cli --dataset piper_dataset_1100 --out out --video --dedup
"""

from __future__ import annotations

import argparse
import sys
import time

from .config import QualityConfig
from .curate import write as write_curated
from .dataset import LeRobotDataset
from .engine import Engine
from .report import write as write_report


def build_config(args) -> QualityConfig:
    cfg = QualityConfig()
    cfg.video.enabled = args.video
    cfg.dedup.enabled = args.dedup
    cfg.pass_threshold = args.threshold
    if args.video_fps:
        cfg.video.sample_fps = args.video_fps
    if args.near_dup_dist:
        cfg.dedup.near_dup_dist = args.near_dup_dist
    if args.judge and args.judge != "none":
        cfg.instruction.enabled = True
        cfg.instruction.backend = args.judge
    return cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Data Q(uality) Engine for LeRobot datasets")
    p.add_argument("--dataset", required=True, help="path to the LeRobot dataset root")
    p.add_argument("--out", default="out", help="output directory")
    p.add_argument("--video", action="store_true", help="enable video-quality metrics (slow)")
    p.add_argument("--dedup", action="store_true", help="enable cross-episode dedup")
    p.add_argument("--threshold", type=float, default=0.60, help="keep episodes scoring >= this")
    p.add_argument("--video-fps", type=float, default=0.0, help="frames/sec sampled per clip")
    p.add_argument("--near-dup-dist", type=float, default=0.0, help="dedup distance threshold")
    p.add_argument("--judge",
                   choices=["none", "heuristic", "claude", "hf", "vegaml"],
                   default="none",
                   help="VLM instruction-following judge backend (heuristic=offline; "
                        "vegaml=calibrated + abstain->review)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    cfg = build_config(args)
    ds = LeRobotDataset(args.dataset)
    if not args.quiet:
        print("Dataset: %s" % ds.root)
        print("  robot=%s fps=%g video_keys=%s" % (
            ds.info.get("robot_type"), ds.fps, ds.video_keys))
    if not ds.has_data():
        print("ERROR: no data/*.parquet found yet — is the download finished?",
              file=sys.stderr)
        return 2
    if cfg.video.enabled and ds.episodes_meta() is None:
        print("WARN: meta/episodes/* not found; video metrics will be skipped "
              "(episode->mp4 mapping unavailable).", file=sys.stderr)

    t0 = time.time()
    records = Engine(ds, cfg).run(verbose=not args.quiet)
    write_report(records, args.out, ds.root, cfg.to_dict())
    summary = write_curated(records, args.out)
    if not args.quiet:
        print("Scored %d episodes in %.1fs — kept %d / dropped %d" % (
            len(records), time.time() - t0,
            summary["n_kept"], summary["n_dropped"]))
        print("Report:  %s/report.md" % args.out)
        print("Curated: %s/curated/keep_episodes.json" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
