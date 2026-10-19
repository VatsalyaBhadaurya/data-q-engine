"""Synthetic data augmentation pipeline: generate -> filter -> write.

For each (optionally pre-curated) source episode, generate trajectory variants,
score every variant with the *same* quality scorer used on real data, and keep
only the ones that pass — RLDX-1's generate-then-filter loop. Surviving
variants are written as a LeRobot-style parquet plus a provenance manifest, and
task instructions are paraphrased alongside.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import QualityConfig
from ..dataset import LeRobotDataset
from ..metrics import trajectory as traj
from ..scoring import Scorer
from . import task_augment, trajectory_augment


DEFAULT_TRANSFORMS = ["jitter", "time_warp", "crop_idle", "gripper_jitter"]


def _score_cfg(threshold: float) -> QualityConfig:
    """A scoring config for synthetic trajectories: video/instruction off."""
    cfg = QualityConfig()
    cfg.video.enabled = False
    cfg.instruction.enabled = False
    cfg.pass_threshold = threshold
    return cfg


def run(dataset_root: str, out_dir: str, *, transforms: Optional[List[str]] = None,
        per_episode: int = 2, threshold: float = 0.65, seed: int = 0,
        keep_list: Optional[List[int]] = None, task_backend: str = "offline",
        task_model: str = "claude-opus-4-8", verbose: bool = True) -> Dict:
    transforms = transforms or DEFAULT_TRANSFORMS
    ds = LeRobotDataset(dataset_root)
    cfg = _score_cfg(threshold)
    jmin, jmax = ds.joint_limits()
    scorer = Scorer(cfg)
    rng = np.random.default_rng(seed)

    episodes = list(ds.iter_episodes())
    if keep_list is not None:
        keep_set = set(keep_list)
        episodes = [e for e in episodes if e.index in keep_set]
    next_ep = (max((e.index for e in episodes), default=-1) + 1) if episodes else 0
    next_ep = max(next_ep, (ds.data()["episode_index"].max() + 1))

    rows: List[dict] = []
    manifest: List[dict] = []
    gidx = int(ds.data()["index"].max() + 1) if ds.has_data() else 0
    kept = 0
    attempted = 0

    for ep in episodes:
        picks = list(rng.choice(transforms,
                                size=min(per_episode, len(transforms)),
                                replace=False))
        for tag, s, a in trajectory_augment.augment(ep.state, ep.action, picks, rng):
            attempted += 1
            m = traj.compute(s, a, np.arange(len(s)) / ds.fps, ds.fps,
                             cfg.trajectory.jump_sigma, cfg.trajectory.idle_speed_eps,
                             joint_min=jmin, joint_max=jmax)
            scored = scorer.score_episode(m)
            critical = any(f.split("(")[0] in cfg.critical_flags for f in scored["flags"])
            if scored["score"] < threshold or critical:
                continue
            kept += 1
            ei = next_ep
            next_ep += 1
            for fi in range(len(s)):
                rows.append({
                    "timestamp": np.float32(fi / ds.fps),
                    "frame_index": np.int64(fi),
                    "episode_index": np.int64(ei),
                    "index": np.int64(gidx),
                    "task_index": np.int64(ep.task_index),
                    "observation.state": s[fi].astype(np.float32),
                    "action": a[fi].astype(np.float32),
                })
                gidx += 1
            manifest.append({"synthetic_episode": int(ei),
                             "source_episode": int(ep.index),
                             "transform": tag, "task_index": int(ep.task_index),
                             "task": ep.task, "score": round(scored["score"], 4),
                             "n_frames": int(len(s))})
        if verbose and (ep.index + 1) % 200 == 0:
            print("  processed source episode %d (kept %d so far)" % (ep.index, kept))

    # ---- write outputs -----------------------------------------------------
    syn_dir = os.path.join(out_dir, "synthetic")
    data_dir = os.path.join(syn_dir, "data", "chunk-000")
    os.makedirs(data_dir, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_parquet(os.path.join(data_dir, "file-000.parquet"))

    with open(os.path.join(syn_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"source": ds.root, "transforms": transforms,
                   "attempted": attempted, "kept": kept,
                   "episodes": manifest}, f, indent=2)

    # ---- task paraphrases --------------------------------------------------
    aug = task_augment.build_augmenter(task_backend, task_model)
    task_variants = {}
    for ti, text in sorted(ds._tasks.items()):
        try:
            task_variants[text] = aug.paraphrase(text, 4)
        except Exception as e:                     # pragma: no cover
            task_variants[text] = {"error": str(e)}
    with open(os.path.join(syn_dir, "task_paraphrases.json"), "w",
              encoding="utf-8") as f:
        json.dump(task_variants, f, indent=2)

    summary = {"attempted": attempted, "kept": kept,
               "accept_rate": round(kept / attempted, 3) if attempted else 0.0,
               "synthetic_episodes": kept, "out_dir": syn_dir}
    return summary
