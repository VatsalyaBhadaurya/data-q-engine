"""Pipeline orchestration: dataset -> metrics -> scores -> dedup -> results."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .config import QualityConfig
from .dataset import LeRobotDataset
from .metrics import dedup as dedup_mod
from .metrics import trajectory as traj
from .metrics import video as video_mod
from .scoring import Scorer


class Engine:
    def __init__(self, dataset: LeRobotDataset, cfg: QualityConfig):
        self.ds = dataset
        self.cfg = cfg

    def run(self, verbose: bool = True) -> List[Dict]:
        jmin, jmax = self.ds.joint_limits()
        records: List[Dict] = []
        fingerprints: Dict[int, np.ndarray] = {}
        lengths: Dict[int, int] = {}
        task_of: Dict[int, int] = {}

        episodes = list(self.ds.iter_episodes())
        # length-outlier reference (p99) from the data itself
        all_lens = np.array([e.state.shape[0] for e in episodes]) if episodes else np.array([0])
        max_ref = int(np.percentile(all_lens, 99) * 1.5) if episodes else 0
        if self.cfg.trajectory.max_frames:
            max_ref = self.cfg.trajectory.max_frames
        scorer = Scorer(self.cfg, max_frames_ref=max_ref)

        for i, ep in enumerate(episodes):
            m = traj.compute(
                ep.state, ep.action, ep.timestamps, self.ds.fps,
                self.cfg.trajectory.jump_sigma,
                self.cfg.trajectory.idle_speed_eps,
                joint_min=jmin, joint_max=jmax,
            )
            if self.cfg.video.enabled and ep.video_refs:
                m.update(video_mod.compute(
                    ep.video_refs, self.cfg.video.sample_fps,
                    self.cfg.video.max_frames_per_clip,
                    self.cfg.video.frozen_diff_eps))
            elif self.cfg.video.enabled:
                m["video_available"] = 0.0

            scored = scorer.score_episode(m)
            rec = {
                "episode_index": ep.index,
                "task_index": ep.task_index,
                "task": ep.task,
                "score": scored["score"],
                "flags": scored["flags"],
                "subscores": scored["subscores"],
                "metrics": m,
            }
            records.append(rec)

            if self.cfg.dedup.enabled:
                fingerprints[ep.index] = traj.fingerprint(
                    ep.state, self.cfg.dedup.resample_len)
                lengths[ep.index] = ep.state.shape[0]
                task_of[ep.index] = ep.task_index

            if verbose and (i + 1) % 100 == 0:
                print("  scored %d/%d episodes" % (i + 1, len(episodes)))

        if self.cfg.dedup.enabled and fingerprints:
            rep, pairs = dedup_mod.find_duplicates(
                fingerprints, lengths, task_of,
                self.cfg.dedup.near_dup_dist,
                self.cfg.dedup.within_task_only)
            for rec in records:
                e = rec["episode_index"]
                keep = rep.get(e, e)
                rec["duplicate_of"] = None if keep == e else keep
                if keep != e:
                    rec["flags"] = rec["flags"] + ["duplicate_of(%d)" % keep]

        # pass/fail decision
        critical = tuple(self.cfg.critical_flags)
        for rec in records:
            dup = rec.get("duplicate_of")
            has_critical = any(f.split("(")[0] in critical for f in rec["flags"])
            rec["keep"] = bool(
                rec["score"] >= self.cfg.pass_threshold
                and dup is None
                and not has_critical
            )
        return records
