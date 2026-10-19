"""Configuration: thresholds and scoring weights for the quality engine.

Every tunable lives here so the scoring policy is auditable in one place.
Thresholds are expressed in the dataset's native units (radians for the 6
arm joints, metres for the gripper) unless noted. Sub-scores are mapped to
[0, 1] where 1 == best; the overall quality score is a weighted mean of the
enabled sub-scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict


@dataclass
class TrajectoryConfig:
    # Fraction of an episode allowed to be "idle" (no motion) before penalty.
    idle_speed_eps: float = 1e-3          # per-step L2 joint delta below this == idle
    max_idle_ratio: float = 0.40          # >40% idle frames -> fully penalised
    # Commanded-vs-achieved tracking error (|state[t+1]-action[t]|), mean over joints.
    tracking_err_good: float = 0.02       # <= good  -> sub-score 1
    tracking_err_bad: float = 0.20        # >= bad   -> sub-score 0
    # Jerk / glitch detection: per-step delta larger than this many robot-wide
    # stds is treated as a discontinuity (teleport / dropped frame).
    jump_sigma: float = 6.0
    max_jump_ratio: float = 0.01          # >1% of frames being jumps -> 0
    # Minimum useful motion: total path length (sum of per-step L2 deltas).
    min_path_length: float = 0.10         # near-static episodes are suspect
    # Episode length sanity (frames). Populated from the data if left as 0.
    min_frames: int = 15
    max_frames: int = 0                   # 0 == derive p99 from dataset
    # Expect at least one gripper actuation for pick/place style tasks.
    expect_gripper_action: bool = True
    gripper_min_travel: float = 0.01      # total gripper motion (metres)


@dataclass
class VideoConfig:
    enabled: bool = False                 # off by default (decoding is slow)
    sample_fps: float = 2.0               # frames/sec sampled per episode clip
    max_frames_per_clip: int = 60         # hard cap on decoded frames per clip
    blur_var_min: float = 40.0            # variance-of-Laplacian; below == blurry
    brightness_min: float = 0.12          # mean luma in [0,1]; outside == bad exposure
    brightness_max: float = 0.92
    frozen_diff_eps: float = 1.0          # mean abs frame diff (0-255) below == frozen
    max_frozen_ratio: float = 0.30        # >30% frozen sampled frames -> 0


@dataclass
class DedupConfig:
    enabled: bool = False
    resample_len: int = 48                # fixed-length trajectory fingerprint
    within_task_only: bool = True         # only compare episodes of the same task
    near_dup_dist: float = 0.08           # normalised RMS distance below == duplicate


@dataclass
class InstructionConfig:
    """VLM instruction-following judge (RLDX "video quality / instruction
    following" filter). Scores whether an episode's frames actually accomplish
    its language instruction."""
    enabled: bool = False
    backend: str = "heuristic"            # heuristic | claude | hf
    model: str = "claude-opus-4-8"        # for backend=claude
    hf_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"   # for backend=hf
    max_frames: int = 4                   # frames sampled from the clip to judge
    fail_threshold: float = 0.5           # below this -> instruction_fail (critical)


@dataclass
class ScoreWeights:
    tracking: float = 0.30
    smoothness: float = 0.20
    idle: float = 0.15
    motion: float = 0.10                  # useful range-of-motion / path length
    gripper: float = 0.10
    limits: float = 0.05
    video: float = 0.10
    instruction: float = 0.20             # only counted when the judge runs


@dataclass
class QualityConfig:
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    instruction: InstructionConfig = field(default_factory=InstructionConfig)
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    # Episodes scoring below this (after reweighting to enabled metrics) are
    # dropped from the curated split. Near-duplicates are also dropped.
    pass_threshold: float = 0.60
    # Flags that force a drop regardless of the weighted score — unrecoverable
    # defects where a merely-mediocre score would otherwise let garbage pass.
    critical_flags: tuple = (
        "contains_nan", "near_static", "too_short", "video_missing",
        "limit_violation", "instruction_fail",
    )

    def to_dict(self) -> Dict:
        return asdict(self)
