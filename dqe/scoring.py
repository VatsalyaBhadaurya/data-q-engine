"""Map raw per-episode metrics to [0,1] sub-scores and an overall score.

Each sub-score is 1 == best, 0 == worst. The overall score is the weighted
mean over the sub-scores that are actually present for an episode (weights
are renormalised so disabling video/dedup doesn't deflate everyone's score).
Human-readable ``flags`` explain every deduction.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .config import QualityConfig


def _ramp_down(x: float, good: float, bad: float) -> float:
    """1 when x<=good, 0 when x>=bad, linear between (bad>good)."""
    if x <= good:
        return 1.0
    if x >= bad:
        return 0.0
    return 1.0 - (x - good) / (bad - good)


def _ramp_up(x: float, bad: float, good: float) -> float:
    """0 when x<=bad, 1 when x>=good, linear between (good>bad)."""
    if x <= bad:
        return 0.0
    if x >= good:
        return 1.0
    return (x - bad) / (good - bad)


class Scorer:
    def __init__(self, cfg: QualityConfig, max_frames_ref: Optional[int] = None):
        self.cfg = cfg
        self.max_frames_ref = max_frames_ref

    def score_episode(self, m: Dict[str, float]) -> Dict:
        t = self.cfg.trajectory
        w = self.cfg.weights
        subs: Dict[str, float] = {}
        weights: Dict[str, float] = {}
        flags: List[str] = []

        # tracking (motion consistency)
        s = _ramp_down(m.get("tracking_err_mean", 0.0),
                       t.tracking_err_good, t.tracking_err_bad)
        subs["tracking"] = s
        weights["tracking"] = w.tracking
        if s < 0.5:
            flags.append("high_tracking_error(%.3f)" % m.get("tracking_err_mean", 0))

        # smoothness via jump ratio (glitches) + jerk presence
        jump = m.get("jump_ratio", 0.0)
        s = _ramp_down(jump, 0.0, t.max_jump_ratio)
        subs["smoothness"] = s
        weights["smoothness"] = w.smoothness
        if jump > t.max_jump_ratio:
            flags.append("discontinuities(jump_ratio=%.3f)" % jump)

        # idle
        idle = m.get("idle_ratio", 0.0)
        s = _ramp_down(idle, 0.0, t.max_idle_ratio)
        subs["idle"] = s
        weights["idle"] = w.idle
        if idle > t.max_idle_ratio:
            flags.append("mostly_idle(%.2f)" % idle)

        # useful motion / path length
        s = _ramp_up(m.get("path_length", 0.0), 0.0, t.min_path_length)
        subs["motion"] = s
        weights["motion"] = w.motion
        if m.get("path_length", 0.0) < t.min_path_length:
            flags.append("near_static(path=%.3f)" % m.get("path_length", 0))

        # gripper
        if t.expect_gripper_action:
            s = _ramp_up(m.get("gripper_travel", 0.0), 0.0, t.gripper_min_travel)
            subs["gripper"] = s
            weights["gripper"] = w.gripper
            if m.get("gripper_travel", 0.0) < t.gripper_min_travel:
                flags.append("no_gripper_action")

        # joint limits
        viol = m.get("limit_violation_ratio", 0.0)
        s = 1.0 - min(viol * 20.0, 1.0)      # any sustained violation is severe
        subs["limits"] = s
        weights["limits"] = w.limits
        if viol > 0.0:
            flags.append("limit_violation(%.3f)" % viol)

        # video (optional)
        if self.cfg.video.enabled and m.get("video_available", 0.0) >= 1.0:
            v = self.cfg.video
            blur_s = _ramp_up(m.get("blur_var", 0.0), v.blur_var_min * 0.5,
                              v.blur_var_min)
            frozen_s = _ramp_down(m.get("frozen_ratio", 0.0), 0.0,
                                  v.max_frozen_ratio)
            b = m.get("brightness", 0.5)
            expo_s = 1.0 if v.brightness_min <= b <= v.brightness_max else 0.0
            s = min(blur_s, frozen_s, expo_s)
            subs["video"] = s
            weights["video"] = w.video
            if blur_s < 0.5:
                flags.append("blurry(var=%.1f)" % m.get("blur_var", 0))
            if frozen_s < 0.5:
                flags.append("frozen_frames(%.2f)" % m.get("frozen_ratio", 0))
            if expo_s < 1.0:
                flags.append("bad_exposure(%.2f)" % b)
        elif self.cfg.video.enabled and m.get("video_available", 1.0) < 1.0:
            flags.append("video_missing")

        # integrity hard flag
        if m.get("nan_ratio", 0.0) > 0.0:
            flags.append("contains_nan(%.3f)" % m["nan_ratio"])

        # length sanity
        nframes = m.get("n_frames", 0.0)
        if nframes < t.min_frames:
            flags.append("too_short(%d)" % int(nframes))
        if self.max_frames_ref and nframes > self.max_frames_ref:
            flags.append("length_outlier(%d)" % int(nframes))

        total_w = sum(weights.values()) or 1.0
        overall = sum(subs[k] * weights[k] for k in subs) / total_w
        return {"score": overall, "subscores": subs, "flags": flags}
