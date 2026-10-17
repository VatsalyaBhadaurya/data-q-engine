"""Trajectory & motion-consistency metrics for a single episode.

All metrics are computed from the measured state and commanded action joint
arrays (T, D). The gripper (last dim) is handled separately from the arm
joints where it makes physical sense. Returns raw numbers; normalisation to
sub-scores happens in ``scoring.py``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _arm_grip(x: np.ndarray):
    """Split a (T, D) array into arm joints and the gripper column."""
    if x.shape[1] >= 2:
        return x[:, :-1], x[:, -1]
    return x, None


def compute(state: np.ndarray, action: np.ndarray, timestamps: np.ndarray,
            fps: float, jump_sigma: float, idle_speed_eps: float,
            joint_min: Optional[np.ndarray] = None,
            joint_max: Optional[np.ndarray] = None) -> Dict[str, float]:
    T, D = state.shape
    out: Dict[str, float] = {"n_frames": float(T)}
    duration = float(timestamps[-1] - timestamps[0]) if T > 1 else 0.0
    out["duration_s"] = duration

    # ---- integrity ---------------------------------------------------------
    out["nan_ratio"] = float(
        (np.isnan(state).any(axis=1) | np.isnan(action).any(axis=1)).mean()
    ) if T else 1.0
    state = np.nan_to_num(state)
    action = np.nan_to_num(action)

    arm, grip = _arm_grip(state)

    # ---- motion / idle -----------------------------------------------------
    if T > 1:
        step = np.linalg.norm(np.diff(arm, axis=0), axis=1)   # per-step arm speed
        path_len = float(step.sum())
        idle_ratio = float((step < idle_speed_eps).mean())
    else:
        step = np.zeros(0)
        path_len = 0.0
        idle_ratio = 1.0
    out["path_length"] = path_len
    out["idle_ratio"] = idle_ratio

    # ---- smoothness (jerk) -------------------------------------------------
    # Normalised mean squared acceleration of the arm joints; lower == smoother.
    if T > 2:
        acc = np.diff(arm, n=2, axis=0)
        out["mean_jerk"] = float(np.sqrt((acc ** 2).sum(axis=1)).mean())
        out["max_jerk"] = float(np.sqrt((acc ** 2).sum(axis=1)).max())
    else:
        out["mean_jerk"] = 0.0
        out["max_jerk"] = 0.0

    # ---- discontinuities (teleport / dropped frames) ----------------------
    if step.size:
        sigma = step.std() + 1e-9
        out["jump_ratio"] = float((step > (step.mean() + jump_sigma * sigma)).mean())
    else:
        out["jump_ratio"] = 0.0

    # ---- commanded-vs-achieved tracking (motion consistency) --------------
    # action[t] is the target that state[t+1] should reach.
    if T > 1:
        err = np.abs(state[1:] - action[:-1])
        out["tracking_err_mean"] = float(err.mean())
        out["tracking_err_max"] = float(err.max())
    else:
        out["tracking_err_mean"] = 0.0
        out["tracking_err_max"] = 0.0

    # ---- gripper activity --------------------------------------------------
    if grip is not None and T > 1:
        out["gripper_travel"] = float(np.abs(np.diff(grip)).sum())
        rng = float(grip.max() - grip.min()) + 1e-9
        normed = (grip - grip.min()) / rng
        out["gripper_transitions"] = float(
            (np.abs(np.diff((normed > 0.5).astype(np.int8)))).sum()
        )
    else:
        out["gripper_travel"] = 0.0
        out["gripper_transitions"] = 0.0

    # ---- joint-limit plausibility -----------------------------------------
    if joint_min is not None and joint_max is not None and D == joint_min.shape[0]:
        span = (joint_max - joint_min) + 1e-9
        below = (state < joint_min).any(axis=1)
        above = (state > joint_max).any(axis=1)
        out["limit_violation_ratio"] = float((below | above).mean())
        # how close, worst joint, to its limit on average (0 safe .. 1 at edge)
        near = np.minimum(state - joint_min, joint_max - state) / span
        out["limit_proximity"] = float(np.clip(1.0 - near.min(axis=1), 0, 1).mean())
    else:
        out["limit_violation_ratio"] = 0.0
        out["limit_proximity"] = 0.0

    return out


def fingerprint(state: np.ndarray, resample_len: int) -> np.ndarray:
    """Fixed-length, scale-normalised trajectory fingerprint for dedup."""
    T, D = state.shape
    if T == 0:
        return np.zeros(resample_len * D, dtype=np.float32)
    xs = np.linspace(0, 1, T)
    grid = np.linspace(0, 1, resample_len)
    res = np.stack([np.interp(grid, xs, state[:, d]) for d in range(D)], axis=1)
    # per-joint z-norm so shape (not absolute pose) drives similarity
    res = (res - res.mean(axis=0)) / (res.std(axis=0) + 1e-6)
    return res.astype(np.float32).ravel()
