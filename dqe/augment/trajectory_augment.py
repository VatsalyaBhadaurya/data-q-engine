"""Trajectory-level synthetic episode generation.

The locally-computable analog of RLDX-1's scene/video augmentation: instead of
synthesizing new pixels, we synthesize new *trajectories* from real ones via
label-preserving kinematic transforms, then let the quality engine filter the
results (RLDX's generate-then-filter loop). Each transform returns a new
(state, action) pair plus a provenance tag; implausible variants are expected
to be rejected downstream by the same scorer used on real data.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _resample(x: np.ndarray, n: int) -> np.ndarray:
    T = x.shape[0]
    if T == n or T < 2:
        return x.copy()
    src = np.linspace(0, 1, T)
    dst = np.linspace(0, 1, n)
    return np.stack([np.interp(dst, src, x[:, d]) for d in range(x.shape[1])], axis=1)


def jitter(state, action, rng, scale=0.01):
    """Add small per-joint Gaussian noise scaled by each joint's own range."""
    span = (state.max(axis=0) - state.min(axis=0)) + 1e-6
    noise = rng.normal(0.0, scale, state.shape).astype(np.float32) * span
    return (state + noise).astype(np.float32), (action + noise).astype(np.float32)


def time_warp(state, action, rng, factor_range=(0.85, 1.15)):
    """Stretch/compress time, keeping the same number of frames (speed change)."""
    f = float(rng.uniform(*factor_range))
    n = state.shape[0]
    warped = int(max(4, round(n * f)))
    s = _resample(_resample(state, warped), n)
    a = _resample(_resample(action, warped), n)
    return s.astype(np.float32), a.astype(np.float32)


def crop_idle(state, action, idle_eps=1e-3):
    """Trim leading/trailing idle frames (tightens loose teleop recordings)."""
    if state.shape[0] < 4:
        return state.copy(), action.copy()
    arm = state[:, :-1] if state.shape[1] >= 2 else state
    speed = np.r_[0.0, np.linalg.norm(np.diff(arm, axis=0), axis=1)]
    moving = np.where(speed >= idle_eps)[0]
    if moving.size < 2:
        return state.copy(), action.copy()
    lo, hi = moving[0], moving[-1] + 1
    return state[lo:hi].copy(), action[lo:hi].copy()


def gripper_jitter(state, action, rng, shift_frac=0.05):
    """Shift the gripper channel slightly in time (actuation-timing variation)."""
    if state.shape[1] < 2:
        return state.copy(), action.copy()
    n = state.shape[0]
    shift = int(rng.integers(-max(1, int(n * shift_frac)), max(2, int(n * shift_frac))))
    s, a = state.copy(), action.copy()
    s[:, -1] = np.roll(s[:, -1], shift)
    a[:, -1] = np.roll(a[:, -1], shift)
    return s, a


TRANSFORMS = {
    "jitter": lambda s, a, rng: jitter(s, a, rng),
    "time_warp": lambda s, a, rng: time_warp(s, a, rng),
    "crop_idle": lambda s, a, rng: crop_idle(s, a),
    "gripper_jitter": lambda s, a, rng: gripper_jitter(s, a, rng),
}


def augment(state: np.ndarray, action: np.ndarray, which: List[str],
            rng: np.random.Generator) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Apply the named transforms; return [(tag, state, action), ...]."""
    out = []
    for name in which:
        fn = TRANSFORMS.get(name)
        if fn is None:
            continue
        s, a = fn(state, action, rng)
        out.append((name, s, a))
    return out
