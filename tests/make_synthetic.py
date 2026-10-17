"""Build a tiny synthetic LeRobot v3.0 dataset to exercise the engine end to end.

Creates a handful of episodes with known defects (idle, glitchy, static,
no-gripper, duplicate) plus clean ones, so the scores and flags are
predictable. No video is written; trajectory + dedup paths are covered.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pandas as pd

D = 7
FPS = 30


def ep_clean(n=120, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    base = np.stack([
        0.8 * np.sin(2 * np.pi * t), 1.0 + 0.3 * t, -0.6 * t,
        0.1 * np.sin(4 * np.pi * t), 0.6 * np.ones(n), 0.0 * t,
        np.where(t > 0.5, 0.08, 0.0),             # gripper closes mid-way
    ], axis=1).astype(np.float32)
    state = base + rng.normal(0, 0.002, base.shape).astype(np.float32)
    action = np.vstack([state[1:], state[-1:]])    # action[t] ~= state[t+1]
    return state, action


def ep_idle(n=120):
    state, action = ep_clean(n)
    state[: int(0.7 * n)] = state[0]               # 70% frozen at start
    action[: int(0.7 * n)] = action[0]
    return state, action


def ep_glitch(n=120):
    state, action = ep_clean(n, seed=1)
    state[60] += 3.0                               # teleport spike
    return state, action


def ep_static(n=120):
    state = np.tile(ep_clean(n)[0][0], (n, 1))     # no motion at all
    return state, state.copy()


def ep_nogrip(n=120):
    state, action = ep_clean(n, seed=2)
    state[:, -1] = 0.0                             # gripper never moves
    action[:, -1] = 0.0
    return state, action


def build(root: str):
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(os.path.join(root, "data", "chunk-000"))
    os.makedirs(os.path.join(root, "meta"))

    makers = [ep_clean, ep_clean, ep_idle, ep_glitch, ep_static, ep_nogrip]
    rows = []
    gidx = 0
    for ei, mk in enumerate(makers):
        state, action = mk()
        n = len(state)
        for fi in range(n):
            rows.append({
                "timestamp": np.float32(fi / FPS),
                "frame_index": np.int64(fi),
                "episode_index": np.int64(ei),
                "index": np.int64(gidx),
                "task_index": np.int64(0),
                "observation.state": state[fi],
                "action": action[fi],
            })
            gidx += 1
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(root, "data", "chunk-000", "file-000.parquet"))

    names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
    info = {
        "codebase_version": "v3.0", "fps": FPS,
        "features": {
            "observation.state": {"dtype": "float32", "names": names, "shape": [D]},
            "action": {"dtype": "float32", "names": names, "shape": [D]},
        },
        "total_episodes": len(makers), "robot_type": "piper",
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    with open(os.path.join(root, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    allp = np.vstack([np.vstack([mk()[0], mk()[1]]) for mk in makers])
    stats = {
        "action": {"min": allp.min(0).tolist(), "max": allp.max(0).tolist()},
        "observation.state": {"min": allp.min(0).tolist(), "max": allp.max(0).tolist()},
    }
    with open(os.path.join(root, "meta", "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return root


if __name__ == "__main__":
    import sys
    print(build(sys.argv[1] if len(sys.argv) > 1 else "tests/_synth_ds"))
