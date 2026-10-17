"""LeRobot v3.0 dataset reader.

Handles the v3.0 "packed" layout where many episodes share one parquet data
file and one mp4 per camera. Trajectory access only needs the data parquet
(grouped by ``episode_index``); video access additionally needs the
``meta/episodes`` metadata that maps each episode to its mp4 file and the
timestamp window inside that file.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Episode:
    index: int
    task_index: int
    task: Optional[str]
    frames: pd.DataFrame            # rows for this episode, sorted by frame_index
    state: np.ndarray               # (T, D) measured joint positions
    action: np.ndarray              # (T, D) commanded joint targets
    timestamps: np.ndarray          # (T,) seconds
    joint_names: List[str]
    # Per-camera video reference: cam -> {"path", "from_ts", "to_ts"}
    video_refs: Dict[str, dict]


class LeRobotDataset:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.meta_dir = os.path.join(self.root, "meta")
        self.info = self._load_json(os.path.join(self.meta_dir, "info.json"))
        self.stats = self._load_json_optional(os.path.join(self.meta_dir, "stats.json"))
        self.features = self.info.get("features", {})
        self.fps = float(self.info.get("fps", 30))
        self.video_keys = [
            k for k, v in self.features.items() if v.get("dtype") == "video"
        ]
        self.joint_names = self.features.get("action", {}).get("names") or []
        self._data = None                  # concatenated data parquet (lazy)
        self._episodes_meta = None         # per-episode metadata (lazy)
        self._tasks = self._load_tasks()

    # ---- loading helpers ---------------------------------------------------
    @staticmethod
    def _load_json(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def _load_json_optional(cls, path: str) -> Optional[dict]:
        return cls._load_json(path) if os.path.exists(path) else None

    def _load_tasks(self) -> Dict[int, str]:
        """Return {task_index: task_string} if a tasks table exists.

        LeRobot v3.0 stores ``meta/tasks.parquet`` indexed by the task string
        with a ``task_index`` column; older variants use plain columns. Handle
        both by flattening the index into a column first.
        """
        for pat in ("tasks.parquet", "tasks/*.parquet", "tasks/**/*.parquet"):
            for path in glob.glob(os.path.join(self.meta_dir, pat), recursive=True):
                try:
                    df = pd.read_parquet(path).reset_index()
                except Exception:
                    continue
                if "task_index" not in df.columns:
                    continue
                txt_cols = [c for c in df.columns if c != "task_index"]
                if not txt_cols:
                    continue
                # prefer an explicit 'task' column, else the remaining column
                txt_col = "task" if "task" in txt_cols else txt_cols[0]
                return {int(r["task_index"]): str(r[txt_col])
                        for _, r in df.iterrows()}
        return {}

    # ---- data parquet ------------------------------------------------------
    def _data_files(self) -> List[str]:
        return sorted(glob.glob(os.path.join(self.root, "data", "**", "*.parquet"),
                                recursive=True))

    def data(self) -> pd.DataFrame:
        if self._data is None:
            files = self._data_files()
            if not files:
                raise FileNotFoundError(
                    "No data parquet files found under %s/data. The dataset may "
                    "still be downloading." % self.root
                )
            self._data = pd.concat((pd.read_parquet(f) for f in files),
                                   ignore_index=True)
        return self._data

    def has_data(self) -> bool:
        return len(self._data_files()) > 0

    # ---- episode metadata (for video mapping) ------------------------------
    def _episodes_meta_files(self) -> List[str]:
        return sorted(glob.glob(os.path.join(self.meta_dir, "episodes", "**",
                                             "*.parquet"), recursive=True))

    def episodes_meta(self) -> Optional[pd.DataFrame]:
        if self._episodes_meta is None:
            files = self._episodes_meta_files()
            if not files:
                return None
            self._episodes_meta = pd.concat(
                (pd.read_parquet(f) for f in files), ignore_index=True)
        return self._episodes_meta

    def _video_path(self, chunk_index: int, file_index: int, video_key: str) -> str:
        rel = self.info["video_path"].format(
            video_key=video_key, chunk_index=int(chunk_index),
            file_index=int(file_index))
        return os.path.join(self.root, rel)

    def _video_ref_for(self, ep_row: pd.Series, video_key: str) -> Optional[dict]:
        ci = ep_row.get("videos/%s/chunk_index" % video_key)
        fi = ep_row.get("videos/%s/file_index" % video_key)
        if ci is None or fi is None or pd.isna(ci) or pd.isna(fi):
            return None
        path = self._video_path(ci, fi, video_key)
        if not os.path.exists(path):
            return None
        return {
            "path": path,
            "from_ts": float(ep_row.get("videos/%s/from_timestamp" % video_key, 0.0)),
            "to_ts": float(ep_row.get("videos/%s/to_timestamp" % video_key, 0.0)),
        }

    # ---- episode iteration -------------------------------------------------
    def _stack(self, series: pd.Series) -> np.ndarray:
        """Turn a parquet list/array column into a (T, D) float array."""
        arr = np.stack([np.asarray(v, dtype=np.float32) for v in series.to_numpy()])
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr

    def iter_episodes(self):
        data = self.data()
        ep_meta = self.episodes_meta()
        meta_by_ep = {}
        if ep_meta is not None and "episode_index" in ep_meta.columns:
            meta_by_ep = {int(r["episode_index"]): r for _, r in ep_meta.iterrows()}

        for ep_idx, g in data.groupby("episode_index", sort=True):
            g = g.sort_values("frame_index")
            task_index = int(g["task_index"].iloc[0]) if "task_index" in g else -1
            video_refs = {}
            task_str = self._tasks.get(task_index)
            row = meta_by_ep.get(int(ep_idx))
            if row is not None:
                if task_str is None and "tasks" in row and row["tasks"] is not None:
                    tlist = list(row["tasks"])
                    task_str = str(tlist[0]) if len(tlist) else None
                for vk in self.video_keys:
                    ref = self._video_ref_for(row, vk)
                    if ref is not None:
                        video_refs[vk] = ref
            yield Episode(
                index=int(ep_idx),
                task_index=task_index,
                task=task_str,
                frames=g,
                state=self._stack(g["observation.state"]),
                action=self._stack(g["action"]),
                timestamps=g["timestamp"].to_numpy(dtype=np.float64)
                if "timestamp" in g else np.arange(len(g)) / self.fps,
                joint_names=self.joint_names,
                video_refs=video_refs,
            )

    # ---- global joint limits (from stats.json) -----------------------------
    def joint_limits(self):
        """Return (min, max) arrays over the union of action+state ranges."""
        if not self.stats:
            return None, None
        lows, highs = [], []
        for key in ("action", "observation.state"):
            s = self.stats.get(key)
            if s and "min" in s and "max" in s:
                lows.append(np.asarray(s["min"], dtype=np.float64))
                highs.append(np.asarray(s["max"], dtype=np.float64))
        if not lows:
            return None, None
        return np.min(np.stack(lows), axis=0), np.max(np.stack(highs), axis=0)
