"""Video-quality metrics for an episode clip.

Decodes a bounded sample of frames from the episode's time window inside the
packed mp4 and measures blur (variance of Laplacian), exposure (mean luma),
and frozen/stuck frames (consecutive near-identical frames). opencv is
imported lazily so the rest of the engine runs without it.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _open_cv():
    import cv2  # noqa: WPS433 (lazy, optional dependency)
    return cv2


def analyse_clip(path: str, from_ts: float, to_ts: float, sample_fps: float,
                 max_frames: int, frozen_diff_eps: float) -> Optional[Dict[str, float]]:
    cv2 = _open_cv()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        span = max(to_ts - from_ts, 0.0)
        n_target = int(min(max_frames, max(1, span * sample_fps)))
        if span <= 0:
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            times = np.linspace(0, total / src_fps, n_target, endpoint=False)
        else:
            times = np.linspace(from_ts, to_ts, n_target, endpoint=False)

        blurs, brights = [], []
        frozen = 0
        prev_small = None
        read = 0
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            read += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurs.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            brights.append(float(gray.mean()) / 255.0)
            small = cv2.resize(gray, (32, 32)).astype(np.float32)
            if prev_small is not None:
                if np.abs(small - prev_small).mean() < frozen_diff_eps:
                    frozen += 1
            prev_small = small

        if read == 0:
            return {"decoded": 0.0, "blur_var": 0.0, "brightness": 0.0,
                    "frozen_ratio": 1.0}
        return {
            "decoded": float(read),
            "blur_var": float(np.median(blurs)),
            "brightness": float(np.median(brights)),
            "frozen_ratio": float(frozen / max(read - 1, 1)),
        }
    finally:
        cap.release()


def compute(video_refs: Dict[str, dict], sample_fps: float, max_frames: int,
            frozen_diff_eps: float) -> Dict[str, float]:
    """Aggregate per-camera clip metrics into worst-case episode metrics."""
    out: Dict[str, float] = {}
    per_cam = {}
    for cam, ref in video_refs.items():
        m = analyse_clip(ref["path"], ref["from_ts"], ref["to_ts"],
                         sample_fps, max_frames, frozen_diff_eps)
        if m is not None:
            per_cam[cam] = m
    if not per_cam:
        return {"video_available": 0.0}
    out["video_available"] = 1.0
    # worst (most degraded) camera drives the episode video metrics
    out["blur_var"] = min(m["blur_var"] for m in per_cam.values())
    out["frozen_ratio"] = max(m["frozen_ratio"] for m in per_cam.values())
    bright = [m["brightness"] for m in per_cam.values()]
    # distance of the most off-exposure camera from the mid-range
    out["brightness"] = min(bright, key=lambda b: 0.5 - abs(b - 0.5))
    return out
