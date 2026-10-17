"""Write the quality report: per-episode CSV, machine-readable JSON, and a
human-readable Markdown summary."""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd


def _flatten(rec: Dict) -> Dict:
    row = {
        "episode_index": rec["episode_index"],
        "task_index": rec["task_index"],
        "task": rec.get("task"),
        "score": round(rec["score"], 4),
        "keep": rec["keep"],
        "duplicate_of": rec.get("duplicate_of"),
        "flags": ";".join(rec["flags"]),
    }
    for k, v in rec["subscores"].items():
        row["sub_" + k] = round(v, 4)
    for k, v in rec["metrics"].items():
        row["m_" + k] = round(float(v), 5)
    return row


def write(records: List[Dict], out_dir: str, dataset_root: str,
          cfg_dict: Dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame([_flatten(r) for r in records]).sort_values("score")
    df.to_csv(os.path.join(out_dir, "episode_scores.csv"), index=False)

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg_dict, "episodes": records}, f, indent=2)

    scores = np.array([r["score"] for r in records]) if records else np.array([0.0])
    kept = [r for r in records if r["keep"]]
    dropped = [r for r in records if not r["keep"]]
    flag_counts = Counter(f.split("(")[0] for r in records for f in r["flags"])

    lines = []
    lines.append("# Data Q Engine — Quality Report\n")
    lines.append("Dataset: `%s`\n" % dataset_root)
    lines.append("## Summary\n")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append("| Episodes scored | %d |" % len(records))
    lines.append("| Kept (curated) | %d |" % len(kept))
    lines.append("| Dropped | %d |" % len(dropped))
    lines.append("| Mean score | %.3f |" % float(scores.mean()))
    lines.append("| Median score | %.3f |" % float(np.median(scores)))
    lines.append("| Min / Max | %.3f / %.3f |" % (scores.min(), scores.max()))
    lines.append("")
    lines.append("## Flag frequency\n")
    lines.append("| Flag | Episodes |")
    lines.append("| --- | --- |")
    for flag, c in flag_counts.most_common():
        lines.append("| %s | %d |" % (flag, c))
    lines.append("")
    lines.append("## Worst 25 episodes\n")
    lines.append("| Episode | Task | Score | Flags |")
    lines.append("| --- | --- | --- | --- |")
    for r in sorted(records, key=lambda r: r["score"])[:25]:
        lines.append("| %d | %s | %.3f | %s |" % (
            r["episode_index"], str(r.get("task") or r["task_index"]),
            r["score"], ", ".join(r["flags"]) or "—"))
    lines.append("")
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
