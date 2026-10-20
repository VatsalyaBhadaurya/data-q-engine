"""Emit a curated split: the list of episodes to keep after quality filtering.

Kept episodes are written as JSON and plain text. The list is directly usable
with LeRobot's ``LeRobotDataset(..., episodes=[...])`` selector, so no parquet
re-packing or video re-encoding is required to train on the clean subset.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List


def write(records: List[Dict], out_dir: str) -> Dict:
    curated_dir = os.path.join(out_dir, "curated")
    os.makedirs(curated_dir, exist_ok=True)

    kept = sorted(r["episode_index"] for r in records if r["keep"])
    review = sorted(r["episode_index"] for r in records if r.get("review"))
    dropped = [
        {
            "episode_index": r["episode_index"],
            "score": round(r["score"], 4),
            "reasons": r["flags"] or ["below_threshold"],
        }
        for r in records if not r["keep"] and not r.get("review")
    ]

    summary = {
        "n_total": len(records),
        "n_kept": len(kept),
        "n_review": len(review),
        "n_dropped": len(dropped),
        "kept_episodes": kept,
        "review_episodes": review,
        "dropped": dropped,
    }
    with open(os.path.join(curated_dir, "keep_episodes.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(curated_dir, "keep_episodes.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(str(e) for e in kept) + ("\n" if kept else ""))
    if review:
        with open(os.path.join(curated_dir, "review_episodes.txt"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(str(e) for e in review) + "\n")

    # a ready-to-paste snippet
    with open(os.path.join(curated_dir, "usage.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Using the curated split\n\n"
            "```python\n"
            "import json\n"
            "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset\n\n"
            "keep = json.load(open('out/curated/keep_episodes.json'))['kept_episodes']\n"
            "ds = LeRobotDataset('<repo-or-root>', episodes=keep)\n"
            "```\n\n"
            "%d of %d episodes kept (%.1f%%).\n" % (
                len(kept), len(records),
                100.0 * len(kept) / max(len(records), 1)),
        )
    return summary
