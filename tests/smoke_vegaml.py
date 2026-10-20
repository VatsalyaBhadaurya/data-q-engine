"""Smoke-test the VegaML instruction judge on real episodes.

Loads the actual VegaML model, pulls real frames through the engine's own
dataset reader + montage, runs the boolean instruction question, and prints
the raw output so we can confirm the adapter's key assumptions
(out["answers"]["success"]["boolean"] / ["abstain"] / ["prediction_set"]).
"""
import json
import sys

from dqe.dataset import LeRobotDataset
from dqe.metrics import video as video_mod
from dqe.augment.instruction_judge import VegaMLInstructionJudge, _montage

N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
root = sys.argv[1] if len(sys.argv) > 1 else "piper_dataset_1100"

ds = LeRobotDataset(root)
print("loading VegaML (first run downloads a reader from HF)...", flush=True)
judge = VegaMLInstructionJudge()
print("loaded.\n", flush=True)

done = 0
for ep in ds.iter_episodes():
    if not ep.video_refs:
        continue
    ref = next(iter(ep.video_refs.values()))
    frames = video_mod.extract_frames(ref["path"], ref["from_ts"], ref["to_ts"], 4)
    if not frames:
        continue
    # raw vegaml call for the first episode, to inspect the full structure
    if done == 0:
        img = _montage(frames)
        q = {"success": {"type": "boolean",
                         "instructions": "These frames are time-ordered from a "
                         "robot-arm demo attempting: \"%s\". Does it accomplish "
                         "that?" % (ep.task or "(none)"),
                         "criteria": {"true": "task completed",
                                      "false": "task not completed"}}}
        raw = judge.v.decide_image(img, q)
        print("=== RAW vegaml output (ep %d) ===" % ep.index)
        print(json.dumps(raw, indent=2, default=str)[:1500])
        print("=== answers['success'] keys:",
              list(raw["answers"]["success"].keys()), "===\n")

    v = judge.judge(ep.task, frames, {})
    print("ep %4d | task=%-40s | score=%.3f success=%s abstain=%s"
          % (ep.index, (ep.task or "")[:40], v["instruction_score"],
             v["success"], v["abstain"]))
    done += 1
    if done >= N:
        break

print("\nsmoke-tested %d real episodes." % done)
