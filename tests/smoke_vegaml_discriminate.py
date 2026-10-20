"""Does the VegaML judge actually discriminate, or rubber-stamp?

For a few real episodes, ask the SAME frames two questions: the true task
instruction vs. a deliberately WRONG one. A judge with real signal should give
a clearly higher P(success) for the true instruction.
"""
import sys

from dqe.dataset import LeRobotDataset
from dqe.metrics import video as video_mod
from dqe.augment.instruction_judge import VegaMLInstructionJudge, _montage

root = sys.argv[1] if len(sys.argv) > 1 else "piper_dataset_1100"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
WRONG = "stack three red cups into a tower"   # clearly not what the arm does

ds = LeRobotDataset(root)
judge = VegaMLInstructionJudge()


def ask(img, instruction):
    q = {"success": {"type": "boolean",
                     "instructions": "These frames are time-ordered from a "
                     "robot-arm demo attempting: \"%s\". Does it accomplish "
                     "that?" % instruction,
                     "criteria": {"true": "task completed",
                                  "false": "task not completed"}}}
    a = judge.v.decide_image(img, q)["answers"]["success"]
    return a["boolean"], a["abstain"]


print("%-5s | %-38s | %-18s | %-18s" % ("ep", "true task", "P(true)", "P(wrong)"))
done = 0
for ep in ds.iter_episodes():
    if not ep.video_refs:
        continue
    ref = next(iter(ep.video_refs.values()))
    frames = video_mod.extract_frames(ref["path"], ref["from_ts"], ref["to_ts"], 4)
    if not frames:
        continue
    img = _montage(frames)
    pt, at = ask(img, ep.task or "(none)")
    pw, aw = ask(img, WRONG)
    print("%-5d | %-38s | %.3f (abs=%-5s) | %.3f (abs=%-5s)  delta=%+.3f"
          % (ep.index, (ep.task or "")[:38], pt, at, pw, aw, pt - pw))
    done += 1
    if done >= N:
        break
print("\nwrong instruction = %r" % WRONG)
