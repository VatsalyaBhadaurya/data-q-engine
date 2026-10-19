# Data Q Engine

A local **data quality** scoring + filtering pipeline for LeRobot-format robot
demonstration datasets, inspired by the data-curation stage of the **RLDX-1**
technical report ([arXiv:2605.03269](http://arxiv.org/html/2605.03269v1)).

RLDX-1 curates its demonstration pool with video-quality filtering
(instruction-following + trajectory plausibility), motion-consistency
filtering, and redundancy pruning. The GPU-heavy parts of that pipeline (an
image-to-video synthesizer, a Qwen3-VL instruction judge, simulator rollouts)
aren't runnable locally, so this engine implements the parts that *are* fully
computable from the recorded demonstrations:

| RLDX concept | Local analog implemented here |
| --- | --- |
| Trajectory plausibility | joint-limit checks, discontinuity/glitch detection, smoothness (jerk) |
| Motion consistency | commanded-vs-achieved tracking error (`state[t+1]` vs `action[t]`) |
| Video quality filter | blur (variance of Laplacian), exposure, frozen-frame detection |
| Instruction following | pluggable VLM judge scores whether frames accomplish the task instruction |
| Redundancy pruning | cross-episode near-duplicate detection on trajectory fingerprints |
| Adaptive data selection | per-episode score + threshold -> curated keep-list |
| Synthetic augmentation | generate trajectory variants + instruction paraphrases, then re-filter |

## Install

```bash
pip install -r requirements.txt   # numpy, pandas, pyarrow, opencv-python
```

## Usage

```bash
# trajectory + dedup (fast: ~2s for 1116 episodes)
python -m dqe.cli --dataset piper_dataset_1100 --out out --dedup

# add video-quality metrics (slower; decodes sampled frames per clip)
python -m dqe.cli --dataset piper_dataset_1100 --out out --dedup --video

# stricter curation
python -m dqe.cli --dataset piper_dataset_1100 --out out --dedup --threshold 0.80
```

Flags: `--video`, `--dedup`, `--threshold T`, `--video-fps F`,
`--near-dup-dist D`, `--judge {none,heuristic,claude,hf}`, `--quiet`.

### VLM instruction-following judge

Scores whether each episode's frames actually accomplish its language
instruction (RLDX's instruction-following filter). Three backends:

```bash
# offline proxy from trajectory signals — no model, no network (default judge)
python -m dqe.cli --dataset piper_dataset_1100 --out out --judge heuristic

# Anthropic vision judge (claude-opus-4-8) — sends sampled frames to the API
pip install anthropic && export ANTHROPIC_API_KEY=...   # (PowerShell: $env:ANTHROPIC_API_KEY=...)
python -m dqe.cli --dataset piper_dataset_1100 --out out --video --judge claude

# local Qwen2-VL on GPU (faithful to RLDX's Qwen3-VL judge; needs weights)
python -m dqe.cli --dataset piper_dataset_1100 --out out --video --judge hf
```

The judge adds an `instruction` sub-score; episodes below
`instruction.fail_threshold` get the critical `instruction_fail` flag and are
dropped. `claude`/`hf` are **opt-in** — `heuristic` is the default and the only
backend that runs with no extra dependencies. `claude` sends frames to an
external API (cost + data leaves your machine); enable it deliberately.

### Synthetic data augmentation

Generate new episodes from real ones via label-preserving kinematic transforms
(jitter, time-warp, idle-crop, gripper-timing jitter), score every variant with
the same quality scorer, and keep only those that pass — RLDX's
generate-then-filter loop. Instructions are paraphrased alongside.

```bash
python -m dqe.augment.cli --dataset piper_dataset_1100 --out out \
    --curated out/curated/keep_episodes.json --per-episode 2 --threshold 0.7
```

Flags: `--curated FILE` (augment only kept episodes), `--per-episode N`,
`--threshold T`, `--transforms jitter,time_warp,crop_idle,gripper_jitter`,
`--task-backend {offline,claude}`, `--seed N`. Outputs land in
`out/synthetic/`: a LeRobot-style `data/*.parquet` of synthetic episodes, a
`manifest.json` (provenance: source episode + transform + score per variant),
and `task_paraphrases.json`.

## Outputs (`out/`)

- `episode_scores.csv` - one row per episode: overall score, keep flag,
  all sub-scores (`sub_*`) and raw metrics (`m_*`), and the explanatory flags.
- `report.md` - summary table, flag-frequency table, worst-25 episodes.
- `report.json` - full machine-readable results + the config used.
- `curated/keep_episodes.json` / `.txt` - the filtered episode list, directly
  usable as `LeRobotDataset(root, episodes=[...])`. `curated/usage.md` has a
  paste-ready snippet.

## How scoring works

Each episode gets sub-scores in `[0,1]` (1 = best) for tracking, smoothness,
idle ratio, useful motion, gripper activity, joint limits, and (optionally)
video. The overall score is a weighted mean over the *enabled* sub-scores
(weights renormalise so disabling video/dedup doesn't deflate scores). An
episode is **kept** iff its score >= `pass_threshold`, it is not a
near-duplicate, and it carries no *critical* flag (`contains_nan`,
`near_static`, `too_short`, `video_missing`, `limit_violation`). All
thresholds and weights live in `dqe/config.py`.

## Architecture

```
dqe/
  config.py            thresholds + scoring weights (single source of truth)
  dataset.py           LeRobot v3.0 reader (packed parquet/mp4, episode->clip map)
  metrics/
    trajectory.py      action/state metrics + dedup fingerprint
    video.py           blur / exposure / frozen-frame (opencv, lazy import)
    dedup.py           union-find near-duplicate clustering
  scoring.py           raw metrics -> sub-scores + flags
  engine.py            orchestration: dataset -> metrics -> scores -> dedup
  report.py            CSV / JSON / Markdown writers
  curate.py            curated keep-list writer
  cli.py               command-line entry point (scoring + judge)
  augment/
    instruction_judge.py  VLM judge: heuristic / claude / hf backends
    task_augment.py       instruction paraphrasing: offline / claude
    trajectory_augment.py kinematic transforms (jitter/warp/crop/gripper)
    synthesize.py         generate -> filter -> write synthetic dataset
    cli.py                augmentation entry point
tests/
  make_synthetic.py    builds a tiny dataset with known defects for validation
```

Optional backends: `pip install anthropic` for the `claude` judge/paraphraser;
`transformers`+`torch` (and a downloaded VLM) for the `hf` judge. Neither is
required for the default offline paths.

## Findings on `piper_dataset_1100` (Agilex Piper, 1116 episodes, 606k frames)

Run with `--dedup` at threshold 0.6:

- **Clean dataset.** Mean score 0.885, all 1116 episodes pass the 0.6 bar
  (min 0.636). Tracking error <= 0.018 rad everywhere (commanded ~= achieved),
  zero joint-limit violations, essentially no discontinuities, no
  near-duplicates at distance 0.08.
- **Idle time is the main signal.** Median idle ratio 0.27, up to 0.69; 77
  episodes exceed the 0.40 idle threshold (setup/settle time in ~20 s clips).
  Raise `--threshold` to curate these out.
- **Label bug surfaced.** The task `"pick the yellow block put into the box"`
  appears under two task IDs - one has a trailing space, splitting 80 episodes
  from the other 776. Worth normalising before training.

Tune thresholds in `dqe/config.py` for your own quality bar.
