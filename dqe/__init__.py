"""Data Q(uality) Engine.

A local quality scoring + filtering pipeline for LeRobot-format robot
demonstration datasets, inspired by the data-curation stage of the RLDX-1
technical report (arXiv:2605.03269). It scores each episode on trajectory,
motion-consistency, video-quality, and cross-episode redundancy signals,
then emits a per-episode report and a curated train split.
"""

__version__ = "0.1.0"
