"""Augmentation & VLM-judge layer.

Implements the two RLDX-1 data-pipeline pieces that sit on top of the quality
core: a VLM instruction-following judge (``instruction_judge``) and synthetic
data augmentation (``task_augment`` + ``trajectory_augment`` + ``synthesize``).
All model-backed paths are optional and lazily imported so the core still runs
with no extra dependencies.
"""
