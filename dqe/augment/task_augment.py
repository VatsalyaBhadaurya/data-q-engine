"""Task (instruction) augmentation — RLDX-1's VLM-generated instruction variants.

Produces paraphrases of each task string so a policy sees linguistic diversity
for the same behaviour. Offline backend uses deterministic rule-based rewrites
(no dependencies); the Claude backend asks the model for natural paraphrases.
"""

from __future__ import annotations

import json
import os
from typing import List

_SYNONYMS = [
    ("pick", ["grab", "take", "lift", "pick up"]),
    ("put", ["place", "drop", "set"]),
    ("into the box", ["in the box", "inside the box", "into the container"]),
    ("the yellow block", ["the yellow cube", "the yellow piece"]),
    ("the black block", ["the black cube", "the black piece"]),
]


class TaskAugmenter:
    def paraphrase(self, instruction: str, n: int) -> List[str]:
        raise NotImplementedError


class OfflineTaskAugmenter(TaskAugmenter):
    """Deterministic synonym/template rewrites. No network."""

    def paraphrase(self, instruction, n):
        base = (instruction or "").strip()
        variants = []
        for frm, tos in _SYNONYMS:
            if frm in base:
                for to in tos:
                    v = base.replace(frm, to, 1)
                    if v != base and v not in variants:
                        variants.append(v)
        # light templating
        if base:
            variants.append("please " + base)
            variants.append(base + " carefully")
        # de-dupe, cap
        seen, out = set(), []
        for v in variants:
            if v not in seen:
                seen.add(v)
                out.append(v)
            if len(out) >= n:
                break
        return out


class ClaudeTaskAugmenter(TaskAugmenter):
    """Natural paraphrases via the Anthropic SDK (opt-in; external calls)."""

    def __init__(self, model: str = "claude-opus-4-8"):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise RuntimeError("backend=claude needs: pip install anthropic") from e
        import anthropic
        if not (os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError("backend=claude needs ANTHROPIC_API_KEY")
        self.model = model
        self.client = anthropic.Anthropic()

    def paraphrase(self, instruction, n):
        schema = {"type": "object",
                  "properties": {"variants": {"type": "array",
                                              "items": {"type": "string"}}},
                  "required": ["variants"], "additionalProperties": False}
        resp = self.client.messages.create(
            model=self.model, max_tokens=512,
            messages=[{"role": "user", "content":
                       "Give %d short, natural paraphrases of this robot task "
                       "instruction, preserving its exact meaning:\n\n\"%s\""
                       % (n, instruction or "")}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        return list(json.loads(text).get("variants", []))[:n]


def build_augmenter(backend: str, model: str = "claude-opus-4-8") -> TaskAugmenter:
    backend = (backend or "offline").lower()
    if backend == "offline":
        return OfflineTaskAugmenter()
    if backend == "claude":
        return ClaudeTaskAugmenter(model)
    raise ValueError("unknown task-augment backend: %s" % backend)
