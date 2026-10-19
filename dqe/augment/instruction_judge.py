"""VLM instruction-following judge.

Given an episode's language instruction plus a few sampled frames, estimate
whether the demonstration actually accomplishes the instruction. This mirrors
RLDX-1's "video quality filtering (instruction following)" stage.

Three backends:
  * ``heuristic`` — no model. Infers task completion from trajectory signals
    (gripper actuation, useful motion, tracking quality). Runs fully offline
    and is the default so the pipeline never requires network/GPU.
  * ``claude``    — Anthropic SDK vision call (``claude-opus-4-8`` by default).
    Faithful VLM judge. Requires ``pip install anthropic`` + ANTHROPIC_API_KEY
    and sends sampled frames to the API (opt-in).
  * ``hf``        — local Qwen2-VL-class model via transformers on GPU.
    Faithful to RLDX's Qwen3-VL judge; needs the model weights downloaded.

Every backend returns the same dict:
    {"instruction_score": float in [0,1], "success": bool, "reason": str}
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "score": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["success", "score", "reason"],
    "additionalProperties": False,
}

_PROMPT = (
    "You are judging a robot-arm demonstration. The frames below are sampled "
    "in time order from a single episode of a Piper 7-DOF arm attempting the "
    "instruction:\n\n  \"{instruction}\"\n\n"
    "Decide whether the demonstration successfully accomplishes that "
    "instruction. Return `success` (did it accomplish the task), `score` (0.0 "
    "= clearly failed, 1.0 = cleanly completed), and a one-sentence `reason`."
)


class InstructionJudge:
    """Base class. Subclasses implement ``judge``."""

    def judge(self, instruction: Optional[str], frames: List[np.ndarray],
              metrics: Dict[str, float]) -> Dict:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
class HeuristicInstructionJudge(InstructionJudge):
    """Offline proxy for pick/place style tasks using trajectory metrics."""

    def judge(self, instruction, frames, metrics):
        grip = metrics.get("gripper_travel", 0.0)
        path = metrics.get("path_length", 0.0)
        idle = metrics.get("idle_ratio", 1.0)
        track = metrics.get("tracking_err_mean", 1.0)

        # pick/place needs a gripper actuation, meaningful motion, low idle,
        # and faithful execution.
        grip_s = min(grip / 0.05, 1.0)
        motion_s = min(path / 3.0, 1.0)
        idle_s = max(0.0, 1.0 - idle / 0.6)
        track_s = max(0.0, 1.0 - track / 0.2)
        score = float(0.4 * grip_s + 0.3 * motion_s + 0.2 * idle_s + 0.1 * track_s)
        success = score >= 0.5
        reason = ("heuristic: gripper=%.3f path=%.2f idle=%.2f track=%.3f"
                  % (grip, path, idle, track))
        return {"instruction_score": score, "success": success, "reason": reason}


# --------------------------------------------------------------------------- #
def _encode_jpeg(frame_bgr: np.ndarray) -> Optional[str]:
    import base64
    import cv2
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


class ClaudeInstructionJudge(InstructionJudge):
    """Anthropic vision judge. Lazy-imports the SDK; opt-in (external calls)."""

    def __init__(self, model: str = "claude-opus-4-8"):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "backend=claude needs the Anthropic SDK: pip install anthropic"
            ) from e
        import anthropic
        if not (os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError(
                "backend=claude needs ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN)"
            )
        self.model = model
        self.client = anthropic.Anthropic()

    def judge(self, instruction, frames, metrics):
        if not frames:
            return {"instruction_score": 0.0, "success": False,
                    "reason": "no frames to judge"}
        content: List[dict] = []
        for fr in frames:
            data = _encode_jpeg(fr)
            if data is None:
                continue
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": data},
            })
        content.append({"type": "text",
                        "text": _PROMPT.format(instruction=instruction or "(none)")})

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
            output_config={"format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}},
        )
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        data = json.loads(text)
        score = float(max(0.0, min(1.0, data.get("score", 0.0))))
        return {"instruction_score": score,
                "success": bool(data.get("success", score >= 0.5)),
                "reason": "claude: " + str(data.get("reason", ""))}


# --------------------------------------------------------------------------- #
class HFVLMInstructionJudge(InstructionJudge):
    """Local Qwen2-VL-class judge via transformers. Experimental — requires the
    model weights and a GPU. Faithful to RLDX's Qwen3-VL instruction judge."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except Exception as e:  # pragma: no cover - heavy optional path
            raise RuntimeError(
                "backend=hf needs transformers+torch and a downloadable VLM"
            ) from e
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto")

    def judge(self, instruction, frames, metrics):  # pragma: no cover - heavy
        from PIL import Image
        import cv2
        pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        msg = [{"role": "user", "content":
                [{"type": "image"} for _ in pil]
                + [{"type": "text",
                    "text": _PROMPT.format(instruction=instruction or "(none)")
                    + " Reply as JSON {\"success\":bool,\"score\":0-1,\"reason\":str}."}]}]
        prompt = self.processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=pil, return_tensors="pt").to(
            self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=200)
        text = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        try:
            data = json.loads(text[text.index("{"):text.rindex("}") + 1])
        except Exception:
            data = {"success": False, "score": 0.0, "reason": text[:120]}
        score = float(max(0.0, min(1.0, data.get("score", 0.0))))
        return {"instruction_score": score,
                "success": bool(data.get("success", score >= 0.5)),
                "reason": "hf: " + str(data.get("reason", ""))}


def build_judge(cfg) -> InstructionJudge:
    backend = (cfg.backend or "heuristic").lower()
    if backend == "heuristic":
        return HeuristicInstructionJudge()
    if backend == "claude":
        return ClaudeInstructionJudge(cfg.model)
    if backend == "hf":
        return HFVLMInstructionJudge(cfg.hf_model)
    raise ValueError("unknown instruction backend: %s" % backend)
