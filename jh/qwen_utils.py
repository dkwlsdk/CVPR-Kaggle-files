import math
import os
import time
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from config import (
    FIX_CLIP_SEC,
    FIX_TIME_MAX_FRAMES,
    FIX_TIME_MIN_FRAMES,
    FIX_TIME_TARGET_FPS,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    RAW_JSONL,
    TYPE_MAX_FRAMES,
    TYPE_MIN_FRAMES,
    TYPE_TARGET_FPS,
)
from utils import LOGGER, append_jsonl, extract_first_json_object, extract_time_fallback


def compute_num_frames(
    duration_sec: Optional[float],
    target_fps: float,
    min_frames: int,
    max_frames: int,
) -> int:
    if duration_sec is None or duration_sec <= 0:
        return min_frames
    n = int(math.ceil(duration_sec * target_fps))
    return max(min_frames, min(max_frames, n))


class QwenRunner:
    def __init__(self, model_name: str = MODEL_NAME):
        LOGGER.info(f"Loading Qwen model: {model_name}")

        hf_token = os.getenv("HF_TOKEN", None)

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            token=hf_token,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            token=hf_token,
        )

        self.input_device = self._infer_input_device()

        # num_frames와 fps 충돌 방지
        if hasattr(self.processor, "video_processor") and self.processor.video_processor is not None:
            if hasattr(self.processor.video_processor, "fps"):
                self.processor.video_processor.fps = None

        LOGGER.info(f"Qwen input device: {self.input_device}")

    def _infer_input_device(self):
        for p in self.model.parameters():
            if p.device.type != "meta":
                return p.device
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.input_device) if hasattr(v, "to") else v
        return out

    def _build_mm_kwargs(
        self,
        media_type: str,
        stage: str,
        duration_hint: Optional[float] = None,
    ) -> Dict[str, Any]:
        if media_type != "video":
            return {}

        if stage == "fix_time":
            num_frames = compute_num_frames(
                duration_sec=duration_hint,
                target_fps=FIX_TIME_TARGET_FPS,
                min_frames=FIX_TIME_MIN_FRAMES,
                max_frames=FIX_TIME_MAX_FRAMES,
            )
            LOGGER.info(
                f"[qwen] stage=fix_time | duration={duration_hint} | num_frames={num_frames}"
            )
            return {
                "num_frames": num_frames,
                "fps": None,
            }

        if stage == "type":
            clip_duration = duration_hint if duration_hint is not None else FIX_CLIP_SEC
            num_frames = compute_num_frames(
                duration_sec=clip_duration,
                target_fps=TYPE_TARGET_FPS,
                min_frames=TYPE_MIN_FRAMES,
                max_frames=TYPE_MAX_FRAMES,
            )
            LOGGER.info(
                f"[qwen] stage=type | duration={clip_duration} | num_frames={num_frames}"
            )
            return {
                "num_frames": num_frames,
                "fps": None,
            }

        # 혹시 다른 video stage가 생길 경우의 기본값
        return {
            "num_frames": 16,
            "fps": None,
        }

    def run_json(
        self,
        media_type: str,
        media_path: str,
        prompt: str,
        rel_path: str,
        stage: str,
        max_retries: int = 3,
        duration_hint: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a precise traffic accident analysis assistant. "
                            "Do not reveal chain-of-thought. "
                            "Return exactly one compact JSON object only. "
                            "No markdown. No explanation. No bullet points. "
                            "The first character of your response must be '{' "
                            "and the last character must be '}'."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": media_type, "path": media_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                mm_kwargs = self._build_mm_kwargs(
                    media_type=media_type,
                    stage=stage,
                    duration_hint=duration_hint,
                )

                try:
                    processed = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        enable_thinking=False,
                        **mm_kwargs,
                    )
                except TypeError:
                    processed = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        **mm_kwargs,
                    )

                processed = self._move_to_device(processed)

                generate_kwargs = dict(
                    **processed,
                    max_new_tokens=96 if stage == "fix_time" else MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

                with torch.inference_mode():
                    output_ids = self.model.generate(**generate_kwargs)

                prompt_len = processed["input_ids"].shape[1]
                gen_ids = output_ids[:, prompt_len:]

                raw = self.processor.tokenizer.batch_decode(
                    gen_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

                append_jsonl(
                    RAW_JSONL,
                    {
                        "path": rel_path,
                        "stage": stage,
                        "attempt": attempt,
                        "raw_output": raw,
                    },
                )

                parsed = extract_first_json_object(raw)
                if parsed is not None:
                    return parsed

                if stage == "fix_time":
                    fallback = extract_time_fallback(raw, duration=duration_hint)
                    if fallback is not None:
                        LOGGER.warning(
                            f"[{stage}] {rel_path} | JSON parse failed but fallback time recovered: {fallback}"
                        )
                        return fallback

                last_error = f"json parse failed: {raw[:500]}"

            except Exception as e:
                last_error = repr(e)

            time.sleep(1.0)

        LOGGER.error(f"Qwen failed | path={rel_path} | stage={stage} | err={last_error}")
        return None