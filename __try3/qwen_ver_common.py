#!/usr/bin/env python3
import importlib
import os
import sys
import types
from typing import Callable, Dict, List, Optional

import qwen_prompt as base_cfg


def _retry_note(failed_times: Optional[List[float]]) -> str:
    if not failed_times:
        return ""
    return (
        f"\n\n[CRITICAL NOTE]\n"
        f"You previously estimated accident times at {failed_times}, but those were INCORRECT. "
        f"DO NOT output these times again. Find a DIFFERENT moment in the video."
    )


def _metadata_block(metadata: Dict[str, str]) -> str:
    return "\n".join(
        [
            f"- region: {metadata.get('region', '')}",
            f"- scene_layout: {metadata.get('scene_layout', '')}",
            f"- weather: {metadata.get('weather', '')}",
            f"- day_time: {metadata.get('day_time', '')}",
            f"- quality (before enhancement): {metadata.get('quality', '')}",
            f"- duration (seconds): {metadata.get('duration', '')}",
            f"- no_frames: {metadata.get('no_frames', '')}",
            f"- frame_height: {metadata.get('height', '')}",
            f"- frame_width: {metadata.get('width', '')}",
        ]
    )


def make_time_prompt(
    metadata: Dict[str, str],
    failed_times: Optional[List[float]],
    experiment_name: str,
    instruction_lines: List[str],
    output_rule_lines: Optional[List[str]] = None,
) -> str:
    retry_note = _retry_note(failed_times)
    metadata_block = _metadata_block(metadata)
    output_rule_lines = output_rule_lines or []

    prompt = f"""{retry_note}
You are an expert traffic accident analyst looking at CCTV footage.

Experiment variant: {experiment_name}.

Your task is to detect the first clear traffic accident in the video and return ONLY the accident start time in seconds.

Video metadata:
{metadata_block}

Instructions:
"""
    for idx, line in enumerate(instruction_lines, start=1):
        prompt += f"{idx}. {line}\n"

    prompt += """
Critical output rules:
- The FINAL output MUST be exactly one JSON object. DO NOT use a list [].
- "accident_time" MUST be a single FLOAT (e.g., 12.34). DO NOT include arithmetic expressions or units.
- Include a brief reasoning in English for your choice inside the JSON under the key "reasoning".
- No markdown, no code blocks, no text before or after the JSON.
"""
    for line in output_rule_lines:
        prompt += f"- {line}\n"

    prompt += """- The JSON must contain exactly these keys:
  "reasoning", "accident_time"

Output format:
{
  "reasoning": "<brief explanation in English>",
  "accident_time": <float>
}
"""
    return prompt.strip()


def clone_base_cfg(version_name: str, time_prompt_builder: Callable[[Dict[str, str], Optional[List[float]]], str]):
    cfg = types.ModuleType('qwen_prompt')
    for name in dir(base_cfg):
        if name.startswith('__'):
            continue
        setattr(cfg, name, getattr(base_cfg, name))

    output_dir = os.path.join(base_cfg.BASE_DIR, f'{version_name}_output')
    cfg.OUTPUT_DIR = output_dir
    cfg.PREDICTION_PATH = os.path.join(output_dir, 'predictions.csv')
    cfg.RAW_LOG_PATH = os.path.join(output_dir, 'raw_outputs.jsonl')
    cfg.FRAME_DIR = os.path.join(output_dir, 'frames')
    cfg.build_time_prompt = time_prompt_builder
    return cfg


def run_with_cfg(cfg) -> None:
    sys.modules['qwen_prompt'] = cfg
    if 'qwen_run' in sys.modules:
        del sys.modules['qwen_run']
    qwen_run = importlib.import_module('qwen_run')
    qwen_run.run_pipeline()
