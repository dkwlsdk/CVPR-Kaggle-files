import argparse
import csv
from pathlib import Path
from typing import Dict, Optional

from qwen_utils import QwenRunner
from utils import LOGGER, get_stage_paths, read_metadata, save_json, validate_type_prediction, write_csv

def build_type_prompt(meta: Dict[str, str], accident_time: float) -> str:
    scene_layout = meta.get("scene_layout", "")
    weather = meta.get("weather", "")
    day_time = meta.get("day_time", "")

    return f"""
You are a senior traffic accident type specialist.
This 4-second clip is centered around accident_time = {accident_time:.3f} seconds.

Your Task:
Classify the accident type in this clip.

Video metadata:
- scene_layout: {scene_layout}
- weather: {weather}
- day_time: {day_time}

Definitions of accident types (choose exactly one):
- rear-end: One vehicle crashes into the back of another vehicle traveling in the same direction.
- head-on: Two vehicles traveling in opposite directions collide front-to-front.
- sideswipe: Two vehicles moving in roughly the same direction make side-to-side contact while overlapping partially.
- t-bone: The front of one vehicle crashes into the side of another vehicle, forming a "T" shape.
- single: An accident involving only one vehicle (e.g., hitting a pole, barrier, guardrail, or going off the road) with no other vehicle collision.

Instructions:
1. Carefully analyze the visible interaction between vehicles and/or objects.
2. Use vehicle motion, approach direction, and first contact geometry.
3. Decide whether the primary accident is single-vehicle or multi-vehicle.
4. If single-vehicle, output "single".
5. If multi-vehicle, choose exactly one: "rear-end", "head-on", "sideswipe", "t-bone".
6. Classify the primary collision only.

Critical output rules:
- Output JSON only.
- No reasoning.
- No analysis.
- No markdown.
- No bullet points.
- No code block.
- No text before JSON.
- No text after JSON.
- The JSON must contain exactly these keys:
  "is_single", "type"

Output format:
{{
  "is_single": <true_or_false>,
  "type": "<rear-end|head-on|sideswipe|t-bone|single>"
}}
""".strip()


def run_stage3_for_video(
    qwen: QwenRunner,
    rel_path: str,
    clip_path: str,
    meta: Dict[str, str],
    accident_time: float,
) -> Optional[Dict[str, object]]:
    paths = get_stage_paths(rel_path)
    LOGGER.info(f"[stage3] {rel_path}")
    obj = qwen.run_json("video", clip_path, build_type_prompt(meta, accident_time), rel_path, "type")
    if obj is None:
        return None

    parsed = validate_type_prediction(obj)
    if parsed is None:
        LOGGER.warning(f"[stage3] invalid prediction: {rel_path} | raw={obj}")
        return None

    out = {"path": rel_path, **parsed}
    save_json(paths["stage3_type_json"], out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2", required=True, help="CSV from stage2_fix_time.py or run_pipeline stage2 summary")
    parser.add_argument("--metadata", required=True, help="Metadata CSV containing path, scene_layout, weather, day_time")
    args = parser.parse_args()

    qwen = QwenRunner()
    meta_by_path = {row["path"]: row for row in read_metadata(Path(args.metadata)) if row.get("path")}

    with open(args.stage2, "r", newline="", encoding="utf-8") as f:
        stage2_rows = list(csv.DictReader(f))

    rows_out = []
    for row in stage2_rows:
        rel_path = row.get("path", "")
        clip_path = row.get("clip_path", "")
        accident_time_text = row.get("accident_time", "")

        if not rel_path or not clip_path or accident_time_text == "":
            LOGGER.warning(f"[stage3] skip invalid stage2 row: {row}")
            continue

        meta = meta_by_path.get(rel_path)
        if meta is None:
            LOGGER.warning(f"[stage3] metadata not found: {rel_path}")
            continue

        try:
            accident_time = float(accident_time_text)
        except Exception:
            LOGGER.warning(f"[stage3] bad accident_time: {rel_path} | value={accident_time_text}")
            continue

        out = run_stage3_for_video(qwen, rel_path, clip_path, meta, accident_time)
        if out is not None:
            rows_out.append(out)

    stage3_csv = Path(args.stage2).parent / "stage3_types.csv"
    write_csv(stage3_csv, rows_out, ["path", "type", "is_single", "reason"])
    LOGGER.info(f"saved: {stage3_csv}")


if __name__ == "__main__":
    main()
