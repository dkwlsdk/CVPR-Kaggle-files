import argparse
import csv
from pathlib import Path
from typing import Dict, Optional

from qwen_utils import QwenRunner
from utils import LOGGER, get_stage_paths, read_metadata, save_json, validate_location_prediction, write_csv


def build_location_prompt(
    meta: Dict[str, str],
    accident_time: float,
    accident_type: str,
    is_single: bool,
    type_reason: str,
) -> str:
    scene_layout = meta.get("scene_layout", "")
    weather = meta.get("weather", "")
    day_time = meta.get("day_time", "")
    accident_scope = "single-vehicle accident" if is_single else "multi-vehicle accident"
    prior_reason_text = type_reason.strip() if type_reason else ""

    reason_block = ""
    if prior_reason_text:
        reason_block = f"\nPrior accident-type hint from previous step:\n- predicted_type_reason: {prior_reason_text}\n"

    return f"""
You are an expert traffic accident analyst looking at a single key frame from CCTV footage.

This image corresponds to the FIRST clear moment of a traffic accident in the video
at approximately accident_time = {accident_time:.3f} seconds.

Video metadata:
- scene_layout: {scene_layout}
- weather: {weather}
- day_time: {day_time}

Prior step result:
- accident_scope: {accident_scope}
- predicted_type: {accident_type}
{reason_block}
Your task is to precisely localize the primary collision point in this frame.

Instructions:
1. Focus on the main collision area where vehicles or objects are physically impacting.
2. Output normalized coordinates of the center of this collision region:
   - center_x: from left (0.0) to right (1.0)
   - center_y: from top (0.0) to bottom (1.0)
3. The coordinates must indicate the center of the actual contact region, not the center of the whole vehicle.
4. Use the prior type result only as context. Final localization must follow the visible image.
5. If uncertain, choose the single best estimate.

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
  "center_x", "center_y"

Output format:
{{
  "center_x": <float>,
  "center_y": <float>
}}
""".strip()


def run_stage4_for_video(
    qwen: QwenRunner,
    rel_path: str,
    frame_path: str,
    meta: Dict[str, str],
    accident_time: float,
    accident_type: str,
    is_single: bool,
    type_reason: str,
) -> Optional[Dict[str, object]]:
    paths = get_stage_paths(rel_path)
    LOGGER.info(f"[stage4] {rel_path}")
    obj = qwen.run_json(
        "image",
        frame_path,
        build_location_prompt(meta, accident_time, accident_type, is_single, type_reason),
        rel_path,
        "location",
    )
    if obj is None:
        return None

    loc = validate_location_prediction(obj)
    if loc is None:
        LOGGER.warning(f"[stage4] invalid prediction: {rel_path} | raw={obj}")
        return None

    out = {"path": rel_path, **loc}
    save_json(
        paths["stage4_location_json"],
        {
            **out,
            "accident_type": accident_type,
            "is_single": is_single,
            "type_reason": type_reason,
        },
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2", required=True, help="CSV from stage2_fix_time.py or run_pipeline stage2 summary")
    parser.add_argument("--stage3", required=True, help="CSV from stage3_type.py or run_pipeline stage3 summary")
    parser.add_argument("--metadata", required=True, help="Metadata CSV containing path, scene_layout, weather, day_time")
    args = parser.parse_args()

    qwen = QwenRunner()
    meta_by_path = {row["path"]: row for row in read_metadata(Path(args.metadata)) if row.get("path")}

    with open(args.stage2, "r", newline="", encoding="utf-8") as f:
        stage2_rows = list(csv.DictReader(f))
    with open(args.stage3, "r", newline="", encoding="utf-8") as f:
        stage3_rows = {row["path"]: row for row in csv.DictReader(f) if row.get("path")}

    rows_out = []
    for row2 in stage2_rows:
        rel_path = row2.get("path", "")
        frame_path = row2.get("frame_path", "")
        accident_time_text = row2.get("accident_time", "")

        if not rel_path or not frame_path or accident_time_text == "":
            LOGGER.warning(f"[stage4] skip invalid stage2 row: {row2}")
            continue

        row3 = stage3_rows.get(rel_path)
        meta = meta_by_path.get(rel_path)
        if row3 is None:
            LOGGER.warning(f"[stage4] stage3 row not found: {rel_path}")
            continue
        if meta is None:
            LOGGER.warning(f"[stage4] metadata not found: {rel_path}")
            continue

        try:
            accident_time = float(accident_time_text)
        except Exception:
            LOGGER.warning(f"[stage4] bad accident_time: {rel_path} | value={accident_time_text}")
            continue

        is_single_text = str(row3.get("is_single", "false")).strip().lower()
        is_single = is_single_text == "true"

        out = run_stage4_for_video(
            qwen=qwen,
            rel_path=rel_path,
            frame_path=frame_path,
            meta=meta,
            accident_time=accident_time,
            accident_type=str(row3.get("type", "")),
            is_single=is_single,
            type_reason=str(row3.get("reason", "")),
        )
        if out is not None:
            rows_out.append(out)

    stage4_csv = Path(args.stage2).parent / "stage4_locations.csv"
    write_csv(stage4_csv, rows_out, ["path", "center_x", "center_y", "box_mode", "reason"])
    LOGGER.info(f"saved: {stage4_csv}")


if __name__ == "__main__":
    main()
