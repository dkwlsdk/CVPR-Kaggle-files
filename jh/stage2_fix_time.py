import argparse
from pathlib import Path
from typing import Dict, Optional

from qwen_utils import QwenRunner
from utils import (
    LOGGER,
    get_stage_paths,
    get_video_info,
    read_metadata,
    save_clip_centered,
    save_frame_image,
    save_json,
    validate_time_prediction,
    write_csv,
)


def build_fix_time_prompt(meta: Dict[str, str], best_candidate_time: Optional[float], duration: float) -> str:
    region = meta.get("region", "")
    scene_layout = meta.get("scene_layout", "")
    weather = meta.get("weather", "")
    day_time = meta.get("day_time", "")
    quality = meta.get("quality", "")
    no_frames = meta.get("no_frames", "")
    height = meta.get("height", "")
    width = meta.get("width", "")

    cand_str = f"{best_candidate_time:.3f}" if best_candidate_time is not None else "none"

    return f"""
You are a CCTV traffic accident timing expert.

Your Task:
Find the FIRST true accident onset time in the FULL video.

Video metadata:
- scene_layout: {scene_layout}
- weather: {weather}
- day_time: {day_time}
- quality: {quality}
- duration_seconds: {duration:.3f}
- no_frames: {no_frames}
- frame_height: {height}
- frame_width: {width}

Weak reference candidate time (may be wrong):
{cand_str}

Instructions:
1. Analyze the FULL video carefully.
2. Decide the accident_time (in seconds) when a traffic accident CLEARLY BEGINS.
3. accident_time must correspond to the earliest collision moment:
   - the first frame where physical contact begins, or
   - the first frame where collision is clearly unavoidable and immediate.
4. Do not rely on the candidate.
5. Use the candidate only as a weak hint.
6. If the candidate is wrong, ignore it.
7. Ignore the exact location and the accident type in this step.
8. Focus only on accurately detecting the first accident_time.

Critical output rules:
- Output JSON only.
- No reasoning.
- No analysis.
- No markdown.
- No bullet points.
- No code block.
- No text before JSON.
- No text after JSON.
- The JSON must contain exactly this key:
  "accident_time"

{{
  "accident_time": <float>
}}
""".strip()


def run_stage2_for_video(
    qwen: QwenRunner,
    rel_path: str,
    video_path: str,
    meta: Dict[str, str],
    best_candidate_time: Optional[float],
) -> Optional[Dict[str, object]]:
    paths = get_stage_paths(rel_path)
    info = get_video_info(video_path)
    prompt = build_fix_time_prompt(meta, best_candidate_time, info["duration"])
    LOGGER.info(f"[stage2] {rel_path} | best_candidate={best_candidate_time}")

    obj = qwen.run_json(
    "video",
    video_path,
    prompt,
    rel_path,
    "fix_time",
    duration_hint=info["duration"],
)
    if obj is None:
        return None

    accident_time = validate_time_prediction(obj, info["duration"])
    if accident_time is None:
        return None

    frame_info = save_frame_image(video_path, paths["stage2_frame_jpg"], accident_time)
    clip_info = save_clip_centered(video_path, paths["stage2_clip_mp4"], accident_time)

    row = {
        "path": rel_path,
        "accident_time": round(accident_time, 3),
        "frame_path": str(paths["stage2_frame_jpg"]),
        "clip_path": str(paths["stage2_clip_mp4"]),
        "frame_index": frame_info["frame_index"],
        "clip_start": clip_info["start_sec"],
        "clip_end": clip_info["end_sec"],
        "source": str(obj.get("source", "")).strip(),
    }
    save_json(paths["stage2_fixed_json"], {
        **row,
        "best_candidate_time_hint": best_candidate_time,
        "policy": "Best candidate is a heuristic hint only. Qwen may output refined or non-candidate time if clearer.",
    })
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--videos-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    qwen = QwenRunner()
    rows_out = []
    metas = [row for row in read_metadata(Path(args.metadata)) if row.get("path")]
    if args.limit is not None:
        metas = metas[: max(0, args.limit)]

    for meta in metas:
        rel_path = meta["path"]
        video_path = str((Path(args.videos_root) / rel_path) if args.videos_root else (Path(args.metadata).parent / rel_path))
        try:
            row = run_stage2_for_video(qwen, rel_path, video_path, meta, None)
            if row is not None:
                rows_out.append(row)
        except Exception as e:
            LOGGER.exception(f"stage2 failed: {rel_path} | {e}")

    stage2_csv = Path(args.metadata).parent / "outputs" / "stage2_fixed_times.csv"
    write_csv(
        stage2_csv,
        rows_out,
        ["path", "accident_time", "frame_path", "clip_path", "frame_index", "clip_start", "clip_end", "source"],
    )
    LOGGER.info(f"saved: {stage2_csv}")


if __name__ == "__main__":
    main()
