import argparse
from pathlib import Path
from typing import Dict, List

from config import FINAL_CSV, STAGE1_CSV, STAGE2_CSV, STAGE3_CSV, STAGE4_CSV
from qwen_utils import QwenRunner
from stage1_candidate_times import process_one_video as run_stage1_for_video
from stage2_fix_time import run_stage2_for_video
from stage3_type import run_stage3_for_video
from stage4_location import run_stage4_for_video
from utils import LOGGER, get_stage_paths, read_metadata, write_csv


FINAL_FIELDNAMES = ["path", "accident_time", "center_x", "center_y", "type"]


def select_rows(rows: List[Dict[str, str]], limit: int | None) -> List[Dict[str, str]]:
    if limit is None:
        return rows
    return rows[: max(0, limit)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="./dataset/test_metadata.csv")
    parser.add_argument("--videos-root", default="./dataset")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N videos from metadata.")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--skip-stage3", action="store_true")
    parser.add_argument("--skip-stage4", action="store_true")
    args = parser.parse_args()

    meta_rows = [row for row in read_metadata(Path(args.metadata)) if row.get("path")]
    meta_rows = select_rows(meta_rows, args.limit)
    LOGGER.info(f"Selected videos: {len(meta_rows)}")

    need_qwen = not (args.skip_stage2 and args.skip_stage3 and args.skip_stage4)
    qwen = QwenRunner() if need_qwen else None

    stage1_rows: List[Dict[str, object]] = []
    stage2_rows: List[Dict[str, object]] = []
    stage3_rows: List[Dict[str, object]] = []
    stage4_rows: List[Dict[str, object]] = []
    final_rows: List[Dict[str, object]] = []

    for idx, meta in enumerate(meta_rows, start=1):
        rel_path = meta["path"]
        video_path = str(Path(args.videos_root) / rel_path) if args.videos_root else str(Path(args.metadata).parent / rel_path)
        paths = get_stage_paths(rel_path)

        LOGGER.info(f"===== [{idx}/{len(meta_rows)}] {rel_path} =====")

        stage1_out = None
        stage2_out = None
        stage3_out = None
        stage4_out = None

        try:
            if not args.skip_stage1:
                stage1_out = run_stage1_for_video(rel_path, video_path)
            else:
                LOGGER.info(f"[stage1] skipped: {rel_path}")

            if stage1_out is not None:
                stage1_rows.append({
                    "path": rel_path,
                    "best_candidate_time": stage1_out.get("best_candidate_time", ""),
                    "best_candidate_score": stage1_out.get("best_candidate_score", ""),
                    "candidate_1": stage1_out.get("candidate_1", ""),
                    "candidate_2": stage1_out.get("candidate_2", ""),
                    "candidate_3": stage1_out.get("candidate_3", ""),
                })
        except Exception as e:
            LOGGER.exception(f"stage1 failed: {rel_path} | {e}")

        try:
            if not args.skip_stage2:
                if qwen is None:
                    raise RuntimeError("Qwen is required for stage2")
                best_candidate_time = stage1_out.get("best_candidate_time") if stage1_out else None
                stage2_out = run_stage2_for_video(qwen, rel_path, video_path, meta, best_candidate_time)
            else:
                LOGGER.info(f"[stage2] skipped: {rel_path}")

            if stage2_out is not None:
                stage2_rows.append(stage2_out)
        except Exception as e:
            LOGGER.exception(f"stage2 failed: {rel_path} | {e}")

        try:
            if not args.skip_stage3 and stage2_out is not None:
                if qwen is None:
                    raise RuntimeError("Qwen is required for stage3")
                stage3_out = run_stage3_for_video(
                    qwen,
                    rel_path,
                    str(stage2_out["clip_path"]),
                    meta,
                    float(stage2_out["accident_time"]),
                )
            elif not args.skip_stage3:
                LOGGER.warning(f"[stage3] skipped because stage2 missing: {rel_path}")
            else:
                LOGGER.info(f"[stage3] skipped: {rel_path}")

            if stage3_out is not None:
                stage3_rows.append(stage3_out)
        except Exception as e:
            LOGGER.exception(f"stage3 failed: {rel_path} | {e}")

        try:
            if not args.skip_stage4 and stage2_out is not None and stage3_out is not None:
                if qwen is None:
                    raise RuntimeError("Qwen is required for stage4")
                stage4_out = run_stage4_for_video(
                    qwen=qwen,
                    rel_path=rel_path,
                    frame_path=str(stage2_out["frame_path"]),
                    meta=meta,
                    accident_time=float(stage2_out["accident_time"]),
                    accident_type=str(stage3_out["type"]),
                    is_single=bool(stage3_out["is_single"]),
                    type_reason=str(stage3_out.get("reason", "")),
                )
            elif not args.skip_stage4:
                LOGGER.warning(f"[stage4] skipped because stage2/stage3 missing: {rel_path}")
            else:
                LOGGER.info(f"[stage4] skipped: {rel_path}")

            if stage4_out is not None:
                stage4_rows.append(stage4_out)
        except Exception as e:
            LOGGER.exception(f"stage4 failed: {rel_path} | {e}")

        final_row = {
            "path": rel_path,
            "accident_time": stage2_out.get("accident_time", "") if stage2_out else "",
            "center_x": stage4_out.get("center_x", "") if stage4_out else "",
            "center_y": stage4_out.get("center_y", "") if stage4_out else "",
            "type": stage3_out.get("type", "") if stage3_out else "",
        }
        final_rows.append(final_row)
        write_csv(paths["final_csv"], [final_row], FINAL_FIELDNAMES)

    write_csv(STAGE1_CSV, stage1_rows, ["path", "best_candidate_time", "best_candidate_score", "candidate_1", "candidate_2", "candidate_3"])
    write_csv(STAGE2_CSV, stage2_rows, ["path", "accident_time", "frame_path", "clip_path", "frame_index", "clip_start", "clip_end", "source"])
    write_csv(STAGE3_CSV, stage3_rows, ["path", "type", "is_single", "reason"])
    write_csv(STAGE4_CSV, stage4_rows, ["path", "center_x", "center_y", "box_mode", "reason"])
    write_csv(FINAL_CSV, final_rows, FINAL_FIELDNAMES)

    LOGGER.info(f"saved stage1 summary: {STAGE1_CSV}")
    LOGGER.info(f"saved stage2 summary: {STAGE2_CSV}")
    LOGGER.info(f"saved stage3 summary: {STAGE3_CSV}")
    LOGGER.info(f"saved stage4 summary: {STAGE4_CSV}")
    LOGGER.info(f"saved final: {FINAL_CSV}")


if __name__ == "__main__":
    main()
