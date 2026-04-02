from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from extract_test_tracking_features import (
    DEFAULT_SMOOTHING_WINDOW,
    INTER_COLUMNS,
    INTRA_COLUMNS,
    calculate_inter_track_features,
    calculate_intra_track_features,
    ensure_dataframe_columns,
    iou_xyxy,
    resolve_track_ids_from_json,
    write_empty_csvs,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAM3_INPUT_DIR = SCRIPT_DIR / "temp3"
DEFAULT_SAM3_OUTPUT_DIR = DEFAULT_SAM3_INPUT_DIR / "sam3_features"
SAM3_MIN_CONFIDENCE = 0.55
SAM3_NMS_IOU_THRESHOLD = 0.85


def canonicalize_video_name(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"_sam3_tracks$", "", stem)
    stem = re.sub(r"^\d+_", "", stem)
    return stem


def iter_matching_sam3_files(video_name: str, input_dir: Path) -> Iterable[Path]:
    target = canonicalize_video_name(video_name)
    if not input_dir.exists():
        return []

    matches: List[Path] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".csv", ".json"}:
            continue
        if not path.stem.endswith("_sam3_tracks"):
            continue
        if canonicalize_video_name(path.name) == target:
            matches.append(path)
    return matches


def find_matching_sam3_file(video_name: str, input_dir: Path) -> Optional[Path]:
    matches = list(iter_matching_sam3_files(video_name, input_dir))
    for suffix in (".json", ".csv"):
        for path in matches:
            if path.suffix.lower() == suffix:
                return path
    return matches[0] if matches else None


def map_sam3_label(class_id: Optional[int], class_name: str) -> Optional[Tuple[int, str]]:
    label = (class_name or "").strip().lower()
    if not label:
        return None

    if class_id in {6, 7}:
        return None
    if "asphalt" in label or "lane marking" in label or "road surface" in label:
        return None
    if class_id == 1 or "roadside structure" in label or "barrier" in label or "guardrail" in label:
        return 1, "obstacle"
    if class_id in {0, 2, 4, 5}:
        return 0, "vehicle"
    if "vehicle" in label or "collision" in label or "self-crash" in label or "intact" in label:
        return 0, "vehicle"
    return None


def _normalize_detection(
    frame_idx: int,
    class_id: Optional[int],
    class_name: str,
    confidence: Optional[float],
    bbox_xyxy: Sequence[float],
) -> Optional[dict]:
    mapped = map_sam3_label(class_id, class_name)
    if mapped is None:
        return None

    if confidence is None or float(confidence) < SAM3_MIN_CONFIDENCE:
        return None

    if len(bbox_xyxy) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    if x2 <= x1 or y2 <= y1:
        return None

    mapped_class_id, mapped_class_name = mapped
    return {
        "frame_idx": int(frame_idx),
        "class_id": mapped_class_id,
        "class_name": mapped_class_name,
        "confidence": float(confidence),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "area": max(0.0, (x2 - x1) * (y2 - y1)),
    }


def load_sam3_detections(input_path: Path) -> Tuple[str, pd.DataFrame]:
    if input_path.suffix.lower() == ".json":
        with input_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        raw_video_name = Path(payload.get("video_path", input_path.name)).name
        video_name = canonicalize_video_name(raw_video_name)
        rows: List[dict] = []
        for frame in payload.get("frames", []):
            frame_idx = int(frame.get("frame_idx", 0))
            for obj in frame.get("objects", []):
                row = _normalize_detection(
                    frame_idx=frame_idx,
                    class_id=obj.get("class_id"),
                    class_name=str(obj.get("class_name", "")),
                    confidence=obj.get("confidence"),
                    bbox_xyxy=obj.get("bbox_xyxy", []),
                )
                if row is not None:
                    rows.append(row)
        return video_name, pd.DataFrame(rows)

    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
        video_name = canonicalize_video_name(input_path.name)
        if "video" in df.columns and not df["video"].dropna().empty:
            video_name = canonicalize_video_name(str(df["video"].dropna().iloc[0]))

        rows = []
        for record in df.to_dict(orient="records"):
            row = _normalize_detection(
                frame_idx=int(record.get("frame_idx", 0)),
                class_id=None if pd.isna(record.get("class_id")) else int(record.get("class_id")),
                class_name=str(record.get("class_name", "")),
                confidence=None if pd.isna(record.get("confidence")) else float(record.get("confidence")),
                bbox_xyxy=[
                    record.get("x1", 0.0),
                    record.get("y1", 0.0),
                    record.get("x2", 0.0),
                    record.get("y2", 0.0),
                ],
            )
            if row is not None:
                rows.append(row)
        return video_name, pd.DataFrame(rows)

    raise ValueError(f"Unsupported SAM3 file type: {input_path}")


def apply_frame_nms(df: pd.DataFrame, iou_threshold: float = SAM3_NMS_IOU_THRESHOLD) -> pd.DataFrame:
    if df.empty:
        return df

    kept_rows: List[dict] = []
    for (_, class_name), group in df.groupby(["frame_idx", "class_name"], sort=True):
        group = group.sort_values(["confidence", "area"], ascending=[False, False])
        frame_kept: List[dict] = []
        for row in group.to_dict(orient="records"):
            bbox = (row["x1"], row["y1"], row["x2"], row["y2"])
            if any(iou_xyxy(bbox, (item["x1"], item["y1"], item["x2"], item["y2"])) >= iou_threshold for item in frame_kept):
                continue
            frame_kept.append(row)
        kept_rows.extend(frame_kept)

    return pd.DataFrame(kept_rows)


def build_tracking_json(video_name: str, detections_df: pd.DataFrame) -> dict:
    frames: List[dict] = []
    if detections_df.empty:
        return {"video_name": video_name, "frames": frames}

    detections_df = detections_df.sort_values(["frame_idx", "class_name", "x1", "y1"]).reset_index(drop=True)
    for frame_idx, group in detections_df.groupby("frame_idx", sort=True):
        detections = []
        for row in group.to_dict(orient="records"):
            detections.append(
                {
                    # Force re-tracking after per-frame NMS because raw SAM3 prompt-specific IDs are noisy.
                    "track_id": None,
                    "class_id": int(row["class_id"]),
                    "class_name": row["class_name"],
                    "bbox_xyxy": [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
                }
            )
        frames.append({"frame_idx": int(frame_idx), "detections": detections})
    return {"video_name": video_name, "frames": frames}


def process_sam3_file(
    input_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> Tuple[bool, str]:
    video_name, detections_df = load_sam3_detections(input_path)
    intra_path = output_dir / "intra" / f"{video_name}.csv"
    inter_path = output_dir / "inter" / f"{video_name}.csv"

    if not overwrite and intra_path.exists() and inter_path.exists():
        return True, video_name

    output_dir.joinpath("intra").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("inter").mkdir(parents=True, exist_ok=True)

    if detections_df.empty:
        write_empty_csvs(video_name, output_dir)
        return True, video_name

    detections_df = apply_frame_nms(detections_df)
    tracking_json = build_tracking_json(video_name, detections_df)
    resolved_df = resolve_track_ids_from_json(tracking_json)
    if resolved_df.empty:
        write_empty_csvs(video_name, output_dir)
        return True, video_name

    intra_df = calculate_intra_track_features(resolved_df, geometry_transform=None, smoothing_window=smoothing_window)
    inter_df = calculate_inter_track_features(intra_df)

    ensure_dataframe_columns(intra_df, INTRA_COLUMNS).to_csv(intra_path, index=False)
    ensure_dataframe_columns(inter_df, INTER_COLUMNS).to_csv(inter_path, index=False)
    return True, video_name


def build_features_for_video(
    video_name: str,
    sam3_input_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> Optional[Tuple[Path, Path]]:
    sam3_path = find_matching_sam3_file(video_name, sam3_input_dir)
    if sam3_path is None:
        return None

    ok, resolved_video_name = process_sam3_file(
        sam3_path,
        output_dir=output_dir,
        overwrite=overwrite,
        smoothing_window=smoothing_window,
    )
    if not ok:
        return None

    intra_path = output_dir / "intra" / f"{resolved_video_name}.csv"
    inter_path = output_dir / "inter" / f"{resolved_video_name}.csv"
    if not intra_path.exists() or not inter_path.exists():
        return None
    return intra_path, inter_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Qwen intra/inter feature CSVs from SAM3 tracking outputs.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Single SAM3 CSV/JSON file to process. If omitted, --video-name is resolved from --input-dir.",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default="-6SQSDj8cYU_00.mp4",
        help="Target video filename used to find the matching SAM3 file inside --input-dir.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_SAM3_INPUT_DIR, help="Directory containing SAM3 outputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SAM3_OUTPUT_DIR, help="Directory to store intra/inter CSVs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSVs.")
    parser.add_argument("--smoothing-window", type=int, default=DEFAULT_SMOOTHING_WINDOW, help="Motion smoothing window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_path is not None:
        ok, video_name = process_sam3_file(
            args.input_path,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            smoothing_window=args.smoothing_window,
        )
        if not ok:
            raise SystemExit(1)
        print(f"Generated SAM3 features for {video_name}")
        print(f"intra: {args.output_dir / 'intra' / f'{video_name}.csv'}")
        print(f"inter: {args.output_dir / 'inter' / f'{video_name}.csv'}")
        return

    paths = build_features_for_video(
        args.video_name,
        sam3_input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        smoothing_window=args.smoothing_window,
    )
    if paths is None:
        raise SystemExit(f"No SAM3 input matched {args.video_name} in {args.input_dir}")

    print(f"Generated SAM3 features for {canonicalize_video_name(args.video_name)}")
    print(f"intra: {paths[0]}")
    print(f"inter: {paths[1]}")


if __name__ == "__main__":
    main()
