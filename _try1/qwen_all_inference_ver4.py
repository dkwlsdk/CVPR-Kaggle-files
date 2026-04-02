#!/usr/bin/env python3
import csv
import json
import math
import os

# Ensure the script only uses GPUs 0 and 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import re
import time
from collections import defaultdict
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ./qwen_all_inference.py

# MODEL_NAME = "Qwen/Qwen3.5-9B"
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

VIDEO_DIR = "/root/Desktop/workspace/ja/CVPR-Kaggle-files/데이터셋/test_videos"
METADATA_CSV_PATH = "/root/Desktop/workspace/ja/CVPR-Kaggle-files/데이터셋/test_metadata.csv"
OUTPUT_DIR = "/root/Desktop/workspace/ja/qwen_inference3-4bit"

# TEST_MODE = False
TEST_MODE = True
TEST_VIDEO_DIR = "/root/Desktop/workspace/ja/temp3"
TEST_LIMIT = 0

FEATURE_ROOT_DIR = "/root/Desktop/workspace/ja/CVPR-Kaggle-files/Detection_Dataset/test_features2"
FEATURE_INTRA_DIR = os.path.join(FEATURE_ROOT_DIR, "intra")
FEATURE_INTER_DIR = os.path.join(FEATURE_ROOT_DIR, "inter")

RESULT_JSON_DIR = os.path.join(OUTPUT_DIR, "results_json")
FRAME_DIR = os.path.join(OUTPUT_DIR, "frames")
RAW_LOG_DIR = os.path.join(OUTPUT_DIR, "raw_logs")
SUBMISSION_CSV_PATH = os.path.join(OUTPUT_DIR, "submission_dl.csv")
RUN_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "run_summary.json")

MIN_PYTHON = (3, 8)
MIN_TRANSFORMERS = (4, 45)

VALID_TYPES = {"rear-end", "head-on", "sideswipe", "t-bone", "single"}
MAX_VIDEO_RETRIES = 2
MAX_NEW_TOKENS = 400
TEMPERATURE = 0.0
TOP_P = 1.0
CUDA_MEMORY_RESERVE_MIB = 256

TIME_TOP_PAIR_K = 5
TIME_TOP_SINGLE_K = 5
LOCAL_FRAME_TOLERANCE = 2
TYPE_TRAJ_PRE_FRAMES = 20
TYPE_TRAJ_POST_FRAMES = 8
LOCATION_FRAME_WINDOW = 8
FEATURE_PROMPT_MAX_CHARS = 2600

DISTANCE_SCALE_FRACTION = 0.35
DISTANCE_SCALE_MIN = 60.0
APPROACH_MAX = 20.0
VREL_MAX = 25.0
BRAKE_MAX = 8.0
VDROP_MAX = 12.0
JERK_MAX = 8.0
DIR_CHANGE_MAX = 1.2
CURVE_MAX = 1.0

VEHICLE_CLASS_NAMES = {
    "bicycle",
    "bike",
    "bus",
    "car",
    "motorbike",
    "motorcycle",
    "pickup",
    "suv",
    "truck",
    "van",
    "vehicle",
}

INTRA_FLOAT_COLUMNS = {
    "original_track_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "dx",
    "dy",
    "dframes",
    "velocity",
    "direction",
    "acceleration",
    "jerk",
    "direction_change",
    "rolling_dx",
    "rolling_dy",
    "traj_direction",
    "curvature",
}
INTRA_INT_COLUMNS = {"frame_idx", "track_id", "class_id"}

INTER_FLOAT_COLUMNS = {
    "distance",
    "dir_A",
    "dir_B",
    "traj_dir_A",
    "traj_dir_B",
    "dx_A",
    "dy_A",
    "dx_B",
    "dy_B",
    "dframes",
    "approach_speed",
    "ttc",
    "relative_angle",
    "trajectory_angle_diff",
    "v_rel",
}
INTER_INT_COLUMNS = {"frame_idx", "track_A", "track_B"}


def append_raw_log(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_submission_csv(rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["path", "accident_time", "center_x", "center_y", "type"]
    with open(SUBMISSION_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        try:
            if math.isnan(float(value)):
                return default
        except (TypeError, ValueError):
            return default
        return float(value)

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        number = float(text)
    except ValueError:
        return default
    if math.isnan(number):
        return default
    return number


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    number = safe_float(value)
    if number is None:
        return default
    try:
        return int(number)
    except (TypeError, ValueError):
        return default


def clamp(value: Optional[float], low: float, high: float) -> float:
    if value is None:
        return low
    return max(low, min(high, float(value)))


def fmt_num(value: Any, digits: int = 3, default: str = "n/a") -> str:
    number = safe_float(value)
    if number is None:
        return default
    return f"{number:.{digits}f}"


def fmt_point(point: Optional[Tuple[Optional[float], Optional[float]]]) -> str:
    if not point:
        return "n/a"
    x, y = point
    if x is None or y is None:
        return "n/a"
    return f"({x:.3f}, {y:.3f})"


def trim_feature_text(text: str, max_chars: int = FEATURE_PROMPT_MAX_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    truncated = text[: max_chars - 14].rstrip()
    return truncated + "\n[truncated]"


def build_secondary_feature_note(
    feature_text: str,
    max_chars: int = 360,
) -> str:
    compact = " ".join(line.strip() for line in feature_text.splitlines() if line.strip())
    if not compact or compact.startswith("No auxiliary tracking feature cues were available."):
        return ""
    if len(compact) > max_chars:
        compact = compact[: max_chars - 14].rstrip() + " [truncated]"
    return (
        "\n\nSecondary tracking reference:\n"
        "- The following cues are noisy and optional. Prioritize the visual evidence.\n"
        f"- {compact}"
    )


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def get_cuda_free_bytes(device_index: int) -> Optional[int]:
    try:
        with torch.cuda.device(device_index):
            free_bytes, _ = torch.cuda.mem_get_info()
        return int(free_bytes)
    except Exception:
        return None


def build_quantization_config() -> Tuple[Optional[BitsAndBytesConfig], Optional[str]]:
    if os.environ.get("QWEN_DISABLE_4BIT", "").strip() == "1":
        return None, "4-bit quantization disabled via QWEN_DISABLE_4BIT=1."

    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        return None, f"bitsandbytes unavailable; loading without 4-bit quantization ({exc})."

    return (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        "4-bit quantization enabled.",
    )


def build_model_load_config() -> Tuple[Dict[str, Any], List[str]]:
    notes: List[str] = []
    manual_device = os.environ.get("QWEN_TARGET_DEVICE", "").strip()
    enable_cpu_offload = os.environ.get("QWEN_ENABLE_CPU_OFFLOAD", "").strip() == "1"
    quantization_config, quantization_note = build_quantization_config()
    if quantization_note:
        notes.append(quantization_note)

    if not torch.cuda.is_available():
        notes.append("CUDA unavailable; loading on CPU.")
        return {"device_map": "cpu", "torch_dtype": torch.float32}, notes

    if manual_device:
        notes.append(f"Using QWEN_TARGET_DEVICE override: {manual_device}")
        config = {"device_map": manual_device, "torch_dtype": "auto"}
        if quantization_config is not None and ("cuda" in manual_device or manual_device.isdigit()):
            config["quantization_config"] = quantization_config
            notes.append("Applied 4-bit quantization to manual GPU device.")
        return config, notes

    max_memory: Dict[Any, str] = {}
    for device_index in range(torch.cuda.device_count()):
        free_bytes = get_cuda_free_bytes(device_index)
        gpu_name = torch.cuda.get_device_name(device_index)
        if free_bytes is None:
            notes.append(f"cuda:{device_index} ({gpu_name}): free memory unavailable")
            continue

        reserve_bytes = CUDA_MEMORY_RESERVE_MIB * 1024 * 1024
        usable_mib = max(1024, (free_bytes // (1024 * 1024)) - CUDA_MEMORY_RESERVE_MIB)
        max_memory[device_index] = f"{usable_mib}MiB"
        notes.append(
            f"cuda:{device_index} ({gpu_name}): free={free_bytes / (1024**3):.2f} GiB, budget={usable_mib} MiB"
        )

    if not max_memory:
        notes.append("GPU memory inspection failed; falling back to cuda:0.")
        config = {"device_map": "cuda:0", "torch_dtype": "auto"}
        if quantization_config is not None:
            config["quantization_config"] = quantization_config
        return config, notes

    if enable_cpu_offload:
        max_memory["cpu"] = "64GiB"
        notes.append("CPU offload enabled via QWEN_ENABLE_CPU_OFFLOAD=1")

    config = {
        "device_map": "auto",
        "max_memory": max_memory,
        "torch_dtype": "auto",
    }
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    return config, notes


def parse_version_tuple(version_text: str) -> tuple:
    parts = []
    for token in version_text.split("."):
        digits = ""
        for ch in token:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def ensure_runtime_compatibility() -> Optional[str]:
    py_version = tuple(map(int, os.sys.version_info[:2]))
    if py_version < MIN_PYTHON:
        return (
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required for {MODEL_NAME}. "
            f"Current Python: {os.sys.version.split()[0]}"
        )

    tf_version = parse_version_tuple(transformers.__version__)
    if tf_version < MIN_TRANSFORMERS:
        return (
            f"transformers {MIN_TRANSFORMERS[0]}.{MIN_TRANSFORMERS[1]}+ is required "
            f"for {MODEL_NAME}. Current transformers: {transformers.__version__}"
        )

    return None


def normalize_metadata(row: Dict[str, str]) -> Dict[str, str]:
    row = dict(row)
    if "scene_layout" not in row and "scene_layoutm" in row:
        row["scene_layout"] = row["scene_layoutm"]
    return row


def load_all_metadata(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            normalized = normalize_metadata(row)
            rel_path = normalized.get("path", "").strip()
            if not rel_path:
                raise ValueError(f"Metadata row {idx} is missing path.")
            rows.append(normalized)
    return rows


def resolve_video_dir() -> str:
    return TEST_VIDEO_DIR if TEST_MODE else VIDEO_DIR


def filter_metadata_for_test(
    metadata_rows: List[Dict[str, str]],
    test_video_dir: str,
    limit: int = 0,
) -> List[Dict[str, str]]:
    if not os.path.isdir(test_video_dir):
        raise FileNotFoundError(f"Test video directory not found: {test_video_dir}")

    available_files = {
        name
        for name in os.listdir(test_video_dir)
        if os.path.isfile(os.path.join(test_video_dir, name))
    }

    filtered = [
        meta for meta in metadata_rows if os.path.basename(meta["path"]) in available_files
    ]

    if limit > 0:
        filtered = filtered[:limit]

    return filtered


def estimate_fps_from_metadata(meta: Dict[str, str]) -> Optional[float]:
    duration = safe_float(meta.get("duration"))
    no_frames = safe_float(meta.get("no_frames"))
    if duration is None or no_frames is None or duration <= 0 or no_frames <= 0:
        return None
    fps = no_frames / duration
    if fps <= 0:
        return None
    return fps


def get_video_dimensions(meta: Dict[str, str]) -> Tuple[Optional[float], Optional[float]]:
    width = safe_float(meta.get("width"))
    height = safe_float(meta.get("height"))
    if width is not None and width <= 0:
        width = None
    if height is not None and height <= 0:
        height = None
    return width, height


def pair_key(track_a: Optional[int], track_b: Optional[int]) -> str:
    if track_a is None or track_b is None:
        return "unknown"
    a, b = sorted((int(track_a), int(track_b)))
    return f"{a}_{b}"


def is_vehicle_class(class_name: Optional[str]) -> bool:
    if not class_name:
        return False
    return class_name.strip().lower() in VEHICLE_CLASS_NAMES


def normalize_box(
    row: Dict[str, Any],
    width: Optional[float],
    height: Optional[float],
) -> Optional[Tuple[float, float, float, float]]:
    if width is None or height is None or width <= 0 or height <= 0:
        return None

    x1 = safe_float(row.get("x1"))
    y1 = safe_float(row.get("y1"))
    x2 = safe_float(row.get("x2"))
    y2 = safe_float(row.get("y2"))
    if None in {x1, y1, x2, y2}:
        return None
    return (
        clamp(x1 / width, 0.0, 1.0),
        clamp(y1 / height, 0.0, 1.0),
        clamp(x2 / width, 0.0, 1.0),
        clamp(y2 / height, 0.0, 1.0),
    )


def normalize_center(
    row: Dict[str, Any],
    width: Optional[float],
    height: Optional[float],
) -> Optional[Tuple[float, float]]:
    if width is None or height is None or width <= 0 or height <= 0:
        return None
    cx = safe_float(row.get("cx"))
    cy = safe_float(row.get("cy"))
    if cx is None or cy is None:
        return None
    return (
        clamp(cx / width, 0.0, 1.0),
        clamp(cy / height, 0.0, 1.0),
    )


def bbox_center_from_row(row: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    cx = safe_float(row.get("cx"))
    cy = safe_float(row.get("cy"))
    if cx is None or cy is None:
        return None
    return cx, cy


def load_csv_rows(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def parse_intra_row(raw_row: Dict[str, str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "video_name": raw_row.get("video_name", "").strip(),
        "track_id_source": raw_row.get("track_id_source", "").strip(),
        "class_name": raw_row.get("class_name", "").strip(),
    }
    for key in INTRA_INT_COLUMNS:
        row[key] = safe_int(raw_row.get(key))
    for key in INTRA_FLOAT_COLUMNS:
        row[key] = safe_float(raw_row.get(key))
    return row


def parse_inter_row(raw_row: Dict[str, str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "video_name": raw_row.get("video_name", "").strip(),
        "pair": raw_row.get("pair", "").strip(),
    }
    for key in INTER_INT_COLUMNS:
        row[key] = safe_int(raw_row.get(key))
    for key in INTER_FLOAT_COLUMNS:
        row[key] = safe_float(raw_row.get(key))
    if not row["pair"]:
        row["pair"] = pair_key(row.get("track_A"), row.get("track_B"))
    return row


def compute_fallback_jerk(intra_rows: List[Dict[str, Any]]) -> None:
    rows_by_track: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in intra_rows:
        track_id = row.get("track_id")
        if track_id is None:
            continue
        rows_by_track[int(track_id)].append(row)

    for rows in rows_by_track.values():
        rows.sort(key=lambda item: (item.get("frame_idx") or -1, safe_float(item.get("x1"), 0.0) or 0.0))
        prev_acceleration: Optional[float] = None
        for row in rows:
            jerk = safe_float(row.get("jerk"))
            if jerk is not None:
                prev_acceleration = safe_float(row.get("acceleration"), 0.0)
                row["jerk"] = jerk
                continue

            dframes = safe_float(row.get("dframes"), 0.0) or 0.0
            acceleration = safe_float(row.get("acceleration"), 0.0) or 0.0
            if prev_acceleration is None or dframes <= 0:
                row["jerk"] = 0.0
            else:
                row["jerk"] = (acceleration - prev_acceleration) / dframes
            prev_acceleration = acceleration


def load_feature_bundle(video_stem: str) -> Dict[str, Any]:
    intra_path = os.path.abspath(os.path.join(FEATURE_INTRA_DIR, f"{video_stem}.csv"))
    inter_path = os.path.abspath(os.path.join(FEATURE_INTER_DIR, f"{video_stem}.csv"))
    bundle: Dict[str, Any] = {
        "status": "missing",
        "intra_path": intra_path,
        "inter_path": inter_path,
        "intra_rows": [],
        "inter_rows": [],
        "error": None,
        "warning": None,
    }

    if not os.path.exists(intra_path) or not os.path.exists(inter_path):
        missing_parts = []
        if not os.path.exists(intra_path):
            missing_parts.append("intra")
        if not os.path.exists(inter_path):
            missing_parts.append("inter")
        bundle["error"] = f"Missing feature CSV(s): {', '.join(missing_parts)}"
        return bundle

    try:
        raw_intra_rows, intra_fieldnames = load_csv_rows(intra_path)
        raw_inter_rows, _ = load_csv_rows(inter_path)
        bundle["intra_rows"] = [parse_intra_row(row) for row in raw_intra_rows]
        bundle["inter_rows"] = [parse_inter_row(row) for row in raw_inter_rows]
        if "jerk" not in intra_fieldnames:
            compute_fallback_jerk(bundle["intra_rows"])
            bundle["warning"] = "Feature CSV was missing jerk; computed fallback jerk during inference."
        else:
            compute_fallback_jerk(bundle["intra_rows"])
        bundle["status"] = "loaded"
    except Exception as exc:
        bundle["status"] = "error"
        bundle["error"] = str(exc)
        bundle["intra_rows"] = []
        bundle["inter_rows"] = []

    return bundle


def compute_pair_risk_score(row: Dict[str, Any], dist_max: float) -> float:
    ttc = safe_float(row.get("ttc"), 9999.0) or 9999.0
    distance = safe_float(row.get("distance"), dist_max) or dist_max
    approach_speed = safe_float(row.get("approach_speed"), 0.0) or 0.0
    v_rel = safe_float(row.get("v_rel"), 0.0) or 0.0

    ttc_score = 1.0 - min(ttc, 5.0) / 5.0
    distance_score = 1.0 - min(distance / max(dist_max, 1.0), 1.0)
    approach_score = clamp(approach_speed / APPROACH_MAX, 0.0, 1.0)
    vrel_score = clamp(v_rel / VREL_MAX, 0.0, 1.0)

    score = 0.40 * ttc_score + 0.25 * distance_score + 0.20 * approach_score + 0.15 * vrel_score
    if approach_speed <= 0:
        score *= 0.6
    return score


def compute_single_risk_score(row: Dict[str, Any]) -> float:
    brake_score = clamp((safe_float(row.get("hard_brake"), 0.0) or 0.0) / BRAKE_MAX, 0.0, 1.0)
    drop_score = clamp((safe_float(row.get("velocity_drop"), 0.0) or 0.0) / VDROP_MAX, 0.0, 1.0)
    jerk_score = clamp((safe_float(row.get("jerk_abs"), 0.0) or 0.0) / JERK_MAX, 0.0, 1.0)
    turn_score = clamp(abs(safe_float(row.get("direction_change"), 0.0) or 0.0) / DIR_CHANGE_MAX, 0.0, 1.0)
    curve_score = clamp((safe_float(row.get("curvature"), 0.0) or 0.0) / CURVE_MAX, 0.0, 1.0)
    score = 0.30 * brake_score + 0.25 * drop_score + 0.20 * jerk_score + 0.15 * turn_score + 0.10 * curve_score
    if is_vehicle_class(row.get("class_name")):
        score *= 1.05
    return score


def prepare_feature_indices(feature_bundle: Dict[str, Any], meta: Dict[str, str]) -> Dict[str, Any]:
    width, height = get_video_dimensions(meta)
    fps = estimate_fps_from_metadata(meta)
    video_diag = math.hypot(width, height) if width is not None and height is not None else None
    dist_max = max((video_diag or 250.0) * DISTANCE_SCALE_FRACTION, DISTANCE_SCALE_MIN)

    intra_rows = sorted(
        feature_bundle.get("intra_rows", []),
        key=lambda row: (
            row.get("track_id") if row.get("track_id") is not None else -1,
            row.get("frame_idx") if row.get("frame_idx") is not None else -1,
        ),
    )
    inter_rows = sorted(
        feature_bundle.get("inter_rows", []),
        key=lambda row: (
            row.get("pair", ""),
            row.get("frame_idx") if row.get("frame_idx") is not None else -1,
        ),
    )

    intra_by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    intra_by_track: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    inter_by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    inter_by_pair: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in intra_rows:
        track_id = row.get("track_id")
        frame_idx = row.get("frame_idx")
        if track_id is None or frame_idx is None:
            continue
        row["bbox_norm"] = normalize_box(row, width, height)
        row["center_norm"] = normalize_center(row, width, height)
        row["time_sec"] = frame_idx / fps if fps is not None else None
        intra_by_track[int(track_id)].append(row)

    for track_rows in intra_by_track.values():
        track_rows.sort(key=lambda item: item.get("frame_idx") or -1)
        prev_velocity: Optional[float] = None
        for row in track_rows:
            velocity = safe_float(row.get("velocity"), 0.0) or 0.0
            row["prev_velocity"] = prev_velocity
            row["velocity_drop"] = max((prev_velocity or velocity) - velocity, 0.0) if prev_velocity is not None else 0.0
            row["hard_brake"] = max(-(safe_float(row.get("acceleration"), 0.0) or 0.0), 0.0)
            row["jerk_abs"] = abs(safe_float(row.get("jerk"), 0.0) or 0.0)
            row["single_risk_score"] = compute_single_risk_score(row)
            prev_velocity = velocity

    for track_rows in intra_by_track.values():
        for row in track_rows:
            frame_idx = row.get("frame_idx")
            if frame_idx is not None:
                intra_by_frame[int(frame_idx)].append(row)

    for row in inter_rows:
        frame_idx = row.get("frame_idx")
        track_a = row.get("track_A")
        track_b = row.get("track_B")
        if frame_idx is None or track_a is None or track_b is None:
            continue
        row["pair"] = row.get("pair") or pair_key(track_a, track_b)
        row["time_sec"] = frame_idx / fps if fps is not None else None
        row["pair_risk_score"] = compute_pair_risk_score(row, dist_max)
        inter_by_frame[int(frame_idx)].append(row)
        inter_by_pair[str(row["pair"])].append(row)

    return {
        "status": feature_bundle.get("status", "missing"),
        "error": feature_bundle.get("error"),
        "warning": feature_bundle.get("warning"),
        "width": width,
        "height": height,
        "fps": fps,
        "video_diag": video_diag,
        "dist_max": dist_max,
        "intra_rows": intra_rows,
        "inter_rows": inter_rows,
        "intra_by_frame": intra_by_frame,
        "intra_by_track": intra_by_track,
        "inter_by_frame": inter_by_frame,
        "inter_by_pair": inter_by_pair,
    }


def build_time_feature_signals(feature_index: Dict[str, Any], meta: Dict[str, str]) -> Dict[str, Any]:
    best_pair_by_key: Dict[str, Dict[str, Any]] = {}
    best_single_by_track: Dict[int, Dict[str, Any]] = {}
    class_counts: Dict[str, int] = defaultdict(int)

    for row in feature_index["intra_rows"]:
        class_name = row.get("class_name") or "unknown"
        class_counts[str(class_name)] += 1
        track_id = row.get("track_id")
        if track_id is None:
            continue
        current = best_single_by_track.get(int(track_id))
        if current is None:
            best_single_by_track[int(track_id)] = row
            continue
        left = (
            int(is_vehicle_class(row.get("class_name"))),
            safe_float(row.get("single_risk_score"), 0.0) or 0.0,
            safe_float(row.get("hard_brake"), 0.0) or 0.0,
            safe_float(row.get("jerk_abs"), 0.0) or 0.0,
        )
        right = (
            int(is_vehicle_class(current.get("class_name"))),
            safe_float(current.get("single_risk_score"), 0.0) or 0.0,
            safe_float(current.get("hard_brake"), 0.0) or 0.0,
            safe_float(current.get("jerk_abs"), 0.0) or 0.0,
        )
        if left > right:
            best_single_by_track[int(track_id)] = row

    for row in feature_index["inter_rows"]:
        key = row.get("pair") or pair_key(row.get("track_A"), row.get("track_B"))
        current = best_pair_by_key.get(str(key))
        if current is None:
            best_pair_by_key[str(key)] = row
            continue
        left = (
            safe_float(row.get("pair_risk_score"), 0.0) or 0.0,
            -(safe_float(row.get("ttc"), 9999.0) or 9999.0),
            -(safe_float(row.get("approach_speed"), 0.0) or 0.0),
        )
        right = (
            safe_float(current.get("pair_risk_score"), 0.0) or 0.0,
            -(safe_float(current.get("ttc"), 9999.0) or 9999.0),
            -(safe_float(current.get("approach_speed"), 0.0) or 0.0),
        )
        if left > right:
            best_pair_by_key[str(key)] = row

    top_pairs = sorted(
        best_pair_by_key.values(),
        key=lambda row: (
            -(safe_float(row.get("pair_risk_score"), 0.0) or 0.0),
            safe_float(row.get("ttc"), 9999.0) or 9999.0,
            safe_float(row.get("distance"), 9999.0) or 9999.0,
        ),
    )[:TIME_TOP_PAIR_K]
    top_singles = sorted(
        best_single_by_track.values(),
        key=lambda row: (
            -int(is_vehicle_class(row.get("class_name"))),
            -(safe_float(row.get("single_risk_score"), 0.0) or 0.0),
            -(safe_float(row.get("hard_brake"), 0.0) or 0.0),
            -(safe_float(row.get("jerk_abs"), 0.0) or 0.0),
        ),
    )[:TIME_TOP_SINGLE_K]

    return {
        "global_summary": {
            "unique_track_count": len(feature_index["intra_by_track"]),
            "unique_pair_count": len(feature_index["inter_by_pair"]),
            "frames_with_tracks": len(feature_index["intra_by_frame"]),
            "class_counts": dict(sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))),
        },
        "top_pairs": top_pairs,
        "top_singles": top_singles,
    }


def get_track_row_near_frame(
    track_rows: List[Dict[str, Any]],
    target_frame_idx: int,
    max_distance: int,
) -> Optional[Dict[str, Any]]:
    candidates = [
        row
        for row in track_rows
        if row.get("frame_idx") is not None and abs(int(row["frame_idx"]) - target_frame_idx) <= max_distance
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            abs(int(row["frame_idx"]) - target_frame_idx),
            -(safe_float(row.get("single_risk_score"), 0.0) or 0.0),
        ),
    )


def build_location_feature_context(feature_index: Dict[str, Any], frame_index: int) -> Dict[str, Any]:
    pair_candidates = [
        row
        for row in feature_index["inter_rows"]
        if row.get("frame_idx") is not None and abs(int(row["frame_idx"]) - frame_index) <= LOCATION_FRAME_WINDOW
    ]
    pair_candidates.sort(
        key=lambda row: (
            abs(int(row["frame_idx"]) - frame_index),
            -(safe_float(row.get("pair_risk_score"), 0.0) or 0.0),
            safe_float(row.get("distance"), 9999.0) or 9999.0,
        )
    )
    selected_pair = pair_candidates[0] if pair_candidates else None
    pair_center_hint = None
    pair_boxes: List[Dict[str, Any]] = []
    if selected_pair is not None:
        for track_id in (selected_pair.get("track_A"), selected_pair.get("track_B")):
            if track_id is None:
                continue
            track_rows = feature_index["intra_by_track"].get(int(track_id), [])
            impact_row = get_track_row_near_frame(track_rows, frame_index, LOCAL_FRAME_TOLERANCE)
            if impact_row is None:
                continue
            pair_boxes.append(
                {
                    "track_id": track_id,
                    "class_name": impact_row.get("class_name"),
                    "bbox_norm": impact_row.get("bbox_norm"),
                    "center_norm": impact_row.get("center_norm"),
                }
            )
        valid_centers = [box["center_norm"] for box in pair_boxes if box.get("center_norm")]
        if valid_centers:
            pair_center_hint = (
                sum(center[0] for center in valid_centers) / len(valid_centers),
                sum(center[1] for center in valid_centers) / len(valid_centers),
            )

    single_candidates = [
        row
        for row in feature_index["intra_rows"]
        if row.get("frame_idx") is not None and abs(int(row["frame_idx"]) - frame_index) <= LOCATION_FRAME_WINDOW
    ]
    single_candidates.sort(
        key=lambda row: (
            abs(int(row["frame_idx"]) - frame_index),
            -(safe_float(row.get("single_risk_score"), 0.0) or 0.0),
        )
    )
    selected_single = single_candidates[0] if single_candidates else None

    return {
        "selected_pair": selected_pair,
        "pair_center_hint": pair_center_hint,
        "pair_boxes": pair_boxes,
        "selected_single": selected_single,
    }


def classify_path_relation(angle: Optional[float]) -> str:
    if angle is None:
        return "unknown"
    if angle < 25.0:
        return "same-direction"
    if 60.0 <= angle <= 120.0:
        return "crossing"
    if angle > 150.0:
        return "opposite-direction"
    return "oblique"


def classify_orientation_relation(angle: Optional[float]) -> str:
    if angle is None:
        return "unknown"
    if angle < 25.0:
        return "rear-end-like"
    if 60.0 <= angle <= 120.0:
        return "side-impact-like"
    if angle > 150.0:
        return "head-on-like"
    return "angled-impact-like"


def format_time_feature_text(feature_index: Dict[str, Any], feature_signals: Dict[str, Any]) -> str:
    if feature_index["status"] != "loaded":
        message = feature_index.get("error") or "No reliable tracking feature CSVs were available."
        return trim_feature_text(message)

    summary = feature_signals["global_summary"]
    lines = [
        "Tracking-derived accident timing cues:",
        (
            f"unique_tracks={summary['unique_track_count']}, "
            f"unique_pairs={summary['unique_pair_count']}, "
            f"frames_with_tracks={summary['frames_with_tracks']}"
        ),
    ]
    if summary["class_counts"]:
        class_items = ", ".join(
            f"{name}:{count}" for name, count in list(summary["class_counts"].items())[:8]
        )
        lines.append(f"class_counts={class_items}")

    if feature_signals["top_pairs"]:
        lines.append("Top multi-object collision cues:")
        for row in feature_signals["top_pairs"]:
            lines.append(
                "  "
                f"frame={row.get('frame_idx')} time={fmt_num(row.get('time_sec'))}s "
                f"pair={row.get('pair')} distance={fmt_num(row.get('distance'))} "
                f"ttc={fmt_num(row.get('ttc'))} approach={fmt_num(row.get('approach_speed'))} "
                f"v_rel={fmt_num(row.get('v_rel'))} path_angle={fmt_num(row.get('trajectory_angle_diff'), 1)} "
                f"relative_angle={fmt_num(row.get('relative_angle'), 1)}"
            )
    else:
        lines.append("Top multi-object collision cues: none")

    if feature_signals["top_singles"]:
        lines.append("Top single-object anomaly cues:")
        for row in feature_signals["top_singles"]:
            lines.append(
                "  "
                f"frame={row.get('frame_idx')} time={fmt_num(row.get('time_sec'))}s "
                f"track={row.get('track_id')} class={row.get('class_name') or 'unknown'} "
                f"velocity={fmt_num(row.get('velocity'))} velocity_drop={fmt_num(row.get('velocity_drop'))} "
                f"accel={fmt_num(row.get('acceleration'))} jerk={fmt_num(row.get('jerk'))} "
                f"dir_change={fmt_num(row.get('direction_change'))} curvature={fmt_num(row.get('curvature'))}"
            )
    else:
        lines.append("Top single-object anomaly cues: none")

    if feature_index.get("warning"):
        lines.append(f"warning={feature_index['warning']}")

    return trim_feature_text("\n".join(lines))


def format_location_feature_text(feature_index: Dict[str, Any], location_context: Dict[str, Any]) -> str:
    if feature_index["status"] != "loaded":
        message = feature_index.get("error") or "No reliable tracking feature CSVs were available."
        return trim_feature_text(message)

    lines = ["Tracking-derived localization cues:"]
    selected_pair = location_context.get("selected_pair")
    if selected_pair is not None:
        lines.append(
            "nearest_pair="
            f"{selected_pair.get('pair')} frame={selected_pair.get('frame_idx')} "
            f"distance={fmt_num(selected_pair.get('distance'))} "
            f"ttc={fmt_num(selected_pair.get('ttc'))} "
            f"approach={fmt_num(selected_pair.get('approach_speed'))}"
        )
        lines.append(f"pair_center_hint={fmt_point(location_context.get('pair_center_hint'))}")
        for box in location_context.get("pair_boxes", [])[:2]:
            lines.append(
                "  "
                f"track={box.get('track_id')} class={box.get('class_name') or 'unknown'} "
                f"center={fmt_point(box.get('center_norm'))} bbox={box.get('bbox_norm')}"
            )
    else:
        lines.append("nearest_pair=none")

    selected_single = location_context.get("selected_single")
    if selected_single is not None:
        lines.append(
            "single_hint="
            f"track={selected_single.get('track_id')} class={selected_single.get('class_name') or 'unknown'} "
            f"frame={selected_single.get('frame_idx')} center={fmt_point(selected_single.get('center_norm'))} "
            f"accel={fmt_num(selected_single.get('acceleration'))} "
            f"jerk={fmt_num(selected_single.get('jerk'))} "
            f"curvature={fmt_num(selected_single.get('curvature'))}"
        )
    else:
        lines.append("single_hint=none")

    return trim_feature_text("\n".join(lines))


def point_inside_bbox(point: Tuple[float, float], box: Tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def distance_from_point_to_bbox(point: Tuple[float, float], box: Tuple[float, float, float, float]) -> float:
    x, y = point
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(math.hypot(dx, dy))


def select_objects_at_predicted_impact(
    feature_index: Dict[str, Any],
    frame_index: int,
    center_x: float,
    center_y: float,
    meta: Dict[str, str],
) -> Dict[str, Any]:
    width = feature_index.get("width")
    height = feature_index.get("height")
    point_norm = (center_x, center_y)
    point_px = None
    if width is not None and height is not None:
        point_px = (center_x * width, center_y * height)

    candidates: List[Dict[str, Any]] = []
    for delta in range(0, LOCAL_FRAME_TOLERANCE + 1):
        for candidate_frame in {frame_index - delta, frame_index + delta}:
            frame_rows = feature_index["intra_by_frame"].get(candidate_frame, [])
            for row in frame_rows:
                bbox_norm = row.get("bbox_norm")
                bbox_px = (
                    safe_float(row.get("x1")),
                    safe_float(row.get("y1")),
                    safe_float(row.get("x2")),
                    safe_float(row.get("y2")),
                )
                use_norm = bbox_norm is not None
                if point_px is not None and None not in bbox_px:
                    inside = point_inside_bbox(point_px, bbox_px)  # type: ignore[arg-type]
                    distance = distance_from_point_to_bbox(point_px, bbox_px)  # type: ignore[arg-type]
                elif use_norm:
                    inside = point_inside_bbox(point_norm, bbox_norm)
                    distance = distance_from_point_to_bbox(point_norm, bbox_norm)
                else:
                    center = bbox_center_from_row(row)
                    if center is None:
                        continue
                    ref_point = point_px if point_px is not None else point_norm
                    distance = float(math.hypot(center[0] - ref_point[0], center[1] - ref_point[1]))
                    inside = False

                candidates.append(
                    {
                        "row": row,
                        "inside": inside,
                        "distance": distance,
                        "frame_gap": abs(candidate_frame - frame_index),
                    }
                )

    candidates.sort(
        key=lambda item: (
            -int(item["inside"]),
            item["distance"],
            item["frame_gap"],
            -(safe_float(item["row"].get("single_risk_score"), 0.0) or 0.0),
        )
    )

    selected: List[Dict[str, Any]] = []
    seen_track_ids = set()
    for candidate in candidates:
        track_id = candidate["row"].get("track_id")
        if track_id is None or track_id in seen_track_ids:
            continue
        selected.append(candidate)
        seen_track_ids.add(track_id)
        if len(selected) >= 2:
            break

    if not selected:
        fallback_rows = sorted(
            feature_index["intra_rows"],
            key=lambda row: (
                abs((row.get("frame_idx") or frame_index) - frame_index),
                -(safe_float(row.get("single_risk_score"), 0.0) or 0.0),
            ),
        )
        for row in fallback_rows:
            track_id = row.get("track_id")
            if track_id is None or track_id in seen_track_ids:
                continue
            selected.append({"row": row, "inside": False, "distance": float("inf"), "frame_gap": abs((row.get("frame_idx") or frame_index) - frame_index)})
            seen_track_ids.add(track_id)
            break

    match_mode = None
    if selected:
        match_mode = "inside_bbox" if any(item["inside"] for item in selected) else "nearest_bbox"
        if len(selected) == 1:
            match_mode = "fallback_single"

    selected_objects = []
    for item in selected:
        row = item["row"]
        selected_objects.append(
            {
                "track_id": row.get("track_id"),
                "frame_idx": row.get("frame_idx"),
                "class_name": row.get("class_name"),
                "bbox_norm": row.get("bbox_norm"),
                "center_norm": row.get("center_norm"),
                "distance_to_point": item["distance"],
                "inside_bbox": item["inside"],
                "single_risk_score": row.get("single_risk_score"),
            }
        )

    return {
        "selected_track_ids": [obj["track_id"] for obj in selected_objects if obj.get("track_id") is not None],
        "matched_frame_idx": selected_objects[0]["frame_idx"] if selected_objects else None,
        "match_mode": match_mode,
        "point_px": point_px,
        "point_norm": point_norm,
        "selected_objects": selected_objects,
    }


def summarize_object_trajectory(
    feature_index: Dict[str, Any],
    track_id: int,
    frame_index: int,
) -> Optional[Dict[str, Any]]:
    track_rows = feature_index["intra_by_track"].get(int(track_id), [])
    if not track_rows:
        return None

    window_rows = [
        row
        for row in track_rows
        if row.get("frame_idx") is not None
        and frame_index - TYPE_TRAJ_PRE_FRAMES <= int(row["frame_idx"]) <= frame_index + TYPE_TRAJ_POST_FRAMES
    ]
    if not window_rows:
        return None

    impact_row = get_track_row_near_frame(track_rows, frame_index, TYPE_TRAJ_PRE_FRAMES + TYPE_TRAJ_POST_FRAMES)
    if impact_row is None:
        return None

    start_row = window_rows[0]
    end_row = window_rows[-1]
    pre_rows = [row for row in window_rows if (row.get("frame_idx") or frame_index) <= frame_index]
    near_rows = [
        row for row in window_rows if abs((row.get("frame_idx") or frame_index) - frame_index) <= LOCAL_FRAME_TOLERANCE
    ]
    mean_velocity_pre = None
    if pre_rows:
        mean_velocity_pre = sum(safe_float(row.get("velocity"), 0.0) or 0.0 for row in pre_rows) / len(pre_rows)

    curvature_near = None
    if near_rows:
        curvature_near = sum(safe_float(row.get("curvature"), 0.0) or 0.0 for row in near_rows) / len(near_rows)

    start_center = bbox_center_from_row(start_row)
    end_center = bbox_center_from_row(end_row)
    movement_span_px = None
    if start_center is not None and end_center is not None:
        movement_span_px = float(math.hypot(end_center[0] - start_center[0], end_center[1] - start_center[1]))

    return {
        "track_id": track_id,
        "class_name": impact_row.get("class_name"),
        "impact_frame_idx": impact_row.get("frame_idx"),
        "start_center_norm": start_row.get("center_norm"),
        "impact_center_norm": impact_row.get("center_norm"),
        "end_center_norm": end_row.get("center_norm"),
        "traj_direction_at_impact": impact_row.get("traj_direction"),
        "mean_velocity_pre_impact": mean_velocity_pre,
        "acceleration_at_impact": impact_row.get("acceleration"),
        "jerk_at_impact": impact_row.get("jerk"),
        "direction_change_at_impact": impact_row.get("direction_change"),
        "curvature_near_impact": curvature_near,
        "movement_span_px": movement_span_px,
        "impact_bbox_norm": impact_row.get("bbox_norm"),
        "impact_single_risk_score": impact_row.get("single_risk_score"),
    }


def summarize_pair_context(
    feature_index: Dict[str, Any],
    track_ids: List[int],
    frame_index: int,
) -> Optional[Dict[str, Any]]:
    if len(track_ids) < 2:
        return None

    key = pair_key(track_ids[0], track_ids[1])
    pair_rows = feature_index["inter_by_pair"].get(key, [])
    if not pair_rows:
        return None

    window_rows = [
        row
        for row in pair_rows
        if row.get("frame_idx") is not None
        and frame_index - TYPE_TRAJ_PRE_FRAMES <= int(row["frame_idx"]) <= frame_index + TYPE_TRAJ_POST_FRAMES
    ]
    if not window_rows:
        return None

    impact_row = min(
        window_rows,
        key=lambda row: (
            abs((row.get("frame_idx") or frame_index) - frame_index),
            safe_float(row.get("distance"), 9999.0) or 9999.0,
        ),
    )
    min_distance = min(safe_float(row.get("distance"), 9999.0) or 9999.0 for row in window_rows)
    min_ttc = min(safe_float(row.get("ttc"), 9999.0) or 9999.0 for row in window_rows)
    path_angle = safe_float(impact_row.get("trajectory_angle_diff"))
    orientation_angle = safe_float(impact_row.get("relative_angle"))

    return {
        "pair": key,
        "impact_frame_idx": impact_row.get("frame_idx"),
        "distance_at_impact": impact_row.get("distance"),
        "min_distance_in_window": min_distance,
        "ttc_at_impact": impact_row.get("ttc"),
        "min_ttc_in_window": min_ttc,
        "approach_speed_at_impact": impact_row.get("approach_speed"),
        "v_rel_at_impact": impact_row.get("v_rel"),
        "relative_angle_at_impact": orientation_angle,
        "trajectory_angle_diff_at_impact": path_angle,
        "path_relation": classify_path_relation(path_angle),
        "orientation_relation": classify_orientation_relation(orientation_angle),
    }


def build_heuristic_interpretation(
    object_summaries: List[Dict[str, Any]],
    pair_summary: Optional[Dict[str, Any]],
) -> List[str]:
    hints: List[str] = []
    if pair_summary is not None:
        path_relation = pair_summary.get("path_relation")
        orientation_relation = pair_summary.get("orientation_relation")
        if path_relation == "same-direction" and orientation_relation == "rear-end-like":
            hints.append("rear-end is the leading hypothesis from trajectory relation.")
        elif path_relation == "same-direction":
            hints.append("sideswipe is plausible if the image shows side overlap or partial lateral contact.")
        elif path_relation == "crossing" and orientation_relation == "side-impact-like":
            hints.append("t-bone is the leading hypothesis from crossing trajectories and side-impact orientation.")
        elif path_relation == "opposite-direction" and orientation_relation == "head-on-like":
            hints.append("head-on is the leading hypothesis from opposite trajectories and frontal orientation.")
        else:
            hints.append("use the pair trajectory relation as the primary cue and the image as verification.")
    else:
        hints.append("pair evidence is weak or unavailable.")

    if len(object_summaries) == 1:
        hints.append("only one reliable object matched the impact point, so single should be strongly considered.")
    elif pair_summary is None and object_summaries:
        strongest_single = max(
            safe_float(obj.get("impact_single_risk_score"), 0.0) or 0.0 for obj in object_summaries
        )
        if strongest_single > 0.45:
            hints.append("single is plausible because the selected object shows strong braking, jerk, or rotation cues.")

    return hints


def build_type_object_context(
    feature_index: Dict[str, Any],
    metadata: Dict[str, str],
    frame_index: int,
    center_x: float,
    center_y: float,
) -> Dict[str, Any]:
    """
    Builds context about objects around the predicted accident location.
    In 'Vision-First' mode, we provide this as auxiliary info rather than primary guidance.
    """
    selection = select_objects_at_predicted_impact(feature_index, frame_index, center_x, center_y, metadata)
    
    context = {
        "selection": selection,
        "tracks": [],
        "pairs": [],
    }
    
    track_ids = selection.get("selected_track_ids", [])
    for tid in track_ids:
        # Use existing summary helper which correctly accesses feature_index["intra_by_track"]
        summary = summarize_object_trajectory(feature_index, int(tid), frame_index)
        if summary:
            context["tracks"].append(summary)
            
    # Include interaction heuristics for the selected tracks as auxiliary context
    if len(track_ids) >= 2:
        pair_summary = summarize_pair_context(feature_index, [int(track_ids[0]), int(track_ids[1])], frame_index)
        if pair_summary:
            context["pairs"].append(pair_summary)
                    
    return context


def format_type_object_feature_text(feature_index: Dict[str, Any], type_context: Dict[str, Any]) -> str:
    lines = []
    
    tracks = type_context.get("tracks", [])
    if tracks:
        lines.append("Visible Object Movements:")
        for t in tracks:
            cls = t.get("class_name", "vehicle")
            tid = t.get("track_id", "unknown")
            vel = t.get("mean_velocity_pre_impact", 0.0)
            acc = t.get("acceleration_at_impact", 0.0)
            lines.append(f"- {cls} (ID {tid}): pre_impact_velocity={vel:.2f}, acceleration={acc:.2f}")

    pairs = type_context.get("pairs", [])
    if pairs:
        lines.append("\nInter-Object Proximity/Trajectory Context:")
        for p in pairs:
            p_id = p.get("pair", "unknown")
            rel = p.get("path_relation", "unknown")
            ori = p.get("orientation_relation", "unknown")
            lines.append(f"- Pair {p_id}: relation={rel}, orientation={ori}")
            
    if not lines:
        return "No specific tracking context was identified for this impact area."
    return "\n".join(lines)


def log_feature_prompt_input(
    raw_log_path: str,
    rel_path: str,
    stage: str,
    feature_index: Dict[str, Any],
    feature_text: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "path": rel_path,
        "stage": stage,
        "feature_status": feature_index.get("status"),
        "feature_text": feature_text,
    }
    if feature_index.get("error"):
        payload["feature_error"] = feature_index.get("error")
    if feature_index.get("warning"):
        payload["feature_warning"] = feature_index.get("warning")
    if extra:
        payload.update(extra)
    append_raw_log(raw_log_path, payload)


def build_time_prompt(metadata: Dict[str, str], feature_text: str = "", attempt: int = 0, failed_times: List[float] = None) -> str:
    region = metadata.get("region", "")
    scene_layout = metadata.get("scene_layout", "")
    weather = metadata.get("weather", "")
    day_time = metadata.get("day_time", "")
    quality = metadata.get("quality", "")
    duration = metadata.get("duration", "")
    no_frames = metadata.get("no_frames", "")
    height = metadata.get("height", "")
    width = metadata.get("width", "")
    secondary_note = build_secondary_feature_note(feature_text)
    
    retry_note = ""
    if attempt > 0:
        retry_note = "\n\n[CRITICAL NOTE]\nYour previous attempt failed to detect a clear collision or produced an invalid response."
        if failed_times:
            retry_note += f" You previously estimated accident times at {failed_times}, but those were INCORRECT. DO NOT output these times again. Find a DIFFERENT moment in the video."
        else:
            retry_note += " Analyze the entire video again carefully with a fresh perspective."

    prompt = f"""{retry_note}
You are an expert traffic accident analyst looking at CCTV footage.

Your task is to detect the first clear traffic accident in the video and return ONLY the accident start time in seconds.

Video metadata:
- region: {region}
- scene_layout: {scene_layout}
- weather: {weather}
- day_time: {day_time}
- quality (before enhancement): {quality}
- duration (seconds): {duration}
- no_frames: {no_frames}
- frame_height: {height}
- frame_width: {width}{secondary_note}

Instructions:
1. Carefully analyze the ENTIRE video.
2. Find the earliest accident_time (in seconds) when a traffic accident CLEARLY BEGINS.
3. accident_time must correspond to the earliest collision moment:
   - the first frame where physical contact begins, or
   - the first frame where collision is clearly unavoidable and immediate.
4. The accident_time is never 0.0.
5. Ignore the exact location and the accident type in this step.
6. Focus only on accurately detecting the first accident_time.
7. {f"Since this video is longer than 3 seconds, the accident_time is almost never between 0 and 1 seconds." if float(duration) >= 3.0 else ""}
{f"8. CRITICAL: You MUST NOT output any of these previously failed times: {failed_times}. Find a DIFFERENT time!" if failed_times else ""}

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

Output format:
{{
  "accident_time": <float>
}}
"""
    return prompt.strip()


def build_location_prompt(metadata: Dict[str, str], accident_time: float, feature_text: str = "") -> str:
    region = metadata.get("region", "")
    scene_layout = metadata.get("scene_layout", "")
    weather = metadata.get("weather", "")
    day_time = metadata.get("day_time", "")
    quality = metadata.get("quality", "")
    duration = metadata.get("duration", "")
    no_frames = metadata.get("no_frames", "")
    height = metadata.get("height", "")
    width = metadata.get("width", "")
    secondary_note = build_secondary_feature_note(feature_text)

    prompt = f"""
You are an expert traffic accident analyst looking at a single key frame from CCTV footage.

This image corresponds to the FIRST clear moment of a traffic accident in the video
at approximately accident_time = {accident_time:.3f} seconds.

Video metadata:
- region: {region}
- scene_layout: {scene_layout}
- weather: {weather}
- day_time: {day_time}
- quality (before enhancement): {quality}
- duration (seconds): {duration}
- no_frames: {no_frames}
- frame_height: {height}
- frame_width: {width}{secondary_note}

Your task is to precisely localize the primary collision point in this frame.

Instructions:
1. Focus on the main collision area where vehicles or objects are physically impacting.
2. Output normalized coordinates of the center of this collision region:
   - center_x: from left (0.0) to right (1.0)
   - center_y: from top (0.0) to bottom (1.0)
3. The coordinates must indicate the center of the actual contact region, not the center of the whole vehicle.
4. Ignore accident type classification in this step.
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
"""
    return prompt.strip()


def build_type_prompt(meta: Dict[str, Any], accident_time: float, feature_text: str) -> str:
    region = meta.get("region", "")
    scene_layout = meta.get("scene_layout", "")
    weather = meta.get("weather", "")
    day_time = meta.get("day_time", "")
    quality = meta.get("quality", "")
    duration = meta.get("duration", "")
    no_frames = meta.get("no_frames", "")
    height = meta.get("height", "")
    width = meta.get("width", "")
    secondary_note = build_secondary_feature_note(feature_text)

    prompt = f"""
You are an expert traffic accident analyst looking at a single key frame from CCTV footage.

This image corresponds to the FIRST clear moment of a traffic accident in the video
at approximately accident_time = {accident_time:.3f} seconds.

Video metadata:
- region: {region}
- scene_layout: {scene_layout}
- weather: {weather}
- day_time: {day_time}
- quality (before enhancement): {quality}
- duration (seconds): {duration}
- no_frames: {no_frames}
- frame_height: {height}
- frame_width: {width}{secondary_note}

Definitions of accident types (choose exactly one):
- rear-end: One vehicle crashes into the back of another vehicle traveling in the same direction.
- head-on: Two vehicles traveling in opposite directions collide front-to-front.
- sideswipe: Two vehicles moving in roughly the same direction make side-to-side contact while overlapping partially.
- t-bone: The front of one vehicle crashes into the side of another vehicle, forming a "T" shape.
- single: An accident involving only one vehicle (e.g., hitting a pole, barrier, guardrail, or going off the road) with no other vehicle collision.

Your task is to classify the accident type in this frame.

Instructions:
1. Carefully analyze the visible interaction between vehicles and/or objects in this image.
2. Use the visual evidence as the main basis for your decision.
3. Choose exactly one type from:
   ["rear-end", "head-on", "sideswipe", "t-bone", "single"].
4. If the optional tracking reference conflicts with the image, trust the image.
5. If uncertain, choose the single best guess.

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
  "type"

Output format:
{{
  "type": "<one of: rear-end, head-on, sideswipe, t-bone, single>"
}}
"""
    return prompt.strip()


def strip_thinking_text(text: str) -> str:
    if not text:
        return text
    # Remove <think>...</think> tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove common conversational preambles like "The user wants me to...", "Okay, I understand...", etc.
    # This is a bit aggressive but helps if the model ignores prefix injection.
    text = re.sub(r"^(?:The user wants|I understand|Okay|Let me|Based on).*?:\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def try_parse_single_json(candidate: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    # Clean up thinking/reasoning prefixes if any
    text = strip_thinking_text(text)
    
    # 1. Try cleaning markdown code blocks
    # Remove ```json ... ``` or just ``` ... ```
    cleaned_text = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL)
    # Also handle cases where only the starting tag exists or trailing text exists
    cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text.strip())
    cleaned_text = re.sub(r"```$", "", cleaned_text.strip())

    # Try parsing the whole cleaned text
    direct = try_parse_single_json(cleaned_text)
    if direct is not None:
        return direct

    # 1.5 Try to find the first '{' and last '}' in the cleaned text if whole parse failed
    try:
        start_idx = cleaned_text.find('{')
        end_idx = cleaned_text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = cleaned_text[start_idx : end_idx + 1]
            parsed = try_parse_single_json(candidate)
            if parsed is not None:
                return parsed
    except Exception:
        pass

    # 2. Fallback: Search for first { and matching } using brace counting
    # This is useful if there's conversational text before/after the JSON
    try:
        start_idx = text.find('{')
        if start_idx != -1:
            depth = 0
            for end in range(start_idx, len(text)):
                ch = text[end]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start_idx : end + 1]
                        parsed = try_parse_single_json(candidate)
                        if parsed is not None:
                            return parsed
    except Exception:
        pass

    return None


def validate_time_prediction(result: Dict[str, Any], meta: Dict[str, str]) -> Optional[float]:
    try:
        accident_time = float(result["accident_time"])
    except (KeyError, TypeError, ValueError):
        return None

    duration = safe_float(meta.get("duration"))
    if duration is not None:
        accident_time = min(max(accident_time, 0.0), duration)
    else:
        accident_time = max(accident_time, 0.0)

    return accident_time


def validate_location_prediction(result: Dict[str, Any]) -> Optional[Dict[str, float]]:
    try:
        center_x = float(result["center_x"])
        center_y = float(result["center_y"])
    except (KeyError, TypeError, ValueError):
        return None

    center_x = min(max(center_x, 0.0), 1.0)
    center_y = min(max(center_y, 0.0), 1.0)
    return {"center_x": center_x, "center_y": center_y}


def perform_type_sanity_check(predicted_type: str, type_context: Dict[str, Any]) -> List[str]:
    """
    Checks for strong contradictions between model prediction and tracking data.
    """
    warnings = []
    pairs = type_context.get("pairs", [])
    
    # Example: head-on predicted but same-direction trajectory
    if predicted_type == "head-on":
        for p in pairs:
            if p.get("path_relation") == "same-direction":
                warnings.append(f"Sanity Check Warning: 'head-on' predicted but Pair {p.get('pair')} shows 'same-direction' trajectory.")
                
    # Example: rear-end predicted but opposite or crossing
    if predicted_type == "rear-end":
        for p in pairs:
            if p.get("path_relation") in ["opposite-direction", "crossing"]:
                 warnings.append(f"Sanity Check Warning: 'rear-end' predicted but Pair {p.get('pair')} shows '{p.get('path_relation')}' trajectory.")

    # Example: single predicted but very close pair with high relative velocity exists
    if predicted_type == "single":
        for p in pairs:
             # Basic heuristic: if distance is low and there's a pair, single might be wrong
             warnings.append(f"Sanity Check Info: 'single' predicted but Pair {p.get('pair')} was identified nearby.")

    return warnings


def validate_type_prediction(raw: Dict[str, Any], type_context: Dict[str, Any]) -> Optional[str]:
    predicted = str(raw.get("type", "")).lower().strip()
    if predicted not in VALID_TYPES:
        return None
    
    # Log sanity checks
    if type_context:
        warnings = perform_type_sanity_check(predicted, type_context)
        for w in warnings:
            print(f"  -> [type] {w}")
        
    return predicted


def move_inputs_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def get_model_input_device(model) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for mapped_device in hf_device_map.values():
            if isinstance(mapped_device, int):
                return torch.device(f"cuda:{mapped_device}")
            if isinstance(mapped_device, str) and mapped_device not in {"cpu", "disk"}:
                return torch.device(mapped_device)

    model_device = getattr(model, "device", None)
    if model_device is not None:
        return torch.device(model_device)

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def extract_frame_at_time(
    video_path: str,
    accident_time: float,
    meta: Dict[str, str],
    fps_override: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    try:
        import cv2  # noqa: F401
    except ImportError:
        return None

    def _open():
        cap_ = cv2.VideoCapture(video_path)
        return cap_ if cap_.isOpened() else None

    cap = _open()
    if cap is None:
        return None

    fps = fps_override if fps_override is not None else cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 and meta.get("duration") and meta.get("no_frames"):
        try:
            fps = float(meta["no_frames"]) / float(meta["duration"])
        except Exception:
            fps = 0

    if fps <= 0:
        cap.release()
        return None

    frame_index = int(accident_time * fps)
    frame_index = max(0, min(frame_index, max(0, total_frames - 1)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if ret and frame is not None:
        cap.release()
        return {"frame": frame, "fps": fps, "frame_index": frame_index}

    cap.release()
    cap = _open()
    if cap is None:
        return None

    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, accident_time) * 1000.0)
    ret, frame = cap.read()
    if ret and frame is not None:
        cap.release()
        return {"frame": frame, "fps": fps, "frame_index": frame_index}

    for delta in range(1, 8):
        for candidate in (frame_index - delta, frame_index + delta):
            if candidate < 0 or candidate >= max(1, total_frames):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(candidate))
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                return {"frame": frame, "fps": fps, "frame_index": int(candidate)}

    if total_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
        ret, frame = cap.read()
        if ret and frame is not None:
            cap.release()
            return {"frame": frame, "fps": fps, "frame_index": max(0, total_frames - 1)}

    cap.release()
    return None


def get_system_prompt_for_stage(stage: str) -> str:
    common = "You are a precise traffic accident analysis assistant. Your response must be a single valid JSON object. No conversational text, no reasoning outside the JSON, no markdown outside the JSON."
    if stage == "time":
        return common + " Your task is to predict the exact second ('accident_time') the accident starts."
    elif stage == "location":
        return common + " Your task is to predict the location ('accident_location') of the accident."
    elif stage == "type":
        return common + " Your task is to predict the accident type ('accident_type')."
    return common


def call_qwen_for_media_single(
    model,
    processor,
    media_type: str,
    media_path: str,
    prompt: str,
    rel_path: str,
    stage: str,
    raw_log_path: str,
    max_retries: int = 1,
) -> Optional[Dict[str, Any]]:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            # Stronger instructions and prefix injection
            prefix = "```json\n{\n  \"reasoning\": \""
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": get_system_prompt_for_stage(stage)},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": media_type,
                            media_type: media_path,
                            "max_pixels": 150528,
                            **({"fps": 1.0} if media_type == "video" else {})
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Standard Prefix Injection pattern: manually append prefix after generation prompt
            # but ensure it's not duplicated if the template somehow added it (unlikely here)
            if not text.endswith(prefix):
                 text += prefix

            image_inputs, video_inputs = process_vision_info(messages)
            
            processed = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            processed = move_inputs_to_device(processed, get_model_input_device(model))

            streamer = TextIteratorStreamer(
                processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = dict(
                **processed,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                # temperature=TEMPERATURE,  # Omitted for do_sample=False
                # top_p=TOP_P,
                streamer=streamer,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            print(f"  -> [{stage}] model output: ", end="", flush=True)
            print(prefix, end="", flush=True)
            collected_text = prefix
            for new_text in streamer:
                print(new_text, end="", flush=True)
                collected_text += new_text

            thread.join()
            print()

            append_raw_log(
                raw_log_path,
                {
                    "path": rel_path,
                    "stage": stage,
                    "attempt": attempt,
                    "raw_output": collected_text,
                },
            )

            parsed = extract_first_json_object(collected_text)
            if parsed is not None:
                print(f"  -> [{stage}] JSON parse succeeded.")
                return parsed

            last_error = f"JSON parse failed on attempt {attempt}. Raw output: {collected_text[:500]}"
            print(f"  -> [{stage}] JSON parse failed on attempt {attempt}.")
        except Exception as exc:
            last_error = str(exc)
            import traceback
            traceback.print_exc()
            print(f"  -> [{stage}] request attempt {attempt} failed: {last_error}")

        time.sleep(1.0)

    print(f"  -> [{stage}] Qwen request failed: {last_error}")
    return None


def call_qwen_for_multi_image(
    model,
    processor,
    image_paths: List[str],
    prompt: str,
    rel_path: str,
    stage: str,
    raw_log_path: str,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """Handles multiple images in a single prompt."""
    image_contents = []
    for path in image_paths:
        image_contents.append({
            "type": "image",
            "image": path,
            "max_pixels": 150528,
        })
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Respond with JSON only. No reasoning. No explanation. /no_think"}],
        },
        {
            "role": "user",
            "content": image_contents + [{"type": "text", "text": prompt}],
        },
    ]

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            image_inputs, video_inputs = process_vision_info(messages)
            processed = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            processed = move_inputs_to_device(processed, get_model_input_device(model))

            streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                **processed,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                streamer=streamer,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            print(f"  -> [{stage}] model output: ", end="", flush=True)
            collected_text = ""
            for new_text in streamer:
                print(new_text, end="", flush=True)
                collected_text += new_text
            thread.join()
            print()

            append_raw_log(raw_log_path, {"path": rel_path, "stage": stage, "attempt": attempt, "raw_output": collected_text})
            parsed = extract_first_json_object(collected_text)
            if parsed is not None:
                return parsed
            if attempt < max_retries:
                time.sleep(1)
        except Exception as e:
            last_error = e
            print(f"  -> [{stage}] error: {e}")
            if attempt < max_retries:
                time.sleep(1)
    return None


def refine_accident_time(
    model,
    processor,
    abs_video_path: str,
    initial_time: float,
    time_feature_signals: Dict[str, Any],
    meta: Dict[str, Any],
    rel_path: str,
    raw_log_path: str,
) -> float:
    """Refines the accident time by comparing the VLM guess with top tracking candidates."""
    duration = safe_float(meta.get("duration", 300.0))
    video_stem = os.path.splitext(os.path.basename(abs_video_path))[0]
    
    # Collect candidate times
    # 1. Initial VLM guess
    candidates = [{"time": initial_time, "source": "vlm_initial"}]
    
    # 2. Tracking-based candidates (top pairs/singles)
    for row in time_feature_signals.get("top_pairs", [])[:2]:
        t = safe_float(row.get("time_sec"))
        if t is not None:
            candidates.append({"time": t, "source": f"tracking_pair_{row.get('pair')}"})
            
    for row in time_feature_signals.get("top_singles", [])[:1]:
        t = safe_float(row.get("time_sec"))
        if t is not None:
            candidates.append({"time": t, "source": f"tracking_single_{row.get('track_id')}"})

    # Deduplicate candidates (within 0.2s)
    final_candidates = []
    for c in candidates:
        if c["time"] < 0 or c["time"] > duration:
            continue
        is_dup = False
        for fc in final_candidates:
            if abs(fc["time"] - c["time"]) < 0.2:
                is_dup = True
                break
        if not is_dup:
            final_candidates.append(c)
    
    # Limit to 5 frames for the prompt
    final_candidates = final_candidates[:5]
    final_candidates.sort(key=lambda x: x["time"])

    frame_paths = []
    for i, c in enumerate(final_candidates):
        extracted = extract_frame_at_time(abs_video_path, c["time"], meta)
        if extracted:
            f_path = os.path.abspath(os.path.join(FRAME_DIR, f"{video_stem}_refine_{i}_t{c['time']:.3f}.jpg"))
            cv2.imwrite(f_path, extracted["frame"])
            frame_paths.append(f_path)
            c["frame_label"] = f"Frame {i+1}"
        else:
            c["frame_label"] = "n/a"

    if not frame_paths:
        return initial_time

    frame_descriptions = "\n".join([
        f"{c['frame_label']}: {c['time']:.2f}s (Source: {c['source']})"
        for c in final_candidates if c["frame_label"] != "n/a"
    ])

    prompt = f"""You are shown sequential frames from potential accident start moments.
{frame_descriptions}

Task: Identify the EXACT moment when the traffic accident (first physical contact or loss of control) CLEARLY starts.
- Compare these moments carefully.
- The earliest frame where collision is visible or unavoidable is usually the best start time.

Output JSON:
{{
  "reasoning": "brief visual comparison of the candidate frames",
  "best_frame_label": "Frame X",
  "refined_time": float
}}
"""
    print(f"  -> [refine_time] comparing {len(frame_paths)} candidates")
    res = call_qwen_for_multi_image(
        model, processor, frame_paths, prompt, rel_path, "refine_time", raw_log_path
    )
    
    if res and "refined_time" in res:
        new_time = safe_float(res["refined_time"])
        if new_time is not None:
            print(f"  -> [refine_time] selected {new_time:.2f}s")
            return new_time
    
    return initial_time


def build_feature_context_payload(feature_bundle: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": feature_bundle.get("status"),
        "intra_csv_path": feature_bundle.get("intra_path"),
        "inter_csv_path": feature_bundle.get("inter_path"),
        "selected_type_track_ids": [],
        "type_track_match_mode": None,
        "feature_error": feature_bundle.get("error"),
        "feature_warning": feature_bundle.get("warning"),
    }


def build_failure_result(
    stage: str,
    error: str,
    video_path: str,
    raw_log_path: str,
    metadata: Optional[Dict[str, str]] = None,
    frame_path: Optional[str] = None,
    frame_index: Optional[int] = None,
    submission_row: Optional[Dict[str, Any]] = None,
    feature_context: Optional[Dict[str, Any]] = None,
    type_object_context_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "status": "failed",
        "failed_stage": stage,
        "error": error,
        "video_path": video_path,
        "metadata_csv_path": METADATA_CSV_PATH,
        "metadata": metadata,
        "frame_path": frame_path,
        "frame_index": frame_index,
        "raw_log_path": raw_log_path,
        "submission_row": submission_row,
        "feature_context": feature_context,
    }
    if type_object_context_summary is not None:
        payload["type_object_context_summary"] = type_object_context_summary
    return payload


def build_submission_row(
    meta: Dict[str, str],
    result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        "path": meta["path"],
        "accident_time": 0.0,
        "center_x": 0.5,
        "center_y": 0.5,
        "type": "single",
    }
    if result is None:
        return row

    row["accident_time"] = round(float(result["accident_time"]), 3)
    row["center_x"] = round(float(result["center_x"]), 3)
    row["center_y"] = round(float(result["center_y"]), 3)
    row["type"] = str(result["type"])
    return row


def build_type_object_context_summary(type_context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if type_context is None:
        return None
    return {
        "selected_track_ids": type_context["selection"].get("selected_track_ids", []),
        "match_mode": type_context["selection"].get("match_mode"),
        "matched_frame_idx": type_context["selection"].get("matched_frame_idx"),
        "track_ids_in_context": [t.get("track_id") for t in type_context.get("tracks", [])],
        "pair_ids_in_context": [p.get("pair") for p in type_context.get("pairs", [])],
    }


def _run_single_video_inference_attempt(
    meta: Dict[str, str],
    video_dir: str,
    model,
    processor,
    attempt: int,
    failed_times: List[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    rel_path = meta["path"]
    video_name = os.path.basename(rel_path)
    video_stem = os.path.splitext(video_name)[0]
    abs_video_path = os.path.abspath(os.path.join(video_dir, video_name))
    raw_log_path = os.path.abspath(os.path.join(RAW_LOG_DIR, f"{video_stem}.jsonl"))
    result_json_path = os.path.abspath(os.path.join(RESULT_JSON_DIR, f"{video_stem}.json"))

    if os.path.exists(raw_log_path):
        os.remove(raw_log_path)

    current_row = build_submission_row(meta)
    feature_bundle = load_feature_bundle(video_stem)
    feature_index = prepare_feature_indices(feature_bundle, meta)
    feature_context = build_feature_context_payload(feature_bundle)
    time_feature_signals = build_time_feature_signals(feature_index, meta)
    time_feature_text = format_time_feature_text(feature_index, time_feature_signals)
    type_object_context: Optional[Dict[str, Any]] = None

    def finalize_failure(
        stage: str,
        error: str,
        frame_path: Optional[str] = None,
        frame_index: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        payload = build_failure_result(
            stage=stage,
            error=error,
            video_path=abs_video_path,
            raw_log_path=raw_log_path,
            metadata=meta,
            frame_path=frame_path,
            frame_index=frame_index,
            submission_row=current_row,
            feature_context=feature_context,
            type_object_context_summary=build_type_object_context_summary(type_object_context),
        )
        # To support retry logic tracking
        if "accident_time" not in payload:
            payload["accident_time"] = current_row.get("accident_time", 0.0)
            
        write_json(result_json_path, payload)
        print(f"  -> [{stage}] failed. Saved result to: {result_json_path}")
        return payload, current_row, "failed"

    try:
        if not os.path.exists(abs_video_path):
            return finalize_failure("video", f"Video file not found: {abs_video_path}")

        print("  -> [time] predicting accident time")
        log_feature_prompt_input(raw_log_path, rel_path, "time", feature_index, time_feature_text)
        raw_time = call_qwen_for_media_single(
            model=model,
            processor=processor,
            media_type="video",
            media_path=abs_video_path,
            prompt=build_time_prompt(meta, time_feature_text, attempt, failed_times),
            rel_path=rel_path,
            stage="time",
            raw_log_path=raw_log_path,
        )
        if raw_time is None:
            return finalize_failure("time", "Failed to get valid JSON response for accident_time.")

        accident_time = validate_time_prediction(raw_time, meta)
        # Update current_row early so finalize_failure can capture it for retry tracking
        if accident_time is not None:
            current_row["accident_time"] = round(float(accident_time), 3)

        video_duration = float(meta.get("duration", 0.0))
        if accident_time is None or float(accident_time) <= 0.0 or (float(accident_time) <= 1.0 and video_duration >= 3.0):
            return finalize_failure("time", f"Invalid, zero, or too early time prediction ({accident_time if accident_time is not None else 0.0:.2f}s) for {video_duration:.1f}s video.")
            
        if failed_times is not None:
            # Rejection threshold: exact match or very close (e.g. within 0.1 seconds)
            for ft in failed_times:
                if abs(float(accident_time) - ft) <= 0.1:
                    return finalize_failure("time", f"Model repeated previously failed time: {accident_time} (close to {ft}). Forcing retry.")

        print(f"  -> [time] parsed accident_time={accident_time:.2f}")

        print("  -> [frame] extracting best frame for location")
        extracted = extract_frame_at_time(abs_video_path, accident_time, meta)
        if extracted is None:
            return finalize_failure("frame", "Failed to extract frame at predicted accident_time.")

        frame = extracted["frame"]
        frame_index = extracted["frame_index"]
        frame_filename = f"{video_stem}_frame_t{accident_time:.3f}.jpg"
        frame_path = os.path.abspath(os.path.join(FRAME_DIR, frame_filename))

        if not cv2.imwrite(frame_path, frame):
            return finalize_failure(
                "frame",
                f"Failed to write extracted frame image: {frame_path}",
                frame_index=frame_index,
            )

        print(f"  -> [frame] saved: {frame_path}")
        print(f"  -> [frame] frame_index={frame_index}")

        location_feature_context = build_location_feature_context(feature_index, frame_index)
        location_feature_text = format_location_feature_text(feature_index, location_feature_context)

        print("  -> [location] predicting collision location")
        log_feature_prompt_input(
            raw_log_path,
            rel_path,
            "location",
            feature_index,
            location_feature_text,
            extra={"frame_index": frame_index},
        )
        raw_location = call_qwen_for_media_single(
            model=model,
            processor=processor,
            media_type="image",
            media_path=frame_path,
            prompt=build_location_prompt(meta, accident_time, location_feature_text),
            rel_path=rel_path,
            stage="location",
            raw_log_path=raw_log_path,
        )
        if raw_location is None:
            return finalize_failure(
                "location",
                "Failed to get valid JSON response for location.",
                frame_path=frame_path,
                frame_index=frame_index,
            )

        location = validate_location_prediction(raw_location)
        if location is None:
            return finalize_failure(
                "location",
                f"Invalid location prediction schema: {raw_location}",
                frame_path=frame_path,
                frame_index=frame_index,
            )

        center_x = location["center_x"]
        center_y = location["center_y"]
        current_row["center_x"] = round(float(center_x), 3)
        current_row["center_y"] = round(float(center_y), 3)
        print(f"  -> [location] center_x={center_x:.4f}, center_y={center_y:.4f}")

        type_object_context = build_type_object_context(feature_index, meta, frame_index, center_x, center_y)
        feature_context["selected_type_track_ids"] = type_object_context["selection"].get("selected_track_ids", [])
        feature_context["type_track_match_mode"] = type_object_context["selection"].get("match_mode")
        type_feature_text = format_type_object_feature_text(feature_index, type_object_context)

        print("  -> [type] predicting accident type")
        log_feature_prompt_input(
            raw_log_path,
            rel_path,
            "type",
            feature_index,
            type_feature_text,
            extra=build_type_object_context_summary(type_object_context),
        )
        raw_type = call_qwen_for_media_single(
            model=model,
            processor=processor,
            media_type="image",
            media_path=frame_path,
            prompt=build_type_prompt(meta, accident_time, type_feature_text),
            rel_path=rel_path,
            stage="type",
            raw_log_path=raw_log_path,
        )
        if raw_type is None:
            return finalize_failure(
                "type",
                "Failed to get valid JSON response for type.",
                frame_path=frame_path,
                frame_index=frame_index,
            )

        accident_type = validate_type_prediction(raw_type, type_object_context)
        if accident_type is None:
            return finalize_failure(
                "type",
                f"Invalid type prediction schema: {raw_type}",
                frame_path=frame_path,
                frame_index=frame_index,
            )

        current_row["type"] = str(accident_type)
        result_data = {
            "accident_time": accident_time,
            "center_x": center_x,
            "center_y": center_y,
            "type": accident_type,
        }
        submission_row = current_row
        payload = {
            "status": "ok",
            "video_path": abs_video_path,
            "metadata_csv_path": METADATA_CSV_PATH,
            "metadata": meta,
            "result": result_data,
            "frame_index": frame_index,
            "frame_path": frame_path,
            "raw_log_path": raw_log_path,
            "submission_row": submission_row,
            "feature_context": feature_context,
            "type_object_context_summary": build_type_object_context_summary(type_object_context),
        }
        write_json(result_json_path, payload)
        print(f"  -> [done] saved result to: {result_json_path}")
        return payload, submission_row, "ok"
    except Exception as exc:
        return finalize_failure("unexpected", str(exc))

def run_single_video_inference(
    meta: Dict[str, str],
    video_dir: str,
    model,
    processor,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    payload, row, status = None, None, "failed"
    failed_times = []
    for attempt in range(MAX_VIDEO_RETRIES):
        if attempt > 0:
            print(f"\n  -> [retry {attempt}] restarting from time prediction... (Previous failed times: {failed_times})")
        
        payload, row, status = _run_single_video_inference_attempt(
            meta, video_dir, model, processor, attempt, failed_times
        )
        
        if status == "ok":
            return payload, row, status
        
        # Track failed accident_time if available to avoid repeating in next attempt
        if payload and "accident_time" in payload:
            t = payload["accident_time"]
            if t not in failed_times:
                failed_times.append(t)
        elif row and "accident_time" in row:
            t = row["accident_time"]
            if t not in failed_times:
                failed_times.append(t)

        print(f"  -> Attempt {attempt + 1}/{MAX_VIDEO_RETRIES} failed. Re-evaluating...")
    return payload, row, status


def build_run_summary(
    video_dir: str,
    total_rows: int,
    success_count: int,
    failure_count: int,
    failed_videos: List[Dict[str, Any]],
    started_at: str,
    finished_at: str,
    global_error: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model_name": MODEL_NAME,
        "video_dir": video_dir,
        "metadata_csv_path": METADATA_CSV_PATH,
        "output_dir": OUTPUT_DIR,
        "total_rows": total_rows,
        "success_count": success_count,
        "failure_count": failure_count,
        "failed_videos": failed_videos,
        "submission_csv_path": SUBMISSION_CSV_PATH,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    if global_error is not None:
        payload["global_error"] = global_error
    payload["test_mode"] = TEST_MODE
    if TEST_MODE:
        payload["test_limit"] = TEST_LIMIT
    return payload


def write_global_failure_outputs(
    metadata_rows: List[Dict[str, str]],
    video_dir: str,
    error: str,
    started_at: str,
) -> None:
    rows = [build_submission_row(meta) for meta in metadata_rows]
    write_submission_csv(rows)
    summary = build_run_summary(
        video_dir=video_dir,
        total_rows=len(metadata_rows),
        success_count=0,
        failure_count=len(metadata_rows),
        failed_videos=[],
        started_at=started_at,
        finished_at=now_utc_iso(),
        global_error=error,
    )
    write_json(RUN_SUMMARY_PATH, summary)
    print(error)
    print(f"Saved default submission to: {SUBMISSION_CSV_PATH}")
    print(f"Saved run summary to: {RUN_SUMMARY_PATH}")


def main() -> None:
    started_at = now_utc_iso()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULT_JSON_DIR, exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(RAW_LOG_DIR, exist_ok=True)

    metadata_rows = load_all_metadata(METADATA_CSV_PATH)
    effective_video_dir = resolve_video_dir()
    if TEST_MODE:
        metadata_rows = filter_metadata_for_test(metadata_rows, effective_video_dir, TEST_LIMIT)

    print(f"Loading model: {MODEL_NAME}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Video dir: {effective_video_dir}")
    print(f"Metadata CSV: {METADATA_CSV_PATH}")
    print(f"Feature dir: {FEATURE_ROOT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Total rows: {len(metadata_rows)}")
    if TEST_MODE:
        print(f"Test mode: enabled (limit={TEST_LIMIT if TEST_LIMIT > 0 else 'all matches'})")

    if not metadata_rows:
        write_submission_csv([])
        summary = build_run_summary(
            video_dir=effective_video_dir,
            total_rows=0,
            success_count=0,
            failure_count=0,
            failed_videos=[],
            started_at=started_at,
            finished_at=now_utc_iso(),
            global_error="No metadata rows matched the configured input selection.",
        )
        write_json(RUN_SUMMARY_PATH, summary)
        print("No metadata rows matched the configured input selection.")
        print(f"Saved empty submission CSV to: {SUBMISSION_CSV_PATH}")
        print(f"Saved run summary to: {RUN_SUMMARY_PATH}")
        return

    compatibility_error = ensure_runtime_compatibility()
    if compatibility_error is not None:
        write_global_failure_outputs(metadata_rows, effective_video_dir, compatibility_error, started_at)
        return

    try:
        model_load_config, load_notes = build_model_load_config()
        for note in load_notes:
            print(f"Model load config: {note}")

        processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            **model_load_config,
        )
        print(f"Model input device: {get_model_input_device(model)}")
    except Exception as exc:
        write_global_failure_outputs(
            metadata_rows,
            effective_video_dir,
            f"Model loading failed: {exc}",
            started_at,
        )
        return

    submission_rows: List[Dict[str, Any]] = []
    failed_videos: List[Dict[str, Any]] = []
    success_count = 0
    failure_count = 0

    for idx, meta in enumerate(metadata_rows, start=1):
        rel_path = meta["path"]
        video_name = os.path.basename(rel_path)
        video_stem = os.path.splitext(video_name)[0]
        result_json_path = os.path.abspath(os.path.join(RESULT_JSON_DIR, f"{video_stem}.json"))

        if os.path.exists(result_json_path):
            print(f"\n[{idx}/{len(metadata_rows)}] Skipping (already exists): {rel_path}")
            try:
                with open(result_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "submission_row" in data:
                        submission_rows.append(data["submission_row"])
                        if data.get("status") == "ok":
                            success_count += 1
                        else:
                            failure_count += 1
                    else:
                        row = build_submission_row(meta)
                        submission_rows.append(row)
                        failure_count += 1
            except Exception as e:
                print(f"  -> Warning: failed to parse existing JSON {result_json_path}: {e}")
                row = build_submission_row(meta)
                submission_rows.append(row)
                failure_count += 1
            continue

        print(f"\n[{idx}/{len(metadata_rows)}] Processing: {meta['path']}")
        payload, submission_row, status = run_single_video_inference(
            meta,
            effective_video_dir,
            model,
            processor,
        )
        submission_rows.append(submission_row)

        if status == "ok":
            success_count += 1
        else:
            failure_count += 1
            failed_videos.append(
                {
                    "path": meta["path"],
                    "failed_stage": payload.get("failed_stage", "unknown"),
                    "error": payload.get("error", ""),
                }
            )

    write_submission_csv(submission_rows)

    summary = build_run_summary(
        video_dir=effective_video_dir,
        total_rows=len(metadata_rows),
        success_count=success_count,
        failure_count=failure_count,
        failed_videos=failed_videos,
        started_at=started_at,
        finished_at=now_utc_iso(),
    )
    write_json(RUN_SUMMARY_PATH, summary)

    print("\nRun finished.")
    print(f"Total rows: {len(metadata_rows)}")
    print(f"Success count: {success_count}")
    print(f"Failure count: {failure_count}")
    print(f"Saved submission CSV to: {SUBMISSION_CSV_PATH}")
    print(f"Saved run summary to: {RUN_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
