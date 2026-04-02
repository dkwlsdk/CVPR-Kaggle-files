import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    FIX_CLIP_SEC,
    RUN_LOG,
    VALID_TYPES,
    ensure_dirs,
    get_stage_dir,
    get_video_output_dir,
)


def setup_logging() -> logging.Logger:
    ensure_dirs()
    logger = logging.getLogger("accident_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(RUN_LOG, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


LOGGER = setup_logging()


def read_metadata(csv_path: Path) -> Iterable[Dict[str, str]]:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            if "scene_layout" not in row and "scene_layoutm" in row:
                row["scene_layout"] = row["scene_layoutm"]
            yield row


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def strip_thinking_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = strip_thinking_text(text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    brace_positions = [i for i, ch in enumerate(text) if ch == "{"]
    for start in brace_positions:
        depth = 0
        for end in range(start, len(text)):
            ch = text[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:end + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
    return None


def get_video_info(video_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = (total_frames / fps) if fps > 0 else 0.0
    cap.release()
    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration": duration,
    }


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def validate_time_prediction(obj: Dict[str, Any], duration: float) -> Optional[float]:
    try:
        t = float(obj["accident_time"])
    except Exception:
        return None
    return clamp(t, 0.0, max(0.0, duration))


def validate_type_prediction(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        t = str(obj["type"]).strip()
    except Exception:
        return None
    if t not in VALID_TYPES:
        return None
    reason = str(obj.get("reason", "")).strip()
    raw_single = obj.get("is_single", t == "single")
    if isinstance(raw_single, str):
        is_single = raw_single.strip().lower() == "true"
    else:
        is_single = bool(raw_single)
    if t == "single":
        is_single = True
    return {
        "type": t,
        "reason": reason[:300],
        "is_single": is_single,
    }


def validate_location_prediction(obj: Dict[str, Any]) -> Optional[Dict[str, float]]:
    try:
        x = float(obj["center_x"])
        y = float(obj["center_y"])
    except Exception:
        return None
    out = {
        "center_x": clamp(x, 0.0, 1.0),
        "center_y": clamp(y, 0.0, 1.0),
    }
    if "box_mode" in obj:
        out["box_mode"] = str(obj["box_mode"]).strip()[:50]
    if "reason" in obj:
        out["reason"] = str(obj["reason"]).strip()[:300]
    return out


def frame_at_time(video_path: str, sec: float) -> Tuple[np.ndarray, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid fps: {video_path}")
    idx = int(sec * fps)
    idx = max(0, min(idx, max(0, total - 1)))

    for candidate in [idx, max(0, idx - 1), min(total - 1, idx + 1)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
        ok, frame = cap.read()
        if ok and frame is not None:
            cap.release()
            return frame, candidate, fps

    cap.release()
    raise RuntimeError(f"Failed to read frame at {sec:.3f}s from {video_path}")


def save_frame_image(video_path: str, out_path: Path, sec: float) -> Dict[str, Any]:
    frame, idx, fps = frame_at_time(video_path, sec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    return {"frame_path": str(out_path), "frame_index": idx, "fps": fps}


def save_clip_centered(video_path: str, out_path: Path, center_sec: float, clip_sec: float = FIX_CLIP_SEC) -> Dict[str, Any]:
    info = get_video_info(video_path)
    fps = info["fps"]
    total = info["total_frames"]
    if fps <= 0 or total <= 0:
        raise RuntimeError(f"Bad video info: {video_path}")

    half = clip_sec / 2.0
    start_sec = clamp(center_sec - half, 0.0, max(0.0, info["duration"]))
    end_sec = clamp(center_sec + half, 0.0, max(0.0, info["duration"]))
    start_idx = int(start_sec * fps)
    end_idx = int(end_sec * fps)
    if end_idx <= start_idx:
        end_idx = min(total - 1, start_idx + max(1, int(fps * clip_sec)))

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (info["width"], info["height"]))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    current = start_idx
    while current <= end_idx:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        current += 1

    cap.release()
    writer.release()
    return {
        "clip_path": str(out_path),
        "start_sec": round(start_sec, 3),
        "end_sec": round(end_sec, 3),
        "start_frame": start_idx,
        "end_frame": end_idx,
    }


def get_stage_paths(rel_path: str) -> Dict[str, Path]:
    video_dir = get_video_output_dir(rel_path)
    stage1_dir = get_stage_dir(rel_path, "stage1")
    stage2_dir = get_stage_dir(rel_path, "stage2")
    stage3_dir = get_stage_dir(rel_path, "stage3")
    stage4_dir = get_stage_dir(rel_path, "stage4")
    return {
        "video_dir": video_dir,
        "stage1_dir": stage1_dir,
        "stage2_dir": stage2_dir,
        "stage3_dir": stage3_dir,
        "stage4_dir": stage4_dir,
        "final_csv": video_dir / "final_prediction.csv",
        "final_json": video_dir / "final_prediction.json",
        "stage1_scores_csv": stage1_dir / "scores.csv",
        "stage1_summary_json": stage1_dir / "summary.json",
        "stage1_candidates_json": stage1_dir / "candidate_times.json",
        "stage2_fixed_json": stage2_dir / "fixed_time.json",
        "stage2_frame_jpg": stage2_dir / "key_frame.jpg",
        "stage2_clip_mp4": stage2_dir / "fixed_clip.mp4",
        "stage3_type_json": stage3_dir / "type.json",
        "stage4_location_json": stage4_dir / "location.json",
    }

import re
from typing import Optional, Dict, Any


def extract_time_fallback(text: str, duration: Optional[float] = None) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = strip_thinking_text(text)
    text = text.replace("```json", "").replace("```", "").strip()

    patterns = [
        r'"accident_time"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        r'\baccident[_ ]?time\b\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)',
        r'\bcollision\b.*?\b(?:at|around)\b\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|secs|seconds)?',
        r'\bimpact\b.*?\b(?:at|around)\b\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|secs|seconds)?',
        r'\bcontact\b.*?\b(?:at|around)\b\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|secs|seconds)?',
        r'\bstarts?\b.*?\b(?:at|around)\b\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|secs|seconds)?',
    ]

    found = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE | re.DOTALL):
            try:
                val = float(m.group(1))
            except Exception:
                continue
            if duration is not None and not (0.0 <= val <= duration + 1e-6):
                continue
            found.append(val)

    if not found:
        return None

    t = found[-1]
    return {
        "accident_time": t,
        "source": "fallback_text_parse",
    }