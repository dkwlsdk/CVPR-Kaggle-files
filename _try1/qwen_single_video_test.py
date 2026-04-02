import csv
import json
import os
import time
from threading import Thread
from typing import Any, Dict, Optional

import cv2
import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer

from qwen_single_video_helpers import (
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    build_location_prompt,
    build_time_prompt,
    build_type_prompt,
    extract_first_json_object,
    extract_frame_at_time,
    move_inputs_to_device,
    normalize_metadata,
    validate_location_prediction,
    validate_time_prediction,
    validate_type_prediction,
)


MODEL_NAME = "Qwen/Qwen3.5-9B"
VIDEO_PATH = "/root/Desktop/workspace/ja/CVPR-Kaggle-files/데이터셋/test_videos/-7-vQ4obVwQ_00.mp4"
METADATA_CSV_PATH = "/root/Desktop/workspace/ja/CVPR-Kaggle-files/데이터셋/test_metadata.csv"
OUTPUT_DIR = "/root/Desktop/workspace/ja/qwen_single_video_test_output"
RAW_LOG_PATH = os.path.join(OUTPUT_DIR, "-7-vQ4obVwQ_00.jsonl")
RESULT_JSON_PATH = os.path.join(OUTPUT_DIR, "-7-vQ4obVwQ_00.json")

MIN_PYTHON = (3, 10)
MIN_TRANSFORMERS = (5, 0)


def append_raw_log(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_result(payload: Dict[str, Any]) -> None:
    with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_failure_result(
    stage: str,
    error: str,
    metadata: Optional[Dict[str, str]] = None,
    frame_path: Optional[str] = None,
    frame_index: Optional[int] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": "failed",
        "failed_stage": stage,
        "error": error,
        "video_path": VIDEO_PATH,
        "metadata_csv_path": METADATA_CSV_PATH,
        "raw_log_path": RAW_LOG_PATH,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    if frame_path is not None:
        payload["frame_path"] = frame_path
    if frame_index is not None:
        payload["frame_index"] = frame_index
    return payload


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
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required for Qwen/Qwen3.5-9B. "
            f"Current Python: {os.sys.version.split()[0]}"
        )

    tf_version = parse_version_tuple(transformers.__version__)
    if tf_version < MIN_TRANSFORMERS:
        return (
            f"transformers {MIN_TRANSFORMERS[0]}.{MIN_TRANSFORMERS[1]}+ is required "
            f"for Qwen/Qwen3.5-9B. Current transformers: {transformers.__version__}"
        )

    return None


def load_metadata_for_video(csv_path: str, video_path: str) -> Dict[str, str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    target_name = os.path.basename(video_path)

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized = normalize_metadata(row)
            candidate_name = os.path.basename(normalized.get("path", ""))
            if candidate_name == target_name:
                return normalized

    raise ValueError(f"No metadata row matched video basename: {target_name}")


def call_qwen_for_media_single(
    model,
    processor,
    media_type: str,
    media_path: str,
    prompt: str,
    rel_path: str,
    stage: str,
    raw_log_path: str,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Respond with JSON only. No reasoning. No explanation. /no_think",
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
            processed = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
            processed = move_inputs_to_device(processed, model.device)

            streamer = TextIteratorStreamer(
                processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = dict(
                **processed,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
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

            last_error = (
                f"JSON parse failed on attempt {attempt}. "
                f"Raw output: {collected_text[:500]}"
            )
            print(f"  -> [{stage}] JSON parse failed on attempt {attempt}.")
        except Exception as exc:
            last_error = str(exc)
            print(f"  -> [{stage}] request attempt {attempt} failed: {last_error}")

        time.sleep(1.0)

    print(f"  -> [{stage}] Qwen request failed: {last_error}")
    return None


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(RAW_LOG_PATH):
        os.remove(RAW_LOG_PATH)

    compatibility_error = ensure_runtime_compatibility()
    if compatibility_error is not None:
        failure = build_failure_result("environment", compatibility_error)
        write_result(failure)
        print(compatibility_error)
        print("Recommended interpreter: /opt/conda/envs/qwen35/bin/python")
        print(f"Saved failure result to: {RESULT_JSON_PATH}")
        return

    if not os.path.exists(VIDEO_PATH):
        failure = build_failure_result("video", f"Video file not found: {VIDEO_PATH}")
        write_result(failure)
        raise FileNotFoundError(failure["error"])

    try:
        metadata = load_metadata_for_video(METADATA_CSV_PATH, VIDEO_PATH)
    except Exception as exc:
        failure = build_failure_result("metadata", str(exc))
        write_result(failure)
        raise

    print(f"Loading model: {MODEL_NAME}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Video path: {VIDEO_PATH}")
    print(f"Metadata CSV: {METADATA_CSV_PATH}")
    print("Resolved metadata:")
    for key in (
        "path",
        "region",
        "scene_layout",
        "weather",
        "day_time",
        "quality",
        "no_frames",
        "duration",
        "height",
        "width",
    ):
        print(f"  - {key}: {metadata.get(key, '')}")

    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
    except Exception as exc:
        failure = build_failure_result("model", str(exc), metadata=metadata)
        write_result(failure)
        print(f"\nModel loading failed: {exc}")
        print(f"Saved failure result to: {RESULT_JSON_PATH}")
        return

    rel_path = metadata.get("path", os.path.basename(VIDEO_PATH))
    abs_video_path = os.path.abspath(VIDEO_PATH)

    print("\n[1/4] Predicting accident time")
    raw_time = call_qwen_for_media_single(
        model=model,
        processor=processor,
        media_type="video",
        media_path=abs_video_path,
        prompt=build_time_prompt(metadata),
        rel_path=rel_path,
        stage="time",
        raw_log_path=RAW_LOG_PATH,
    )
    if raw_time is None:
        failure = build_failure_result(
            "time",
            "Failed to get valid JSON response for accident_time.",
            metadata=metadata,
        )
        write_result(failure)
        return

    accident_time = validate_time_prediction(raw_time, metadata)
    if accident_time is None:
        failure = build_failure_result(
            "time",
            f"Invalid time prediction schema: {raw_time}",
            metadata=metadata,
        )
        write_result(failure)
        return

    print(f"  -> [time] parsed accident_time={accident_time:.4f}")

    print("\n[2/4] Extracting frame")
    extracted = extract_frame_at_time(abs_video_path, accident_time, metadata)
    if extracted is None:
        failure = build_failure_result(
            "frame",
            "Failed to extract frame at predicted accident_time.",
            metadata=metadata,
        )
        write_result(failure)
        return

    frame = extracted["frame"]
    frame_index = extracted["frame_index"]
    frame_stem = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    frame_filename = f"{frame_stem}_frame_t{accident_time:.3f}.jpg"
    frame_path = os.path.join(OUTPUT_DIR, frame_filename)

    if not cv2.imwrite(frame_path, frame):
        failure = build_failure_result(
            "frame",
            f"Failed to write extracted frame image: {frame_path}",
            metadata=metadata,
            frame_index=frame_index,
        )
        write_result(failure)
        return

    abs_frame_path = os.path.abspath(frame_path)
    print(f"  -> [frame] saved: {abs_frame_path}")
    print(f"  -> [frame] frame_index={frame_index}")

    print("\n[3/4] Predicting collision location")
    raw_location = call_qwen_for_media_single(
        model=model,
        processor=processor,
        media_type="image",
        media_path=abs_frame_path,
        prompt=build_location_prompt(metadata, accident_time),
        rel_path=rel_path,
        stage="location",
        raw_log_path=RAW_LOG_PATH,
    )
    if raw_location is None:
        failure = build_failure_result(
            "location",
            "Failed to get valid JSON response for location.",
            metadata=metadata,
            frame_path=abs_frame_path,
            frame_index=frame_index,
        )
        write_result(failure)
        return

    location = validate_location_prediction(raw_location)
    if location is None:
        failure = build_failure_result(
            "location",
            f"Invalid location prediction schema: {raw_location}",
            metadata=metadata,
            frame_path=abs_frame_path,
            frame_index=frame_index,
        )
        write_result(failure)
        return

    center_x = location["center_x"]
    center_y = location["center_y"]
    print(f"  -> [location] center_x={center_x:.4f}, center_y={center_y:.4f}")

    print("\n[4/4] Predicting accident type")
    raw_type = call_qwen_for_media_single(
        model=model,
        processor=processor,
        media_type="image",
        media_path=abs_frame_path,
        prompt=build_type_prompt(metadata, accident_time),
        rel_path=rel_path,
        stage="type",
        raw_log_path=RAW_LOG_PATH,
    )
    if raw_type is None:
        failure = build_failure_result(
            "type",
            "Failed to get valid JSON response for type.",
            metadata=metadata,
            frame_path=abs_frame_path,
            frame_index=frame_index,
        )
        write_result(failure)
        return

    accident_type = validate_type_prediction(raw_type)
    if accident_type is None:
        failure = build_failure_result(
            "type",
            f"Invalid type prediction schema: {raw_type}",
            metadata=metadata,
            frame_path=abs_frame_path,
            frame_index=frame_index,
        )
        write_result(failure)
        return

    result = {
        "video_path": VIDEO_PATH,
        "metadata_csv_path": METADATA_CSV_PATH,
        "metadata": metadata,
        "result": {
            "accident_time": accident_time,
            "center_x": center_x,
            "center_y": center_y,
            "type": accident_type,
        },
        "frame_index": frame_index,
        "frame_path": abs_frame_path,
        "raw_log_path": RAW_LOG_PATH,
    }
    write_result(result)

    print("\nFinal result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved result to: {RESULT_JSON_PATH}")
    print(f"Saved raw log to: {RAW_LOG_PATH}")


if __name__ == "__main__":
    main()
