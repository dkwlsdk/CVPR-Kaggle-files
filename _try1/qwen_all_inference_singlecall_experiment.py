import csv
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from qwen_all_inference import (
    DEFAULT_MAX_RETRIES,
    METADATA_CSV_PATH,
    MODEL_NAME,
    TEST_LIMIT,
    TEST_MODE,
    append_raw_log,
    build_model_load_config,
    build_submission_row,
    build_time_clip_candidates,
    build_time_feature_signals,
    ensure_runtime_compatibility,
    extract_first_json_object,
    extract_frame_at_time,
    filter_metadata_for_test,
    format_time_feature_text,
    get_model_input_device,
    load_all_metadata,
    load_feature_bundle,
    move_inputs_to_device,
    now_utc_iso,
    prepare_feature_indices,
    resolve_video_dir,
    run_single_video_inference,
    safe_float,
    clamp,
    validate_location_prediction,
    validate_time_prediction,
    validate_type_prediction,
    write_json,
)


EXPERIMENT_MODE = os.environ.get("QWEN_EXPERIMENT_MODE", "compare").strip() or "compare"
EXPERIMENT_OUTPUT_DIR = "/root/Desktop/workspace/ja/qwen_inference_singlecall"
RESULT_JSON_DIR = os.path.join(EXPERIMENT_OUTPUT_DIR, "results_json")
RAW_LOG_DIR = os.path.join(EXPERIMENT_OUTPUT_DIR, "raw_logs")
FRAME_DIR = os.path.join(EXPERIMENT_OUTPUT_DIR, "frames")
COMPARISON_DIR = os.path.join(EXPERIMENT_OUTPUT_DIR, "comparisons")
SUBMISSION_CSV_PATH = os.path.join(EXPERIMENT_OUTPUT_DIR, "submission_dl.csv")
RUN_SUMMARY_PATH = os.path.join(EXPERIMENT_OUTPUT_DIR, "run_summary.json")

CANDIDATE_FRAME_COUNT = 3
SINGLECALL_MAX_NEW_TOKENS = 256
ALLOWED_INPUT_MODES = {"video+frames", "video_only"}
ALLOWED_ASSESSMENTS = {"likely_impact", "possible_context", "unlikely"}
CANDIDATE_TIME_TOLERANCE_SEC = 0.5
ALLOWED_EXPERIMENT_MODES = {"compare", "single_call_only", "baseline_only"}


def write_submission_csv_to(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["path", "accident_time", "center_x", "center_y", "type"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_experiment_output_dirs() -> None:
    os.makedirs(EXPERIMENT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULT_JSON_DIR, exist_ok=True)
    os.makedirs(RAW_LOG_DIR, exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(COMPARISON_DIR, exist_ok=True)


def build_experiment_run_summary(
    video_dir: str,
    total_rows: int,
    experiment_success_count: int,
    experiment_failure_count: int,
    experiment_failed_videos: List[Dict[str, Any]],
    started_at: str,
    finished_at: str,
    baseline_success_count: int = 0,
    baseline_failure_count: int = 0,
    mode: str = EXPERIMENT_MODE,
    global_error: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model_name": MODEL_NAME,
        "mode": mode,
        "video_dir": video_dir,
        "metadata_csv_path": METADATA_CSV_PATH,
        "output_dir": EXPERIMENT_OUTPUT_DIR,
        "submission_csv_path": SUBMISSION_CSV_PATH,
        "comparison_dir": COMPARISON_DIR,
        "total_rows": total_rows,
        "experiment_success_count": experiment_success_count,
        "experiment_failure_count": experiment_failure_count,
        "experiment_failed_videos": experiment_failed_videos,
        "baseline_success_count": baseline_success_count,
        "baseline_failure_count": baseline_failure_count,
        "started_at": started_at,
        "finished_at": finished_at,
        "test_mode": TEST_MODE,
    }
    if TEST_MODE:
        payload["test_limit"] = TEST_LIMIT
    if global_error is not None:
        payload["global_error"] = global_error
    return payload


def coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not number.is_integer():
        return None
    return int(number)


def build_default_candidate_times(meta: Dict[str, str]) -> List[float]:
    duration = safe_float(meta.get("duration"), 0.0) or 0.0
    if duration <= 0:
        return [0.0, 0.0, 0.0]
    return [duration * 0.25, duration * 0.50, duration * 0.75]


def select_candidate_entries(
    time_clip_candidates: List[Dict[str, Any]],
    meta: Dict[str, str],
) -> List[Dict[str, Any]]:
    duration = safe_float(meta.get("duration"))
    selected: List[Dict[str, Any]] = []

    for index, candidate in enumerate(time_clip_candidates[:CANDIDATE_FRAME_COUNT]):
        time_sec = safe_float(candidate.get("time_sec"))
        if time_sec is None:
            continue
        if duration is not None:
            time_sec = clamp(time_sec, 0.0, duration)
        else:
            time_sec = max(time_sec, 0.0)
        selected.append(
            {
                "candidate_index": len(selected),
                "candidate_time": round(float(time_sec), 3),
                "source": candidate.get("source") or "feature_candidate",
                "label": str(candidate.get("label") or ""),
                "score": safe_float(candidate.get("score"), 0.0) or 0.0,
                "clip_start_sec": safe_float(candidate.get("clip_start_sec")),
                "clip_end_sec": safe_float(candidate.get("clip_end_sec")),
                "fallback": False,
            }
        )

    if selected:
        return selected

    fallback_times = build_default_candidate_times(meta)
    for index, time_sec in enumerate(fallback_times):
        if duration is not None:
            time_sec = clamp(time_sec, 0.0, duration)
        else:
            time_sec = max(time_sec, 0.0)
        selected.append(
            {
                "candidate_index": index,
                "candidate_time": round(float(time_sec), 3),
                "source": "duration_fraction",
                "label": f"fraction_{index + 1}",
                "score": 0.0,
                "clip_start_sec": None,
                "clip_end_sec": None,
                "fallback": True,
            }
        )
    return selected


def extract_candidate_frames(
    abs_video_path: str,
    video_stem: str,
    meta: Dict[str, str],
    candidate_entries: List[Dict[str, Any]],
    raw_log_path: str,
) -> List[Dict[str, Any]]:
    extracted_frames: List[Dict[str, Any]] = []
    for candidate in candidate_entries:
        candidate_index = int(candidate["candidate_index"])
        candidate_time = float(candidate["candidate_time"])
        started = time.perf_counter()
        extracted = extract_frame_at_time(abs_video_path, candidate_time, meta)
        elapsed_sec = time.perf_counter() - started

        if extracted is None:
            append_raw_log(
                raw_log_path,
                {
                    "stage": "candidate_frame_extraction",
                    "candidate_index": candidate_index,
                    "candidate_time": candidate_time,
                    "elapsed_sec": elapsed_sec,
                    "status": "failed",
                    "error": "Failed to extract frame at candidate time.",
                },
            )
            continue

        frame_filename = f"{video_stem}_candidate{candidate_index:02d}_t{candidate_time:.3f}.jpg"
        frame_path = os.path.abspath(os.path.join(FRAME_DIR, frame_filename))
        if not cv2.imwrite(frame_path, extracted["frame"]):
            append_raw_log(
                raw_log_path,
                {
                    "stage": "candidate_frame_extraction",
                    "candidate_index": candidate_index,
                    "candidate_time": candidate_time,
                    "elapsed_sec": elapsed_sec,
                    "status": "failed",
                    "error": f"Failed to write frame image: {frame_path}",
                },
            )
            continue

        payload = {
            **candidate,
            "frame_path": frame_path,
            "frame_index": extracted["frame_index"],
            "fps": extracted["fps"],
            "image_attachment_order": len(extracted_frames) + 1,
        }
        extracted_frames.append(payload)
        append_raw_log(
            raw_log_path,
            {
                "stage": "candidate_frame_extraction",
                "candidate_index": candidate_index,
                "candidate_time": candidate_time,
                "frame_path": frame_path,
                "frame_index": extracted["frame_index"],
                "fps": extracted["fps"],
                "elapsed_sec": elapsed_sec,
                "status": "ok",
            },
        )

    return extracted_frames


def build_candidate_feature_notes(
    candidate_entries: List[Dict[str, Any]],
    extracted_frames: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    extracted_by_index = {
        int(frame_info["candidate_index"]): frame_info for frame_info in extracted_frames
    }
    notes: List[Dict[str, Any]] = []
    for candidate in candidate_entries:
        candidate_index = int(candidate["candidate_index"])
        extracted = extracted_by_index.get(candidate_index)
        notes.append(
            {
                "candidate_index": candidate_index,
                "candidate_time": candidate["candidate_time"],
                "source": candidate["source"],
                "label": candidate["label"],
                "score": candidate["score"],
                "clip_start_sec": candidate.get("clip_start_sec"),
                "clip_end_sec": candidate.get("clip_end_sec"),
                "attached_image": extracted is not None,
                "image_attachment_order": extracted.get("image_attachment_order") if extracted else None,
                "frame_path": extracted.get("frame_path") if extracted else None,
                "frame_index": extracted.get("frame_index") if extracted else None,
            }
        )
    return notes


def build_singlecall_prompt(
    metadata: Dict[str, str],
    time_feature_text: str,
    candidate_times: List[float],
    candidate_feature_notes: List[Dict[str, Any]],
) -> str:
    region = metadata.get("region", "")
    scene_layout = metadata.get("scene_layout", "")
    weather = metadata.get("weather", "")
    day_time = metadata.get("day_time", "")
    quality = metadata.get("quality", "")
    duration = metadata.get("duration", "")
    no_frames = metadata.get("no_frames", "")
    height = metadata.get("height", "")
    width = metadata.get("width", "")
    feature_block = time_feature_text.strip() if time_feature_text.strip() else "No auxiliary tracking feature cues were available."

    candidate_lines = []
    for note in candidate_feature_notes:
        attached_text = "yes" if note["attached_image"] else "no"
        attachment_order = note["image_attachment_order"]
        candidate_lines.append(
            "- "
            f"candidate_index={note['candidate_index']}, "
            f"candidate_time={float(note['candidate_time']):.3f}s, "
            f"source={note['source']}, "
            f"label={note['label'] or 'n/a'}, "
            f"score={float(note['score']):.3f}, "
            f"attached_image={attached_text}, "
            f"image_attachment_order_after_video={attachment_order if attachment_order is not None else 'none'}"
        )
    if not candidate_lines:
        candidate_lines.append("- no candidate frames were attached in this request")

    prompt = f"""
You are an expert traffic accident analyst looking at CCTV footage.

You are given:
1. The full video as the primary evidence.
2. Zero or more candidate reference frames after the video as secondary high-resolution evidence.

Candidate reference times derived before this request:
{", ".join(f"{time_value:.3f}s" for time_value in candidate_times) if candidate_times else "none"}

Candidate reference details:
{os.linesep.join(candidate_lines)}

Video metadata:
- region: {region}
- scene_layout: {scene_layout}
- weather: {weather}
- day_time: {day_time}
- quality (before enhancement): {quality}
- duration (seconds): {duration}
- no_frames: {no_frames}
- frame_height: {height}
- frame_width: {width}

Auxiliary tracking feature cues:
{feature_block}

Task:
1. Analyze the full video and identify the earliest true accident start time.
2. Use the candidate reference frames only as secondary high-resolution evidence.
3. Predict the collision center coordinates for the chosen accident moment.
4. Classify the accident type at that chosen impact moment.

Priority rules:
- The video is the primary evidence for accident_time.
- If candidate frames are attached, use the selected candidate frame as secondary evidence for location and type refinement.
- If the video conflicts with candidate frames, trust the video for time and the selected candidate frame for location/type refinement.
- If this request has attached candidate images, set input_mode to "video+frames".
- If this request has no attached candidate images, set input_mode to "video_only" and set selected_candidate_index=null and selected_candidate_time=null.

Definitions of accident types (choose exactly one):
- rear-end: One vehicle crashes into the back of another vehicle traveling in the same direction.
- head-on: Two vehicles traveling in opposite directions collide front-to-front.
- sideswipe: Two vehicles moving in roughly the same direction make side-to-side contact while overlapping partially.
- t-bone: The front of one vehicle crashes into the side of another vehicle, forming a "T" shape.
- single: An accident involving only one vehicle with no other vehicle collision.

Critical output rules:
- Output JSON only.
- No reasoning.
- No analysis.
- No markdown.
- No code block.
- No text before JSON.
- No text after JSON.
- The JSON must contain exactly these top-level keys:
  "accident_time", "center_x", "center_y", "type", "input_mode",
  "selected_candidate_index", "selected_candidate_time", "candidate_summaries"
- "type" must be one of: ["rear-end", "head-on", "sideswipe", "t-bone", "single"]
- "input_mode" must be one of: ["video+frames", "video_only"]
- "candidate_summaries" must be an array of 1 to 3 objects
- each candidate summary must contain:
  "candidate_index", "candidate_time", "assessment"
- allowed assessment values are:
  ["likely_impact", "possible_context", "unlikely"]
- candidate summary may optionally contain:
  "used_for_final"
- if selected_candidate_index is not null, mark the matching candidate summary with "used_for_final": true

Output format:
{{
  "accident_time": <float>,
  "center_x": <float>,
  "center_y": <float>,
  "type": "<one of: rear-end, head-on, sideswipe, t-bone, single>",
  "input_mode": "<video+frames or video_only>",
  "selected_candidate_index": <integer or null>,
  "selected_candidate_time": <float or null>,
  "candidate_summaries": [
    {{
      "candidate_index": <integer>,
      "candidate_time": <float>,
      "assessment": "<likely_impact or possible_context or unlikely>",
      "used_for_final": <true or false, optional>
    }}
  ]
}}
"""
    return prompt.strip()


def call_qwen_for_media_bundle(
    model,
    processor,
    media_items: List[Dict[str, str]],
    prompt: str,
    rel_path: str,
    stage: str,
    raw_log_path: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_new_tokens: int = SINGLECALL_MAX_NEW_TOKENS,
) -> Dict[str, Any]:
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
            "content": [*media_items, {"type": "text", "text": prompt}],
        },
    ]

    append_raw_log(
        raw_log_path,
        {
            "path": rel_path,
            "stage": stage,
            "event": "request_start",
            "media_items": media_items,
            "max_new_tokens": max_new_tokens,
        },
    )

    last_error = None
    for attempt in range(1, max_retries + 1):
        request_started = time.perf_counter()
        try:
            processed = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
            processed = move_inputs_to_device(processed, get_model_input_device(model))

            generation_kwargs = dict(
                **processed,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

            with torch.inference_mode():
                output_ids = model.generate(**generation_kwargs)

            input_ids = processed.get("input_ids")
            if input_ids is not None and hasattr(input_ids, "shape"):
                generated_ids = output_ids[:, input_ids.shape[-1] :]
            else:
                generated_ids = output_ids

            collected_text = processor.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0].strip()
            elapsed_sec = time.perf_counter() - request_started

            append_raw_log(
                raw_log_path,
                {
                    "path": rel_path,
                    "stage": stage,
                    "attempt": attempt,
                    "event": "raw_output",
                    "elapsed_sec": elapsed_sec,
                    "raw_output": collected_text,
                },
            )
            print(f"  -> [{stage}] model output: {collected_text}")

            parsed = extract_first_json_object(collected_text)
            if parsed is None:
                last_error = f"JSON parse failed on attempt {attempt}."
                append_raw_log(
                    raw_log_path,
                    {
                        "path": rel_path,
                        "stage": stage,
                        "attempt": attempt,
                        "event": "parse_failure",
                        "elapsed_sec": elapsed_sec,
                        "error": last_error,
                    },
                )
                return {
                    "parsed": None,
                    "raw_output": collected_text,
                    "error": None,
                    "elapsed_sec": elapsed_sec,
                    "parse_error": last_error,
                }

            append_raw_log(
                raw_log_path,
                {
                    "path": rel_path,
                    "stage": stage,
                    "attempt": attempt,
                    "event": "parsed_json",
                    "elapsed_sec": elapsed_sec,
                    "parsed_json": parsed,
                },
            )
            print(f"  -> [{stage}] JSON parse succeeded.")
            return {
                "parsed": parsed,
                "raw_output": collected_text,
                "error": None,
                "elapsed_sec": elapsed_sec,
                "parse_error": None,
            }
        except Exception as exc:
            elapsed_sec = time.perf_counter() - request_started
            last_error = str(exc)
            append_raw_log(
                raw_log_path,
                {
                    "path": rel_path,
                    "stage": stage,
                    "attempt": attempt,
                    "event": "request_exception",
                    "elapsed_sec": elapsed_sec,
                    "error": last_error,
                },
            )
            print(f"  -> [{stage}] request attempt {attempt} failed: {last_error}")

    return {
        "parsed": None,
        "raw_output": None,
        "error": last_error,
        "elapsed_sec": None,
        "parse_error": None,
    }


def validate_singlecall_prediction(
    result: Dict[str, Any],
    meta: Dict[str, str],
    candidate_entries: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    accident_time = validate_time_prediction(result, meta)
    if accident_time is None:
        return None, "Invalid or missing accident_time."

    location = validate_location_prediction(result)
    if location is None:
        return None, "Invalid or missing center_x/center_y."

    accident_type = validate_type_prediction(result)
    if accident_type is None:
        return None, "Invalid or missing type."

    input_mode = str(result.get("input_mode", "")).strip()
    if input_mode not in ALLOWED_INPUT_MODES:
        return None, f"Invalid input_mode: {result.get('input_mode')}"

    candidate_time_by_index = {
        int(candidate["candidate_index"]): float(candidate["candidate_time"])
        for candidate in candidate_entries
    }

    selected_candidate_index = coerce_optional_int(result.get("selected_candidate_index"))
    if input_mode == "video_only":
        if selected_candidate_index is not None:
            return None, "selected_candidate_index must be null when input_mode is video_only."
    elif selected_candidate_index not in candidate_time_by_index:
        return None, f"selected_candidate_index must be one of {sorted(candidate_time_by_index)}."

    selected_candidate_time_value = result.get("selected_candidate_time")
    selected_candidate_time = safe_float(selected_candidate_time_value)
    if selected_candidate_index is None:
        if selected_candidate_time is not None:
            return None, "selected_candidate_time must be null when selected_candidate_index is null."
    else:
        if selected_candidate_time is None:
            return None, "selected_candidate_time is required when selected_candidate_index is set."
        expected_time = candidate_time_by_index[selected_candidate_index]
        if abs(selected_candidate_time - expected_time) > CANDIDATE_TIME_TOLERANCE_SEC:
            return None, "selected_candidate_time does not match the referenced candidate time."

    candidate_summaries_raw = result.get("candidate_summaries")
    if not isinstance(candidate_summaries_raw, list):
        return None, "candidate_summaries must be a list."
    if not 1 <= len(candidate_summaries_raw) <= CANDIDATE_FRAME_COUNT:
        return None, "candidate_summaries must have length 1..3."

    validated_summaries: List[Dict[str, Any]] = []
    seen_indices = set()
    used_for_final_indices: List[int] = []
    for summary in candidate_summaries_raw:
        if not isinstance(summary, dict):
            return None, "Each candidate summary must be an object."
        summary_index = coerce_optional_int(summary.get("candidate_index"))
        if summary_index is None or summary_index not in candidate_time_by_index:
            return None, f"Invalid candidate_index in candidate_summaries: {summary.get('candidate_index')}"
        if summary_index in seen_indices:
            return None, f"Duplicate candidate_index in candidate_summaries: {summary_index}"
        seen_indices.add(summary_index)

        summary_time = safe_float(summary.get("candidate_time"))
        if summary_time is None:
            return None, f"Missing candidate_time for candidate_index {summary_index}"
        expected_time = candidate_time_by_index[summary_index]
        if abs(summary_time - expected_time) > CANDIDATE_TIME_TOLERANCE_SEC:
            return None, f"candidate_time mismatch for candidate_index {summary_index}"

        assessment = str(summary.get("assessment", "")).strip()
        if assessment not in ALLOWED_ASSESSMENTS:
            return None, f"Invalid assessment for candidate_index {summary_index}: {assessment}"

        used_for_final = bool(summary.get("used_for_final", False))
        if used_for_final:
            used_for_final_indices.append(summary_index)

        validated_summaries.append(
            {
                "candidate_index": summary_index,
                "candidate_time": round(float(summary_time), 3),
                "assessment": assessment,
                "used_for_final": used_for_final,
            }
        )

    if len(used_for_final_indices) > 1:
        return None, "At most one candidate summary may have used_for_final=true."
    if used_for_final_indices:
        if selected_candidate_index is None:
            return None, "used_for_final=true is not allowed when selected_candidate_index is null."
        if used_for_final_indices[0] != selected_candidate_index:
            return None, "used_for_final summary must match selected_candidate_index."
    if selected_candidate_index is not None and selected_candidate_index not in seen_indices:
        return None, "selected_candidate_index must appear in candidate_summaries."
    if selected_candidate_index is not None and not used_for_final_indices:
        return None, "The selected candidate must be marked with used_for_final=true."

    validated = {
        "accident_time": round(float(accident_time), 3),
        "center_x": round(float(location["center_x"]), 3),
        "center_y": round(float(location["center_y"]), 3),
        "type": accident_type,
        "input_mode": input_mode,
        "selected_candidate_index": selected_candidate_index,
        "selected_candidate_time": round(float(selected_candidate_time), 3) if selected_candidate_time is not None else None,
        "candidate_summaries": validated_summaries,
    }
    return validated, None


def build_failure_payload(
    stage: str,
    error: str,
    abs_video_path: str,
    meta: Dict[str, str],
    raw_log_path: str,
    default_row: Dict[str, Any],
    candidate_entries: List[Dict[str, Any]],
    extracted_frames: List[Dict[str, Any]],
    fallback_info: Dict[str, Any],
    timing: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "status": "failed",
        "failed_stage": stage,
        "error": error,
        "video_path": abs_video_path,
        "metadata_csv_path": METADATA_CSV_PATH,
        "metadata": meta,
        "raw_log_path": raw_log_path,
        "submission_row": default_row,
        "candidate_context": {
            "candidate_entries": candidate_entries,
            "extracted_frames": extracted_frames,
        },
        "fallback": fallback_info,
        "timing": timing,
    }


def write_comparison_payload(
    path: str,
    rel_path: str,
    baseline_payload: Optional[Dict[str, Any]],
    baseline_status: Optional[str],
    baseline_latency_sec: Optional[float],
    experimental_payload: Dict[str, Any],
    experimental_status: str,
    experimental_latency_sec: float,
    candidate_entries: List[Dict[str, Any]],
) -> None:
    payload = {
        "path": rel_path,
        "baseline_status": baseline_status,
        "baseline_result": baseline_payload.get("result") if baseline_payload else None,
        "baseline_failed_stage": baseline_payload.get("failed_stage") if baseline_payload else None,
        "baseline_error": baseline_payload.get("error") if baseline_payload else None,
        "baseline_latency_sec": baseline_latency_sec,
        "baseline_raw_log_path": baseline_payload.get("raw_log_path") if baseline_payload else None,
        "experimental_status": experimental_status,
        "experimental_result": experimental_payload.get("result"),
        "experimental_failed_stage": experimental_payload.get("failed_stage"),
        "experimental_error": experimental_payload.get("error"),
        "experimental_latency_sec": experimental_latency_sec,
        "experimental_input_mode": (
            experimental_payload.get("result", {}).get("input_mode")
            if isinstance(experimental_payload.get("result"), dict)
            else None
        ),
        "experimental_raw_log_path": experimental_payload.get("raw_log_path"),
        "multimodal_fallback_used": experimental_payload.get("fallback", {}).get("used"),
        "fallback_reason": experimental_payload.get("fallback", {}).get("reason"),
        "candidate_times": [
            {
                "candidate_index": item["candidate_index"],
                "candidate_time": item["candidate_time"],
                "source": item["source"],
                "label": item["label"],
            }
            for item in candidate_entries
        ],
    }
    write_json(path, payload)


def run_singlecall_video_inference(
    meta: Dict[str, str],
    video_dir: str,
    model,
    processor,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    total_started = time.perf_counter()
    rel_path = meta["path"]
    video_name = os.path.basename(rel_path)
    video_stem = os.path.splitext(video_name)[0]
    abs_video_path = os.path.abspath(os.path.join(video_dir, video_name))
    raw_log_path = os.path.abspath(os.path.join(RAW_LOG_DIR, f"{video_stem}.jsonl"))
    result_json_path = os.path.abspath(os.path.join(RESULT_JSON_DIR, f"{video_stem}.json"))

    if os.path.exists(raw_log_path):
        os.remove(raw_log_path)

    default_row = build_submission_row(meta)
    fallback_info: Dict[str, Any] = {"used": False, "reason": None}
    timing: Dict[str, Any] = {
        "candidate_frame_extraction_sec": 0.0,
        "multimodal_request_sec": None,
        "video_only_request_sec": None,
        "qwen_request_sec": 0.0,
        "total_inference_sec": 0.0,
    }

    candidate_entries: List[Dict[str, Any]] = []
    extracted_frames: List[Dict[str, Any]] = []

    def finalize_failure(stage: str, error: str) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        timing["qwen_request_sec"] = round(float(timing["qwen_request_sec"]), 4)
        timing["total_inference_sec"] = round(time.perf_counter() - total_started, 4)
        payload = build_failure_payload(
            stage=stage,
            error=error,
            abs_video_path=abs_video_path,
            meta=meta,
            raw_log_path=raw_log_path,
            default_row=default_row,
            candidate_entries=candidate_entries,
            extracted_frames=extracted_frames,
            fallback_info=fallback_info,
            timing=timing,
        )
        write_json(result_json_path, payload)
        print(f"  -> [{stage}] failed. Saved result to: {result_json_path}")
        return payload, default_row, "failed"

    if not os.path.exists(abs_video_path):
        return finalize_failure("video", f"Video file not found: {abs_video_path}")

    feature_bundle = load_feature_bundle(video_stem)
    feature_index = prepare_feature_indices(feature_bundle, meta)
    time_feature_signals = build_time_feature_signals(feature_index, meta)
    time_feature_text = format_time_feature_text(feature_index, time_feature_signals)
    time_clip_candidates = build_time_clip_candidates(time_feature_signals, meta)
    candidate_entries = select_candidate_entries(time_clip_candidates, meta)

    append_raw_log(
        raw_log_path,
        {
            "path": rel_path,
            "stage": "candidate_selection",
            "feature_status": feature_bundle.get("status"),
            "feature_error": feature_bundle.get("error"),
            "feature_warning": feature_bundle.get("warning"),
            "candidate_entries": candidate_entries,
        },
    )

    extraction_started = time.perf_counter()
    extracted_frames = extract_candidate_frames(abs_video_path, video_stem, meta, candidate_entries, raw_log_path)
    timing["candidate_frame_extraction_sec"] = round(time.perf_counter() - extraction_started, 4)

    candidate_feature_notes = build_candidate_feature_notes(candidate_entries, extracted_frames)
    candidate_times = [float(candidate["candidate_time"]) for candidate in candidate_entries]

    multimodal_prompt = build_singlecall_prompt(meta, time_feature_text, candidate_times, candidate_feature_notes)
    multimodal_media_items = [{"type": "video", "path": abs_video_path}]
    for frame_info in extracted_frames:
        multimodal_media_items.append({"type": "image", "path": frame_info["frame_path"]})

    request_result: Optional[Dict[str, Any]] = None
    stage_used = "singlecall_multimodal"

    if extracted_frames:
        print(f"  -> [singlecall_multimodal] attempting video + {len(extracted_frames)} frame(s)")
        request_result = call_qwen_for_media_bundle(
            model=model,
            processor=processor,
            media_items=multimodal_media_items,
            prompt=multimodal_prompt,
            rel_path=rel_path,
            stage="singlecall_multimodal",
            raw_log_path=raw_log_path,
            max_new_tokens=SINGLECALL_MAX_NEW_TOKENS,
        )
        timing["multimodal_request_sec"] = request_result.get("elapsed_sec")
        if request_result.get("elapsed_sec") is not None:
            timing["qwen_request_sec"] += float(request_result["elapsed_sec"])
    else:
        append_raw_log(
            raw_log_path,
            {
                "path": rel_path,
                "stage": "singlecall_multimodal",
                "event": "skip",
                "reason": "No candidate frames were available for attachment.",
            },
        )
        request_result = {
            "parsed": None,
            "raw_output": None,
            "error": "No candidate frames were available for attachment.",
            "elapsed_sec": None,
            "parse_error": None,
        }

    if request_result.get("error") is not None:
        fallback_info["used"] = True
        fallback_info["reason"] = request_result["error"]
        video_only_notes = []
        for note in candidate_feature_notes:
            video_only_note = dict(note)
            video_only_note["attached_image"] = False
            video_only_note["image_attachment_order"] = None
            video_only_notes.append(video_only_note)
        video_only_prompt = build_singlecall_prompt(meta, time_feature_text, candidate_times, video_only_notes)
        stage_used = "singlecall_video_only"
        print("  -> [singlecall_video_only] retrying with video only")
        request_result = call_qwen_for_media_bundle(
            model=model,
            processor=processor,
            media_items=[{"type": "video", "path": abs_video_path}],
            prompt=video_only_prompt,
            rel_path=rel_path,
            stage="singlecall_video_only",
            raw_log_path=raw_log_path,
            max_new_tokens=SINGLECALL_MAX_NEW_TOKENS,
        )
        timing["video_only_request_sec"] = request_result.get("elapsed_sec")
        if request_result.get("elapsed_sec") is not None:
            timing["qwen_request_sec"] += float(request_result["elapsed_sec"])

    if request_result.get("parse_error") is not None:
        append_raw_log(
            raw_log_path,
            {
                "path": rel_path,
                "stage": stage_used,
                "event": "validation_failure",
                "error": request_result["parse_error"],
            },
        )
        return finalize_failure(stage_used, request_result["parse_error"])

    parsed_result = request_result.get("parsed")
    if parsed_result is None:
        return finalize_failure(stage_used, "Single-call request did not return a parsed JSON object.")

    validated_result, validation_error = validate_singlecall_prediction(parsed_result, meta, candidate_entries)
    if validation_error is not None:
        append_raw_log(
            raw_log_path,
            {
                "path": rel_path,
                "stage": stage_used,
                "event": "validation_failure",
                "error": validation_error,
                "parsed_json": parsed_result,
            },
        )
        return finalize_failure(stage_used, validation_error)

    timing["qwen_request_sec"] = round(float(timing["qwen_request_sec"]), 4)
    timing["total_inference_sec"] = round(time.perf_counter() - total_started, 4)
    result_data = dict(validated_result)
    submission_row = build_submission_row(meta, result_data)
    payload = {
        "status": "ok",
        "video_path": abs_video_path,
        "metadata_csv_path": METADATA_CSV_PATH,
        "metadata": meta,
        "result": result_data,
        "raw_log_path": raw_log_path,
        "submission_row": submission_row,
        "candidate_context": {
            "candidate_entries": candidate_entries,
            "extracted_frames": extracted_frames,
        },
        "fallback": fallback_info,
        "timing": timing,
    }
    write_json(result_json_path, payload)
    print(f"  -> [done] saved result to: {result_json_path}")
    return payload, submission_row, "ok"


def load_model_and_processor():
    print(f"Loading model: {MODEL_NAME}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model_load_config, load_notes = build_model_load_config()
    for note in load_notes:
        print(f"Model load config: {note}")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        quantization_config=quantization_config,
        **model_load_config,
    )
    print(f"Model input device: {get_model_input_device(model)}")
    return model, processor


def main() -> None:
    started_at = now_utc_iso()
    ensure_experiment_output_dirs()

    if EXPERIMENT_MODE not in ALLOWED_EXPERIMENT_MODES:
        summary = build_experiment_run_summary(
            video_dir="",
            total_rows=0,
            experiment_success_count=0,
            experiment_failure_count=0,
            experiment_failed_videos=[],
            baseline_success_count=0,
            baseline_failure_count=0,
            started_at=started_at,
            finished_at=now_utc_iso(),
            global_error=(
                f"Invalid QWEN_EXPERIMENT_MODE: {EXPERIMENT_MODE}. "
                f"Allowed values: {sorted(ALLOWED_EXPERIMENT_MODES)}"
            ),
        )
        write_json(RUN_SUMMARY_PATH, summary)
        print(summary["global_error"])
        print(f"Saved run summary to: {RUN_SUMMARY_PATH}")
        return

    metadata_rows = load_all_metadata(METADATA_CSV_PATH)
    effective_video_dir = resolve_video_dir()
    if TEST_MODE:
        metadata_rows = filter_metadata_for_test(metadata_rows, effective_video_dir, TEST_LIMIT)

    print(f"Experiment mode: {EXPERIMENT_MODE}")
    print(f"Video dir: {effective_video_dir}")
    print(f"Metadata CSV: {METADATA_CSV_PATH}")
    print(f"Experiment output dir: {EXPERIMENT_OUTPUT_DIR}")
    print(f"Total rows: {len(metadata_rows)}")
    if TEST_MODE:
        print(f"Test mode: enabled (limit={TEST_LIMIT if TEST_LIMIT > 0 else 'all matches'})")

    if not metadata_rows:
        write_submission_csv_to(SUBMISSION_CSV_PATH, [])
        summary = build_experiment_run_summary(
            video_dir=effective_video_dir,
            total_rows=0,
            experiment_success_count=0,
            experiment_failure_count=0,
            experiment_failed_videos=[],
            baseline_success_count=0,
            baseline_failure_count=0,
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
        rows = [build_submission_row(meta) for meta in metadata_rows]
        write_submission_csv_to(SUBMISSION_CSV_PATH, rows)
        summary = build_experiment_run_summary(
            video_dir=effective_video_dir,
            total_rows=len(metadata_rows),
            experiment_success_count=0,
            experiment_failure_count=len(metadata_rows),
            experiment_failed_videos=[],
            baseline_success_count=0,
            baseline_failure_count=0,
            started_at=started_at,
            finished_at=now_utc_iso(),
            global_error=compatibility_error,
        )
        write_json(RUN_SUMMARY_PATH, summary)
        print(compatibility_error)
        print(f"Saved default submission to: {SUBMISSION_CSV_PATH}")
        print(f"Saved run summary to: {RUN_SUMMARY_PATH}")
        return

    try:
        model, processor = load_model_and_processor()
    except Exception as exc:
        rows = [build_submission_row(meta) for meta in metadata_rows]
        write_submission_csv_to(SUBMISSION_CSV_PATH, rows)
        summary = build_experiment_run_summary(
            video_dir=effective_video_dir,
            total_rows=len(metadata_rows),
            experiment_success_count=0,
            experiment_failure_count=len(metadata_rows),
            experiment_failed_videos=[],
            baseline_success_count=0,
            baseline_failure_count=0,
            started_at=started_at,
            finished_at=now_utc_iso(),
            global_error=f"Model loading failed: {exc}",
        )
        write_json(RUN_SUMMARY_PATH, summary)
        print(f"Model loading failed: {exc}")
        print(f"Saved default submission to: {SUBMISSION_CSV_PATH}")
        print(f"Saved run summary to: {RUN_SUMMARY_PATH}")
        return

    submission_rows: List[Dict[str, Any]] = []
    experiment_failed_videos: List[Dict[str, Any]] = []
    experiment_success_count = 0
    experiment_failure_count = 0
    baseline_success_count = 0
    baseline_failure_count = 0

    for idx, meta in enumerate(metadata_rows, start=1):
        print(f"\n[{idx}/{len(metadata_rows)}] Processing: {meta['path']}")
        baseline_payload: Optional[Dict[str, Any]] = None
        baseline_status: Optional[str] = None
        baseline_latency_sec: Optional[float] = None

        if EXPERIMENT_MODE in {"compare", "baseline_only"}:
            baseline_started = time.perf_counter()
            baseline_payload, _, baseline_status = run_single_video_inference(
                meta,
                effective_video_dir,
                model,
                processor,
            )
            baseline_latency_sec = round(time.perf_counter() - baseline_started, 4)
            if baseline_status == "ok":
                baseline_success_count += 1
            else:
                baseline_failure_count += 1

        if EXPERIMENT_MODE == "baseline_only":
            row = baseline_payload.get("submission_row") if baseline_payload else build_submission_row(meta)
            submission_rows.append(row)
            continue

        experimental_started = time.perf_counter()
        experimental_payload, experimental_row, experimental_status = run_singlecall_video_inference(
            meta,
            effective_video_dir,
            model,
            processor,
        )
        experimental_latency_sec = round(time.perf_counter() - experimental_started, 4)
        submission_rows.append(experimental_row)

        if experimental_status == "ok":
            experiment_success_count += 1
        else:
            experiment_failure_count += 1
            experiment_failed_videos.append(
                {
                    "path": meta["path"],
                    "failed_stage": experimental_payload.get("failed_stage", "unknown"),
                    "error": experimental_payload.get("error", ""),
                }
            )

        if EXPERIMENT_MODE == "compare":
            comparison_path = os.path.abspath(
                os.path.join(COMPARISON_DIR, f"{os.path.splitext(os.path.basename(meta['path']))[0]}.json")
            )
            candidate_entries = (
                experimental_payload.get("candidate_context", {}).get("candidate_entries", [])
                if isinstance(experimental_payload.get("candidate_context"), dict)
                else []
            )
            write_comparison_payload(
                path=comparison_path,
                rel_path=meta["path"],
                baseline_payload=baseline_payload,
                baseline_status=baseline_status,
                baseline_latency_sec=baseline_latency_sec,
                experimental_payload=experimental_payload,
                experimental_status=experimental_status,
                experimental_latency_sec=experimental_latency_sec,
                candidate_entries=candidate_entries,
            )

    write_submission_csv_to(SUBMISSION_CSV_PATH, submission_rows)
    summary = build_experiment_run_summary(
        video_dir=effective_video_dir,
        total_rows=len(metadata_rows),
        experiment_success_count=experiment_success_count,
        experiment_failure_count=experiment_failure_count,
        experiment_failed_videos=experiment_failed_videos,
        baseline_success_count=baseline_success_count,
        baseline_failure_count=baseline_failure_count,
        started_at=started_at,
        finished_at=now_utc_iso(),
    )
    write_json(RUN_SUMMARY_PATH, summary)

    print("\nExperiment run finished.")
    print(f"Total rows: {len(metadata_rows)}")
    print(f"Experimental success count: {experiment_success_count}")
    print(f"Experimental failure count: {experiment_failure_count}")
    if EXPERIMENT_MODE in {"compare", "baseline_only"}:
        print(f"Baseline success count: {baseline_success_count}")
        print(f"Baseline failure count: {baseline_failure_count}")
    print(f"Saved experiment submission CSV to: {SUBMISSION_CSV_PATH}")
    print(f"Saved experiment run summary to: {RUN_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
