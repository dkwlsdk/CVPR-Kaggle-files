import argparse
import fcntl
#!/usr/bin/env python3
import importlib
import os
import csv
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Sequence

import cv2
import torch
from PIL import Image
from transformers import PretrainedConfig

PROGRAM_START_TIME = time.perf_counter()

# --- GPU 환경 및 보안 설정 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# --- 모델 및 생성 제어 하이퍼파라미터 ---
MODEL_NAME = "Qwen/Qwen3.5-9B"
# MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
# MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# OUTPUT_DIR = "/root/Desktop/workspace/ja/qwen_test_output_9B/04.frame64_fps2_all"
OUTPUT_DIR = "/root/Desktop/workspace/ja/qwen_test_output_9B/08.frame64_fps2_ver2"

# 모델이 생성할 최대 답변 길이
MAX_NEW_TOKENS = 15360
# 답변의 창의성 제어 (0.0은 가장 결정론적인 답변)
TEMPERATURE = 0.0
# 답변 생성 시 고려할 누적 확률 범위
TOP_P = 1.0
# 동일 문구 반복 방지를 위한 가중치
REPETITION_PENALTY = 1.2

# PyTorch 메모리 파편화 방지 설정
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# HuggingFace 인증 토큰 (비공개 모델 로드 시 필요)
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

# --- 시각 분석 및 샘플링 설정 ---
# 프레임당 픽셀 수 제한 (OOM 에러 방지를 위해 중요)
MIN_PIXELS = 256 * 28 * 28     # 최소 픽셀
MAX_PIXELS = 1024 * 28 * 28    # 최대 픽셀 (기존 1280에서 1024로 하향)
# 비디오 분석 시 초당 샘플링할 프레임 수
VIDEO_FPS = 2.0

# --- 디버그 출력 설정 ---
DEBUG_VISIBLE_EVIDENCE = True
MAX_EVIDENCE_CHARS = 220
SNAP_TIME_TO_VIDEO_FPS_GRID = True
MAX_VIDEO_SAMPLED_FRAMES = 64
TIME_STAGE_MAX_PIXELS = MIN_PIXELS

# --- 재시도 및 검증 로직 설정 ---
MAX_TIME_RETRIES = 2           # 사고 시점 예측 실패 시 최대 재시도 횟수
# 이전 예측값과 현재 예측값이 동일한지 판단할 오차 범위 (초 단위)
TIME_MATCH_THRESHOLD = 0.1
# 모델 호출 자체의 재시도 횟수 (JSON 파싱 실패 등 대비)
MAX_CALL_RETRIES = 2

# --- vLLM (Ver2) 전용 세부 설정 (FPS 및 픽셀 한도) ---
COARSE_VIDEO_FPS = 2.0
TIME_COARSE_MAX_PIXELS = 384 * 28 * 28
TIME_REFINE_FPS = 6.0
TIME_REFINE_MAX_FRAMES = 24
TIME_REFINE_MAX_PIXELS = 640 * 28 * 28
TIME_BOUNDARY_FPS = 12.0
TIME_BOUNDARY_MAX_FRAMES = 18
TIME_BOUNDARY_MAX_PIXELS = 768 * 28 * 28

# --- 로케이션 및 파트너 검증 설정 ---
LOCATION_CONTEXT_OFFSETS = (-2, 0, 2)
LOCATION_MIN_FOCUS_FRACTION = 0.42
LOCATION_WIDE_MIN_FOCUS_FRACTION = 0.72
TYPE_MIN_FOCUS_FRACTION = 0.48
TYPE_WIDE_MIN_FOCUS_FRACTION = 0.72
PARTNER_VERIFY_MIN_CONFIDENCE = 0.65

# --- 사고 판별용 키워드 사전 ---
FIXED_OBJECT_KEYWORDS = (
    "guardrail", "barrier", "median", "divider", "pole", "post", "wall", "curb", "sign", "tree", "fence", "rail"
)
AFTERMATH_KEYWORDS = (
    "flipping", "spinning", "sliding", "resting", "burning", "leaking", "drifting", "tumbling", "stopped", "parked", "debris"
)

# --- 경로 및 기타 설정 ---
BASE_DIR = "/root/Desktop/workspace/ja"
VIDEO_DIR = "/root/Desktop/workspace/ja/CVPR-Kaggle-files/데이터셋/test_videos"
METADATA_CSV_PATH = "/root/Desktop/workspace/ja/CVPR-Kaggle-files/데이터셋/test_metadata.csv"

# 결과 저장 경로 설정
PREDICTION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
RAW_LOG_PATH = os.path.join(OUTPUT_DIR, "raw_outputs.jsonl")
FRAME_DIR = os.path.join(OUTPUT_DIR, "frames")

# 사고 유형 분류 목록
VALID_TYPES = {"rear-end", "head-on", "sideswipe", "t-bone", "single"}

# 테스트 모드 설정 (True일 경우 특정 폴더의 일부 영상만 처리)
# TEST_MODE = False
TEST_MODE = True
TEST_VIDEO_DIR = "/root/Desktop/workspace/ja/temp"
# TEST_VIDEO_DIR = "/root/Desktop/workspace/ja/CVPR-Kaggle-files/데이터셋/test_videos"
TEST_LIMIT = 0
SKIP_EXISTING = True
# SKIP_EXISTING = False

# GPU 메모리 예약 공간 (MiB 단위)
CUDA_MEMORY_RESERVE_MIB = 256
VLLM_MAX_MODEL_LEN = 32768
VLLM_GPU_MEMORY_UTILIZATION = 0.90

# --- 하드웨어 및 모델 로드 유틸리티 ---

def ensure_runtime_compatibility() -> Optional[str]:
    """필요한 라이브러리 버전을 충족하는지 확인합니다."""
    MIN_PYTHON = (3, 8)
    py_version = tuple(map(int, os.sys.version_info[:2]))
    if py_version < MIN_PYTHON:
        return f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ 버전이 필요합니다."
    return None

def patch_transformers_rope_validation_compat() -> None:
    """transformers dev 버전과 vLLM Qwen3.5 config 간 rope validation 타입 충돌을 우회합니다."""
    original = getattr(PretrainedConfig, "_check_received_keys", None)
    if original is None or getattr(original, "_qwen35_vllm_compat", False):
        return

    def patched_check_received_keys(
        rope_type: str,
        received_keys: set,
        required_keys: set,
        optional_keys: Optional[set] = None,
        ignore_keys: Optional[set] = None,
    ):
        if ignore_keys is not None and not isinstance(ignore_keys, set):
            ignore_keys = set(ignore_keys)
        return original(
            rope_type,
            received_keys,
            required_keys,
            optional_keys=optional_keys,
            ignore_keys=ignore_keys,
        )

    patched_check_received_keys._qwen35_vllm_compat = True
    PretrainedConfig._check_received_keys = staticmethod(patched_check_received_keys)

def infer_tensor_parallel_size() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return max(1, len([token for token in visible.split(",") if token.strip()]))
    return max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1

def load_vllm_model():
    """vLLM 엔진과 샘플링 파라미터를 로드합니다."""
    patch_transformers_rope_validation_compat()
    try:
        vllm_mod = importlib.import_module("vllm")
        LLM = getattr(vllm_mod, "LLM")
        SamplingParams = getattr(vllm_mod, "SamplingParams")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("vllm이 설치되지 않았습니다. pip install vllm") from exc

    tensor_parallel_size = infer_tensor_parallel_size()
    print(
        f"  -> vLLM 로드: model={MODEL_NAME}, tp={tensor_parallel_size}, "
        f"max_model_len={VLLM_MAX_MODEL_LEN}"
    )

    model = LLM(
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=VLLM_MAX_MODEL_LEN,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        limit_mm_per_prompt={"image": MAX_VIDEO_SAMPLED_FRAMES},
        mm_processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
        max_num_seqs=1,
    )
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
    )
    return model, sampling_params

# --- 프롬프트 빌딩 섹션 ---

def build_time_prompt(metadata: Dict[str, str], sampled_timestamps: List[float], failed_indices: Optional[List[int]] = None) -> str:
    """샘플링된 프레임 시퀀스에서 첫 사고 프레임 인덱스를 찾는 프롬프트를 생성합니다."""
    region = metadata.get("region", "")
    scene_layout = metadata.get("scene_layout", "")
    weather = metadata.get("weather", "")
    day_time = metadata.get("day_time", "")
    quality = metadata.get("quality", "")
    duration = metadata.get("duration", "")
    no_frames = metadata.get("no_frames", "")
    height = metadata.get("height", "")
    width = metadata.get("width", "")

    retry_note = ""
    if failed_indices:
        retry_note = (
            "\n\n[CRITICAL NOTE]\n"
            f"You previously selected frame indices {failed_indices}, but those were INCORRECT. "
            "Do not output those frame indices again. Choose a different frame index."
        )

    frame_catalog = ", ".join(f"{idx}: {ts:.3f}s" for idx, ts in enumerate(sampled_timestamps))

    output_rules = """
Critical output rules:
- The FINAL output MUST be exactly one JSON object.
- No markdown.
- No bullet points.
- No code block.
- No text before JSON.
- No text after JSON.
- Required keys:
  "frame_index", "confidence", "evidence"
- "frame_index" must be an integer index from the provided frame list.
- "confidence" must be a float between 0.0 and 1.0.
- "evidence" must be a short summary of visible cues only, max 2 sentences.

Output format:
{
  "frame_index": <int>,
  "confidence": <float>,
  "evidence": "<brief visible evidence>"
}
""".strip() if DEBUG_VISIBLE_EVIDENCE else """
Critical output rules:
- The FINAL output MUST be the JSON object.
- No reasoning.
- No analysis.
- No markdown.
- No bullet points.
- No code block.
- No text before JSON.
- No text after JSON.
- The JSON must contain exactly this key:
  "frame_index"

Output format:
{
  "frame_index": <int>
}
""".strip()

    prompt = f"""{retry_note}
You are an expert traffic accident analyst looking at a chronological sequence of sampled CCTV frames.

Your task is to identify the FIRST sampled frame where a traffic accident clearly begins.

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

The frames are shown in chronological order.
The exact timestamp of each sampled frame is:
{frame_catalog}

Instructions:
1. Analyze the sampled frames in order.
2. Choose the earliest frame_index where the accident clearly begins.
3. Select the first frame where physical contact begins, or where impact is clearly immediate and unavoidable.
4. If the accident begins between two sampled frames, choose the earliest sampled frame that clearly shows the accident has started.
5. Ignore location and accident type in this step.

{output_rules}
"""
    return prompt.strip()

def build_location_prompt(metadata: Dict[str, str], accident_time: float) -> str:
    """사고 위치를 좌표로 찾기 위한 이미지 분석 프롬프트를 생성합니다."""
    region = metadata.get("region", "")
    scene_layout = metadata.get("scene_layout", "")
    weather = metadata.get("weather", "")
    day_time = metadata.get("day_time", "")
    quality = metadata.get("quality", "")
    duration = metadata.get("duration", "")
    no_frames = metadata.get("no_frames", "")
    height = metadata.get("height", "")
    width = metadata.get("width", "")

    output_rules = """
Critical output rules:
- The FINAL output MUST be exactly one JSON object.
- No markdown.
- No bullet points.
- No code block.
- No text before JSON.
- No text after JSON.
- Required keys:
  "center_x", "center_y", "confidence", "evidence"
- "confidence" must be a float between 0.0 and 1.0.
- "evidence" must be a short summary of visible cues only, max 2 sentences.

Output format:
{
  "center_x": <float>,
  "center_y": <float>,
  "confidence": <float>,
  "evidence": "<brief visible evidence>"
}
""".strip() if DEBUG_VISIBLE_EVIDENCE else """
Critical output rules:
- The FINAL output MUST be the JSON object.
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
{
  "center_x": <float>,
  "center_y": <float>
}
""".strip()

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
- frame_width: {width}

Your task is to precisely localize the primary collision point in this frame.

Instructions:
1. Focus on the main collision area where vehicles or objects are physically impacting.
2. Output normalized coordinates of the center of this collision region:
   - center_x: from left (0.0) to right (1.0)
   - center_y: from top (0.0) to bottom (1.0)
3. The coordinates must indicate the center of the actual contact region, not the center of the whole vehicle.
4. Ignore accident type classification in this step.
5. If uncertain, choose the single best estimate.

{output_rules}
"""
    return prompt.strip()

def build_type_prompt(metadata: Dict[str, str], accident_time: float) -> str:
    """사고 유형을 분류하기 위한 이미지 분석 프롬프트를 생성합니다."""
    region = metadata.get("region", "")
    scene_layout = metadata.get("scene_layout", "")
    weather = metadata.get("weather", "")
    day_time = metadata.get("day_time", "")
    quality = metadata.get("quality", "")
    duration = metadata.get("duration", "")
    no_frames = metadata.get("no_frames", "")
    height = metadata.get("height", "")
    width = metadata.get("width", "")

    output_rules = """
Critical output rules:
- The FINAL output MUST be exactly one JSON object.
- No markdown.
- No bullet points.
- No code block.
- No text before JSON.
- No text after JSON.
- Required keys:
  "type", "confidence", "evidence"
- "confidence" must be a float between 0.0 and 1.0.
- "evidence" must be a short summary of visible cues only, max 2 sentences.

Output format:
{
  "type": "<one of: rear-end, head-on, sideswipe, t-bone, single>",
  "confidence": <float>,
  "evidence": "<brief visible evidence>"
}
""".strip() if DEBUG_VISIBLE_EVIDENCE else """
Critical output rules:
- The FINAL output MUST be the JSON object.
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
{
  "type": "<one of: rear-end, head-on, sideswipe, t-bone, single>"
}
""".strip()

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
- frame_width: {width}

Definitions of accident types (choose exactly one):
- rear-end: One vehicle crashes into the back of another vehicle traveling in the same direction.
- head-on: Two vehicles traveling in opposite directions collide front-to-front.
- sideswipe: Two vehicles moving in roughly the same direction make side-to-side contact while overlapping partially.
- t-bone: The front of one vehicle crashes into the side of another vehicle, forming a "T" shape.
- single: An accident involving only one vehicle (e.g., hitting a pole, barrier, guardrail, or going off the road) with no other vehicle collision.

Your task is to classify the accident type in this frame.

Instructions:
1. Carefully analyze the visible interaction between vehicles and/or objects.
2. Choose exactly one type from:
   ["rear-end", "head-on", "sideswipe", "t-bone", "single"].
3. If uncertain, choose the single best guess.

{output_rules}
"""
    return prompt.strip()

# --- 유틸리티 및 JSON 파싱 섹션 ---

def strip_thinking_text(text: str) -> str:
    """모델 답변에서 <think> 태그나 불필요한 서두 문구를 제거합니다."""
    if not text:
        return text
    # <think>...</think> 태그 제거
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # "The user wants me to...", "Okay, I understand..." 등 일반적인 대화 시작 문구 제거
    text = re.sub(r"^(?:The user wants|I understand|Okay|Let me|Based on).*?:\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()

def try_parse_single_json(candidate: str) -> Optional[Dict[str, Any]]:
    """문자열을 JSON 딕셔너리로 파싱을 시도합니다."""
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None

def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """텍스트에서 유효한 첫 번째 JSON 객체를 추출합니다."""
    # 0. 전처리
    # </think> 태그가 있으면 그 이후의 텍스트가 가장 정답일 확률이 높음
    if "</think>" in text:
        text = text.split("</think>")[-1]
    
    # 주석 및 특수값 처리
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\bNaN\b", "null", text)
    text = re.sub(r"\bInfinity\b", "999999", text)
    
    # 1. 모든 { ... } 구조를 스택 기반으로 추출하여 유효성 검사
    stack = []
    candidates = []
    for i, char in enumerate(text):
        if char == '{':
            stack.append(i)
        elif char == '}' and stack:
            start = stack.pop()
            cand = text[start : i + 1]
            candidates.append(cand.replace('\\"', '"')) # 이스케이프 보정

    # 가장 뒤(가장 전역적인 혹은 뒤에 쓰인 것)부터 유효한 JSON 탐색
    for cand in reversed(candidates):
        parsed = try_parse_single_json(cand)
        if parsed and (
            "frame_index" in parsed or 
            "center_x" in parsed or 
            "type" in parsed or 
            "collision_partner" in parsed
        ):
            return parsed
    
    return None

def validate_frame_index_prediction(result: Dict[str, Any], num_frames: int) -> Optional[int]:
    """예측된 프레임 인덱스가 유효한 범위인지 확인합니다."""
    try:
        frame_index = int(result["frame_index"])
    except (KeyError, TypeError, ValueError):
        return None
    if frame_index < 0 or frame_index >= num_frames:
        return None
    return frame_index

def maybe_get_confidence(result: Optional[Dict[str, Any]]) -> Optional[float]:
    if result is None:
        return None
    raw_conf = result.get("confidence")
    if raw_conf is None:
        return None
    try:
        confidence = float(raw_conf)
    except (TypeError, ValueError):
        return None
    return min(max(confidence, 0.0), 1.0)

def maybe_get_evidence(result: Optional[Dict[str, Any]]) -> Optional[str]:
    if result is None:
        return None
    raw_evidence = result.get("evidence")
    if raw_evidence is None:
        return None
    evidence = str(raw_evidence).strip()
    if not evidence:
        return None
    evidence = re.sub(r"\s+", " ", evidence)
    if len(evidence) > MAX_EVIDENCE_CHARS:
        evidence = evidence[: MAX_EVIDENCE_CHARS - 3].rstrip() + "..."
    return evidence

def print_debug_payload(stage: str, result: Optional[Dict[str, Any]]) -> None:
    if not DEBUG_VISIBLE_EVIDENCE or result is None:
        return
    confidence = maybe_get_confidence(result)
    evidence = maybe_get_evidence(result)
    if confidence is None and not evidence:
        return
    parts = []
    if confidence is not None:
        parts.append(f"confidence={confidence:.2f}")
    if evidence:
        parts.append(f"evidence={evidence}")
    print(f"  -> [{stage}] 디버그: " + " | ".join(parts))

def format_elapsed(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    minutes, rem = divmod(total_seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours > 0:
        return f"{hours:d}시간 {minutes:02d}분 {rem:05.2f}초"
    if minutes > 0:
        return f"{minutes:d}분 {rem:05.2f}초"
    return f"{rem:.2f}초"

def snap_time_to_grid(accident_time: float, fps: float) -> float:
    if fps <= 0:
        return accident_time
    step = 1.0 / fps
    snapped = round(accident_time / step) * step
    return round(snapped, 4)

def validate_location_prediction(result: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """예측된 좌표값을 0.0~1.0 사이로 정규화합니다."""
    try:
        center_x = float(result["center_x"])
        center_y = float(result["center_y"])
    except (KeyError, TypeError, ValueError):
        return None
    center_x = min(max(center_x, 0.0), 1.0)
    center_y = min(max(center_y, 0.0), 1.0)
    return {"center_x": center_x, "center_y": center_y}

def validate_type_prediction(result: Dict[str, Any]) -> Optional[str]:
    """예측된 사고 유형이 사전에 정의된 값인지 확인합니다."""
    try:
        accident_type = str(result["type"]).strip()
    except (KeyError, TypeError, ValueError):
        return None
    if accident_type not in VALID_TYPES:
        return None
    return accident_type

def append_raw_log(path: str, payload: Dict[str, Any]) -> None:
    """원시 모델 출력을 추후 디버깅을 위해 JSONL 파일에 추가 저장합니다."""
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def build_raw_log_payload(
    rel_path: str,
    stage: str,
    attempt: int,
    raw_output: str,
    parsed_output: Optional[Dict[str, Any]] = None,
    resolved_time_frame_index: Optional[int] = None,
    resolved_time_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """raw_outputs.jsonl에 저장할 공통 로그 payload를 생성합니다."""
    payload: Dict[str, Any] = {
        "path": rel_path,
        "stage": stage,
    }
    if parsed_output is not None:
        payload[f"parsed_output_{stage}"] = parsed_output
    if stage == "time":
        if resolved_time_frame_index is not None:
            payload["final_accident_frame"] = resolved_time_frame_index
        if resolved_time_seconds is not None:
            payload["final_accident_time"] = round(float(resolved_time_seconds), 4)
    payload["attempt"] = attempt
    payload["raw_output"] = raw_output
    return payload

def load_video_frames_for_vlm(
    video_path: str,
    sample_fps: float,
    max_frames: int = MAX_VIDEO_SAMPLED_FRAMES,
) -> Optional[Dict[str, Any]]:
    """OpenCV로 비디오를 샘플링해 VLM에 바로 전달할 프레임 리스트를 생성합니다."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if raw_fps <= 0:
        raw_fps = max(sample_fps, 1.0)

    if total_frames <= 0:
        cap.release()
        return None

    step = max(1, int(round(raw_fps / max(sample_fps, 0.1))))
    frame_indices = list(range(0, total_frames, step))
    if not frame_indices or frame_indices[-1] != total_frames - 1:
        frame_indices.append(total_frames - 1)

    if len(frame_indices) > max_frames:
        scale = (len(frame_indices) - 1) / max(max_frames - 1, 1)
        reduced = []
        for i in range(max_frames):
            reduced.append(frame_indices[min(round(i * scale), len(frame_indices) - 1)])
        frame_indices = sorted(set(reduced))

    frames = []
    target_set = set(frame_indices)
    next_targets = iter(sorted(target_set))
    current_target = next(next_targets, None)
    current_index = 0

    while current_target is not None:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if current_index == current_target:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            current_target = next(next_targets, None)
        current_index += 1

    cap.release()
    if not frames:
        return None

    effective_sample_fps = min(sample_fps, len(frames)) if len(frames) > 1 else sample_fps
    timestamps = [frame_idx / raw_fps for frame_idx in frame_indices[: len(frames)]]
    return {
        "frames": frames,
        "raw_fps": raw_fps,
        "sample_fps": effective_sample_fps,
        "num_frames": len(frames),
        "frame_indices": frame_indices[: len(frames)],
        "timestamps": timestamps,
    }

def resize_image_to_pixel_budget(image: Image.Image, max_pixels: int) -> Image.Image:
    """이미지를 지정한 픽셀 예산 이하로 축소해 멀티모달 토큰 수를 제어합니다."""
    width, height = image.size
    current_pixels = width * height
    if current_pixels <= max_pixels:
        return image

    scale = (max_pixels / float(current_pixels)) ** 0.5
    new_width = max(28, int(width * scale))
    new_height = max(28, int(height * scale))

    # Qwen 계열 비전 인코더가 다루기 쉬운 28 배수로 정렬합니다.
    new_width = max(28, (new_width // 28) * 28)
    new_height = max(28, (new_height // 28) * 28)
    return image.resize((new_width, new_height), Image.LANCZOS)

def extract_frame_at_time(
    video_path: str,
    accident_time: float,
    meta: Dict[str, str],
    fps_override: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """지정된 시간(초)에 해당하는 비디오 프레임을 추출합니다."""
    import cv2
    def _open():
        cap_ = cv2.VideoCapture(video_path)
        return cap_ if cap_.isOpened() else None
    
    cap = _open()
    if cap is None:
        return None
        
    fps = fps_override if fps_override is not None else cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # FPS가 비정상적일 경우 메타데이터 기반 계산 시도
    if fps <= 0 and meta.get("duration") and meta.get("no_frames"):
        try:
            fps = float(meta["no_frames"]) / float(meta["duration"])
        except Exception:
            fps = 0
            
    if fps <= 0:
        cap.release()
        return None
        
    # 시간(s) * FPS = 프레임 인덱스
    frame_index = int(accident_time * fps)
    frame_index = max(0, min(frame_index, max(0, total_frames - 1)))
    
    # 1차 시도: 프레임 번호로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if ret and frame is not None:
        cap.release()
        return {"frame": frame, "fps": fps, "frame_index": frame_index}
        
    # 2차 시도 (실패 시): 밀리초(ms) 단위 이동
    cap.release()
    cap = _open()
    if cap is None:
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, accident_time) * 1000.0)
    ret, frame = cap.read()
    if ret and frame is not None:
        cap.release()
        return {"frame": frame, "fps": fps, "frame_index": frame_index}
        
    cap.release()
    return None

def render_chatml_prompt(messages: List[Dict[str, Any]]) -> Tuple[str, List[Image.Image]]:
    """메시지 리스트를 vLLM용 ChatML 프롬프트와 이미지 리스트로 변환합니다."""
    parts: List[str] = []
    images: List[Image.Image] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        role_parts: List[str] = []

        if isinstance(content, str):
            role_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                item_type = item.get("type")
                if item_type == "text":
                    role_parts.append(str(item.get("text", "")))
                elif item_type == "image":
                    image = item.get("image")
                    if not isinstance(image, Image.Image):
                        raise TypeError(f"Expected PIL.Image, got {type(image)}")
                    images.append(image)
                    role_parts.append("<|vision_start|><|image_pad|><|vision_end|>")

        parts.append(f"<|im_start|>{role}\n{''.join(role_parts)}<|im_end|>")

    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts), images

def messages_to_vllm_request(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt, images = render_chatml_prompt(messages)
    request: Dict[str, Any] = {"prompt": prompt}
    if images:
        request["multi_modal_data"] = {"image": images}
    return request

def run_vllm_request(
    model,
    sampling_params,
    messages: List[Dict[str, Any]],
    rel_path: str,
    stage: str,
    max_retries: int = MAX_CALL_RETRIES,
) -> Optional[Dict[str, Any]]:
    """vLLM으로 단일 멀티모달 요청을 실행하고 JSON 응답을 파싱합니다."""
    request = messages_to_vllm_request(messages)
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            outputs = model.generate([request], sampling_params=sampling_params)
            collected_text = ""
            if outputs and outputs[0].outputs:
                collected_text = outputs[0].outputs[0].text.strip()

            print(f"  -> [{stage}] (시도 {attempt}/{max_retries}) 모델 출력: {collected_text}")

            parsed = extract_first_json_object(collected_text)
            append_raw_log(
                RAW_LOG_PATH,
                build_raw_log_payload(
                    rel_path=rel_path,
                    stage=stage,
                    attempt=attempt,
                    raw_output=collected_text,
                    parsed_output=parsed,
                ),
            )
            if parsed is not None:
                return parsed

            last_error = f"JSON 파싱 실패 (시도 {attempt})"
        except Exception as exc:
            last_error = f"에러 발생: {str(exc)}"

        time.sleep(1.0)

    print(f"    [오류] Qwen 요청 최종 실패: {last_error}")
    return None

def call_qwen_for_frame_sequence(
    model,
    sampling_params,
    frames: List[Any],
    timestamps: List[float],
    prompt: str,
    rel_path: str,
    stage: str,
    max_retries: int = MAX_CALL_RETRIES,
) -> Optional[Dict[str, Any]]:
    """여러 장의 샘플링 프레임을 순서대로 보여주고 첫 사고 프레임 인덱스를 추론합니다."""

    resized_frames = [resize_image_to_pixel_budget(frame, TIME_STAGE_MAX_PIXELS) for frame in frames]
    if resized_frames:
        original_pixels = frames[0].size[0] * frames[0].size[1]
        resized_pixels = resized_frames[0].size[0] * resized_frames[0].size[1]
        if resized_pixels != original_pixels:
            print(
                f"  -> [{stage}] vLLM 입력 축소: 첫 프레임 {frames[0].size[0]}x{frames[0].size[1]} "
                f"-> {resized_frames[0].size[0]}x{resized_frames[0].size[1]}"
            )

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for idx, (frame, ts) in enumerate(zip(resized_frames, timestamps)):
        content.append({"type": "text", "text": f"Frame {idx} at {ts:.3f} seconds."})
        content.append({"type": "image", "image": frame})

    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "Return exactly one JSON object. No markdown. No code fences. Ground every answer in the provided frames."
                    if DEBUG_VISIBLE_EVIDENCE
                    else "Respond with JSON only. No reasoning. No explanation. /no_think"
                ),
            }],
        },
        {
            "role": "user",
            "content": content,
        },
    ]
    return run_vllm_request(model, sampling_params, messages, rel_path, stage, max_retries=max_retries)

def call_qwen_for_media(
    model,
    sampling_params,
    media_type: str,
    media_path: str,
    prompt: str,
    rel_path: str,
    stage: str,
    max_retries: int = MAX_CALL_RETRIES,
    fps: float = VIDEO_FPS,
) -> Optional[Dict[str, Any]]:
    """VLM 모델을 호출하여 비디오 또는 이미지 분석 결과를 받아옵니다."""
    content: List[Dict[str, Any]] = []

    if media_type == "video":
        sampled_video = load_video_frames_for_vlm(media_path, sample_fps=fps)
        if not sampled_video:
            print(f"  -> [{stage}] 비디오 샘플링 실패: {media_path}")
            return None
        print(
            f"  -> [{stage}] OpenCV 샘플링: {sampled_video['num_frames']}프레임 "
            f"(sample_fps={sampled_video['sample_fps']:.2f}, raw_fps={sampled_video['raw_fps']:.2f})"
        )
        content.append({"type": "text", "text": prompt})
        for idx, (frame, ts) in enumerate(zip(sampled_video["frames"], sampled_video["timestamps"])):
            content.append({"type": "text", "text": f"Frame {idx} at {ts:.3f} seconds."})
            content.append({"type": "image", "image": frame})
    elif media_type == "image":
        with Image.open(media_path) as image:
            content.append({"type": "image", "image": image.convert("RGB")})
        content.append({"type": "text", "text": prompt})
    else:
        print(f"  -> [{stage}] 지원하지 않는 media_type: {media_type}")
        return None

    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "Return exactly one JSON object. No markdown. No code fences. Ground every answer in the provided media."
                    if DEBUG_VISIBLE_EVIDENCE
                    else "Respond with JSON only. No reasoning. No explanation. /no_think"
                ),
            }],
        },
        {
            "role": "user",
            "content": content,
        }
    ]

    return run_vllm_request(model, sampling_params, messages, rel_path, stage, max_retries=max_retries)

def normalize_metadata(row: Dict[str, str]) -> Dict[str, str]:
    """메타데이터 CSV의 열 이름을 표준 형식으로 정규화합니다."""
    row = dict(row)
    # 가끔 파일에 scene_layoutm 등으로 오타가 있는 경우를 대비
    if "scene_layout" not in row and "scene_layoutm" in row:
        row["scene_layout"] = row["scene_layoutm"]
    return row

def load_all_metadata(csv_path: str):
    """CSV 파일로부터 모든 메타데이터를 로드하는 제너레이터입니다."""
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield normalize_metadata(row)

def append_prediction_row(csv_path: str, row: Dict[str, Any], fieldnames: List[str]) -> None:
    """영상 하나 처리 직후 submission.csv에 즉시 append 저장합니다."""
    file_exists = os.path.exists(csv_path)
    needs_header = (not file_exists) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def strip_order_prefix(filename: str) -> str:
    match = re.match(r"^\d+_(.+)$", filename)
    return match.group(1) if match else filename

def build_video_lookup(video_dir: str) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for name in os.listdir(video_dir):
        full_path = os.path.abspath(os.path.join(video_dir, name))
        if not os.path.isfile(full_path):
            continue

        lookup.setdefault(name, full_path)
        stripped_name = strip_order_prefix(name)
        lookup.setdefault(stripped_name, full_path)
    return lookup

def resolve_video_path(rel_path: Optional[str], video_lookup: Dict[str, str]) -> Optional[str]:
    if not rel_path:
        return None
    return video_lookup.get(os.path.basename(rel_path))

# --- VER 2 함수 및 로직 추가 ---

def system_prompt() -> str:
    if DEBUG_VISIBLE_EVIDENCE:
        return (
            "Return exactly one JSON object. No markdown. No code fences. "
            "Ground every answer in the provided frames."
        )
    return "Respond with JSON only. No reasoning. No explanation. /no_think"


def build_time_prompt_v2(
    metadata: Dict[str, str],
    timestamps: Sequence[float],
    window_note: str,
    failed_indices: Optional[List[int]] = None,
) -> str:
    retry_note = ""
    if failed_indices:
        retry_note = (
            "\n\n[CRITICAL NOTE]\n"
            f"You previously selected frame indices {failed_indices}, but those were INCORRECT. "
            "Do not output those frame indices again. Choose a different frame index."
        )
    frame_catalog = ", ".join(f"{idx}: {ts:.3f}s" for idx, ts in enumerate(timestamps))
    return f"""{retry_note}
You are an expert traffic accident analyst reviewing chronological CCTV frames.

Goal:
- Identify the FIRST frame where the accident has genuinely started.

Decision rules:
1. Prefer the earliest frame where first physical contact is visible, or where the impact has unmistakably just started.
2. If the first impact happens between sampled frames, choose the earliest sampled frame that clearly shows the accident has already started.
3. Do not choose a later aftermath frame just because damage, debris, spin, or a final rest position is easier to see there.
4. A later strike against a guardrail, barrier, pole, wall, curb, sign, or tree does NOT define the accident start if an earlier vehicle collision already started the crash.
5. If the first impact truly is with a fixed roadside object, choose the first frame where the vehicle first touches that object.
6. Do not anchor on the most dramatic frame; be precise about the first impact.

Window note:
- {window_note}

Video metadata:
- region: {metadata.get("region", "")}
- scene_layout: {metadata.get("scene_layout", "")}
- weather: {metadata.get("weather", "")}
- day_time: {metadata.get("day_time", "")}
- quality (before enhancement): {metadata.get("quality", "")}
- duration (seconds): {metadata.get("duration", "")}
- no_frames: {metadata.get("no_frames", "")}
- frame_height: {metadata.get("height", "")}
- frame_width: {metadata.get("width", "")}

The frames are shown in chronological order.
The exact timestamp of each sampled frame is:
{frame_catalog}

Critical output rules:
- The FINAL output MUST be exactly one JSON object.
- No markdown.
- No bullet points.
- No code block.
- No text before JSON.
- No text after JSON.
- Required keys:
  "frame_index", "confidence", "evidence"
- "frame_index" must be an integer index from the provided frame list.
- "confidence" must be a float between 0.0 and 1.0.
- "evidence" must be a short summary of visible cues only, max 2 sentences.

Output format:
{{
  "frame_index": <int>,
  "confidence": <float>,
  "evidence": "<brief visible evidence>"
}}
""".strip()

def build_location_prompt_v2(metadata: Dict[str, str], accident_time: float, crop_hint: str) -> str:
    return f"""
You are an expert traffic accident analyst localizing the primary collision point.

You will receive images in this order:
1. A context frame before the target frame.
2. The TARGET frame, which is the first clear accident frame.
3. A context frame after the target frame.
4. A tight zoom crop from the TARGET frame.
5. A wider context crop from the TARGET frame.

Rules:
- Output coordinates must refer to image 2, the full TARGET frame.
- Use the context frames to resolve blur or occlusion.
- Output the center of the actual first-contact region, not the center of a whole vehicle.
- If the target frame is slightly after first contact due to sampling, estimate the first-contact region rather than a later resting position.
- The crops are only focus aids and together cover approximately {crop_hint} in the target frame.

Video metadata:
- region: {metadata.get("region", "")}
- scene_layout: {metadata.get("scene_layout", "")}
- weather: {metadata.get("weather", "")}
- day_time: {metadata.get("day_time", "")}
- quality (before enhancement): {metadata.get("quality", "")}
- duration (seconds): {metadata.get("duration", "")}
- no_frames: {metadata.get("no_frames", "")}
- frame_height: {metadata.get("height", "")}
- frame_width: {metadata.get("width", "")}
- target accident_time: {accident_time:.3f}s

Critical output rules:
- The FINAL output MUST be exactly one JSON object.
- No markdown.
- No bullet points.
- No code block.
- No text before JSON.
- No text after JSON.
- Required keys:
  "center_x", "center_y", "confidence", "evidence"
- "center_x" and "center_y" must be normalized to the full TARGET frame.
- "confidence" must be a float between 0.0 and 1.0.
- "evidence" must be a short summary of visible cues only, max 2 sentences.

Output format:
{{
  "center_x": <float>,
  "center_y": <float>,
  "confidence": <float>,
  "evidence": "<brief visible evidence>"
}}
""".strip()

def build_type_prompt_v2(metadata: Dict[str, str], accident_time: float, crop_hint: str) -> str:
    return f"""
You are an expert traffic accident analyst classifying the collision geometry at first impact.

You will receive a short chronological sequence around the first impact, followed by tight and wide crops near the collision area.
Classify the accident at the moment of first impact, not later secondary impacts or the final resting position.

Definitions:
- rear-end: one vehicle strikes the rear of another traveling in roughly the same direction.
- head-on: two vehicles traveling in opposite directions collide front-to-front.
- sideswipe: side-to-side contact while vehicles move in roughly the same direction and overlap laterally.
- t-bone: the front of one vehicle strikes the side of another at a strong crossing angle.
- single: one vehicle's first impact is with a fixed roadside object or infrastructure, with no other vehicle struck first.

Rules:
- Focus on the FIRST impact partner visible in the sequence.
- Ignore later secondary impacts with barriers or roadside objects if another collision already started earlier.
- A stopped or parked vehicle is still a vehicle, not a fixed roadside object.
- Use the crops near {crop_hint} only as visual aids; classify using the full sequence.
- If uncertain, choose the single best type.

Video metadata:
- region: {metadata.get("region", "")}
- scene_layout: {metadata.get("scene_layout", "")}
- weather: {metadata.get("weather", "")}
- day_time: {metadata.get("day_time", "")}
- quality (before enhancement): {metadata.get("quality", "")}
- duration (seconds): {metadata.get("duration", "")}
- no_frames: {metadata.get("no_frames", "")}
- frame_height: {metadata.get("height", "")}
- frame_width: {metadata.get("width", "")}
- target accident_time: {accident_time:.3f}s

Critical output rules:
- The FINAL output MUST be exactly one JSON object.
- No markdown.
- No bullet points.
- No code block.
- No text before JSON.
- No text after JSON.
- Required keys:
  "type", "confidence", "evidence"
- "type" must be exactly one of: rear-end, head-on, sideswipe, t-bone, single.
- "confidence" must be a float between 0.0 and 1.0.
- "evidence" must be a short summary of visible cues only, max 2 sentences.

Output format:
{{
  "type": "<rear-end|head-on|sideswipe|t-bone|single>",
  "confidence": <float>,
  "evidence": "<brief visible evidence>"
}}
""".strip()


def build_partner_verification_prompt_v2(metadata: Dict[str, str], accident_time: float, crop_hint: str) -> str:
    return f"""
You are verifying only the identity of the FIRST impact partner in a traffic accident.

Decide whether the first impact partner is:
- vehicle: another vehicle, including a stopped or parked vehicle.
- fixed_object: fixed roadside infrastructure or objects such as a guardrail, pole, barrier, wall, curb, sign, or tree.

Rules:
- Focus only on the FIRST impact partner, not later secondary impacts.
- If a vehicle later rests against a barrier after an earlier vehicle collision, the answer is vehicle.
- Debris, glare, lane markings, shadows, and reflections are not impact partners.
- Use the full sequence first, and use the crops near {crop_hint} only as visual aids.

Video metadata:
- region: {metadata.get("region", "")}
- scene_layout: {metadata.get("scene_layout", "")}
- weather: {metadata.get("weather", "")}
- day_time: {metadata.get("day_time", "")}
- quality (before enhancement): {metadata.get("quality", "")}
- duration (seconds): {metadata.get("duration", "")}
- no_frames: {metadata.get("no_frames", "")}
- frame_height: {metadata.get("height", "")}
- frame_width: {metadata.get("width", "")}
- target accident_time: {accident_time:.3f}s

Critical output rules:
- The FINAL output MUST be exactly one JSON object.
- No markdown.
- No bullet points.
- No code block.
- No text before JSON.
- No text after JSON.
- Required keys:
  "collision_partner", "confidence", "evidence"
- "collision_partner" must be exactly one of: vehicle, fixed_object.
- "confidence" must be a float between 0.0 and 1.0.
- "evidence" must be a short summary of visible cues only, max 2 sentences.

Output format:
{{
  "collision_partner": "<vehicle|fixed_object>",
  "confidence": <float>,
  "evidence": "<brief visible evidence>"
}}
""".strip()

def reduce_evenly(values: Sequence[Any], max_items: int) -> List[Any]:
    if max_items <= 0:
        return []
    if len(values) <= max_items:
        return list(values)
    scale = (len(values) - 1) / max(max_items - 1, 1)
    reduced = [values[min(round(i * scale), len(values) - 1)] for i in range(max_items)]
    deduped: List[Any] = []
    for value in reduced:
        if not deduped or value != deduped[-1]:
            deduped.append(value)
    return deduped


def get_video_info(video_path: str, meta: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 and meta and meta.get("duration") and meta.get("no_frames"):
        try:
            fps = float(meta["no_frames"]) / float(meta["duration"])
        except Exception:
            fps = 0.0
    if fps <= 0 or total_frames <= 0:
        return None
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration": max(0.0, (total_frames - 1) / fps),
        "width": width,
        "height": height,
    }


def build_sample_times(start_time: float, end_time: float, sample_fps: float, max_frames: int) -> List[float]:
    start_time = max(0.0, float(start_time))
    end_time = max(start_time, float(end_time))
    if max_frames <= 1 or end_time <= start_time:
        return [start_time]
    step = 1.0 / max(sample_fps, 0.1)
    times: List[float] = []
    current = start_time
    while current < end_time + 1e-9:
        times.append(round(current, 6))
        current += step
    if not times or abs(times[-1] - end_time) > 1e-6:
        times.append(round(end_time, 6))
    return reduce_evenly(times, max_frames)


def capture_frames_by_indices(
    video_path: str,
    frame_indices: Sequence[int],
    meta: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    info = get_video_info(video_path, meta)
    if info is None:
        return None
    normalized_indices: List[int] = []
    for raw_index in frame_indices:
        frame_index = max(0, min(int(raw_index), info["total_frames"] - 1))
        if not normalized_indices or frame_index != normalized_indices[-1]:
            normalized_indices.append(frame_index)
    if not normalized_indices:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    items: List[Dict[str, Any]] = []
    for frame_index in normalized_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        items.append(
            {
                "frame_index": frame_index,
                "timestamp": frame_index / info["fps"],
                "bgr": frame.copy(),
                "image": Image.fromarray(rgb),
            }
        )
    cap.release()
    if not items:
        return None
    return {
        **info,
        "items": items,
        "frames": [item["image"] for item in items],
        "timestamps": [item["timestamp"] for item in items],
        "frame_indices": [item["frame_index"] for item in items],
        "num_frames": len(items),
    }


def sample_video_window(
    video_path: str,
    meta: Dict[str, str],
    start_time: float,
    end_time: float,
    sample_fps: float,
    max_frames: int,
) -> Optional[Dict[str, Any]]:
    info = get_video_info(video_path, meta)
    if info is None:
        return None
    target_times = build_sample_times(start_time, end_time, sample_fps, max_frames)
    frame_indices = [round(ts * info["fps"]) for ts in target_times]
    return capture_frames_by_indices(video_path, frame_indices, meta)


def capture_context_by_offsets(
    video_path: str,
    center_frame_index: int,
    offsets: Sequence[int],
    meta: Dict[str, str],
) -> Optional[List[Dict[str, Any]]]:
    info = get_video_info(video_path, meta)
    if info is None:
        return None
    indices = [max(0, min(center_frame_index + offset, info["total_frames"] - 1)) for offset in offsets]
    captured = capture_frames_by_indices(video_path, indices, meta)
    if captured is None:
        return None
    by_index = {item["frame_index"]: item for item in captured["items"]}
    items: List[Dict[str, Any]] = []
    for offset, index in zip(offsets, indices):
        item = by_index.get(index)
        if item is not None:
            items.append({"offset": int(offset), **item})
    return items or None


def call_vllm_for_images(
    model,
    sampling_params,
    prompt: str,
    labeled_images: Sequence[Tuple[str, Image.Image]],
    rel_path: str,
    stage: str,
    max_pixels: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for label, image in labeled_images:
        prepared = resize_image_to_pixel_budget(image, max_pixels) if max_pixels else image
        content.append({"type": "text", "text": label})
        content.append({"type": "image", "image": prepared})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt()}]},
        {"role": "user", "content": content},
    ]
    return run_vllm_request(model, sampling_params, messages, rel_path, stage)


def pixel_box_to_norm(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 / width, y1 / height, x2 / width, y2 / height)


def box_to_text(box: Tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = pixel_box_to_norm(box, max(width, 1), max(height, 1))
    return f"x=[{x1:.3f}, {x2:.3f}], y=[{y1:.3f}, {y2:.3f}]"


def clamp_box(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def build_center_box(width: int, height: int, center_x: float, center_y: float, fraction: float) -> Tuple[int, int, int, int]:
    cx = min(max(center_x, 0.0), 1.0) * max(width - 1, 1)
    cy = min(max(center_y, 0.0), 1.0) * max(height - 1, 1)
    half_w = max(1, int(round(width * fraction / 2.0)))
    half_h = max(1, int(round(height * fraction / 2.0)))
    return clamp_box(
        (
            int(round(cx)) - half_w,
            int(round(cy)) - half_h,
            int(round(cx)) + half_w,
            int(round(cy)) + half_h,
        ),
        width,
        height,
    )


def expand_box(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
    scale: float,
    min_fraction: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    target_width = max(int(width * min_fraction), int(box_width * scale))
    target_height = max(int(height * min_fraction), int(box_height * scale))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_w = target_width / 2.0
    half_h = target_height / 2.0
    return clamp_box(
        (
            int(round(cx - half_w)),
            int(round(cy - half_h)),
            int(round(cx + half_w)),
            int(round(cy + half_h)),
        ),
        width,
        height,
    )


def merge_boxes(
    left: Tuple[int, int, int, int],
    right: Tuple[int, int, int, int],
    width: int,
    height: int,
    min_fraction: float,
) -> Tuple[int, int, int, int]:
    merged = (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )
    return expand_box(merged, width, height, scale=1.1, min_fraction=min_fraction)


def estimate_motion_box(
    prev_bgr: Optional[Any],
    curr_bgr: Optional[Any],
    next_bgr: Optional[Any],
    min_fraction: float,
) -> Optional[Tuple[int, int, int, int]]:
    if prev_bgr is None or curr_bgr is None or next_bgr is None:
        return None
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)
    motion = cv2.absdiff(prev_gray, curr_gray)
    motion = cv2.addWeighted(motion, 0.5, cv2.absdiff(curr_gray, next_gray), 0.5, 0.0)
    motion = cv2.GaussianBlur(motion, (5, 5), 0)
    _, mask = cv2.threshold(motion, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    height, width = curr_gray.shape[:2]
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < width * height * 0.0015:
        return None
    x, y, box_w, box_h = cv2.boundingRect(contour)
    return expand_box((x, y, x + box_w, y + box_h), width, height, scale=1.7, min_fraction=min_fraction)


def validate_type_prediction_v2(result: Dict[str, Any]) -> Optional[str]:
    raw = str(result.get("type", "")).strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "rearend": "rear-end",
        "rear-end": "rear-end",
        "headon": "head-on",
        "head-on": "head-on",
        "side-swipe": "sideswipe",
        "sidewipe": "sideswipe",
        "sideswipe": "sideswipe",
        "tbone": "t-bone",
        "t-bone": "t-bone",
        "single": "single",
    }
    accident_type = aliases.get(raw, raw)
    if accident_type not in VALID_TYPES:
        return None
    return accident_type


def validate_collision_partner_prediction(result: Dict[str, Any]) -> Optional[str]:
    raw = str(result.get("collision_partner", "")).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "vehicle": "vehicle",
        "other_vehicle": "vehicle",
        "parked_vehicle": "vehicle",
        "stopped_vehicle": "vehicle",
        "fixed_object": "fixed_object",
        "fixed": "fixed_object",
        "object": "fixed_object",
        "roadside_object": "fixed_object",
        "stationary_object": "fixed_object",
        "infrastructure": "fixed_object",
    }
    return aliases.get(raw)


def normalize_text(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def extract_visible_evidence_text(result: Optional[Dict[str, Any]]) -> str:
    if not result:
        return ""
    parts: List[str] = []
    evidence = maybe_get_evidence(result)
    if evidence:
        parts.append(evidence)
    raw_type = result.get("type")
    if raw_type:
        parts.append(str(raw_type))
    raw_partner = result.get("collision_partner")
    if raw_partner:
        parts.append(str(raw_partner))
    return normalize_text(" ".join(parts))


def count_keyword_hits(text: str, keywords: Sequence[str]) -> int:
    normalized = normalize_text(text)
    hits = 0
    for keyword in keywords:
        needle = normalize_text(keyword)
        if not needle:
            continue
        if " " in needle:
            if needle in normalized:
                hits += 1
        elif re.search(rf"\b{re.escape(needle)}\b", normalized):
            hits += 1
    return hits


def looks_like_aftermath_result(result: Optional[Dict[str, Any]]) -> bool:
    return count_keyword_hits(extract_visible_evidence_text(result), AFTERMATH_KEYWORDS) > 0


def should_run_partner_verification(
    time_result: Optional[Dict[str, Any]],
    location_result: Optional[Dict[str, Any]],
    type_result: Optional[Dict[str, Any]],
    predicted_type: str,
) -> bool:
    if predicted_type == "single":
        return True
    combined = " ".join(
        part
        for part in (
            extract_visible_evidence_text(time_result),
            extract_visible_evidence_text(location_result),
            extract_visible_evidence_text(type_result),
        )
        if part
    )
    fixed_hits = count_keyword_hits(combined, FIXED_OBJECT_KEYWORDS)
    vehicle_hits = count_keyword_hits(combined, VEHICLE_KEYWORDS)
    if fixed_hits >= 2:
        return True
    if predicted_type in {"rear-end", "sideswipe"} and fixed_hits >= 1:
        return True
    return fixed_hits >= 1 and vehicle_hits == 0

def predict_accident_time_ver2(
    model,
    sampling_params,
    video_path: str,
    meta: Dict[str, str],
    rel_path: str,
) -> Optional[Dict[str, Any]]:
    info = get_video_info(video_path, meta)
    if info is None:
        return None
    coarse_sample = load_video_frames_for_vlm(video_path, sample_fps=COARSE_VIDEO_FPS)
    if not coarse_sample:
        return None

    print(
        f"  -> [time] coarse 샘플링: {coarse_sample['num_frames']}프레임 "
        f"(sample_fps={coarse_sample['sample_fps']:.2f}, raw_fps={coarse_sample['raw_fps']:.2f})"
    )

    failed_indices: List[int] = []
    coarse_result = None
    coarse_idx = None
    for _ in range(MAX_TIME_RETRIES):
        coarse_result = call_vllm_for_images(
            model,
            sampling_params,
            build_time_prompt_v2(
                meta,
                coarse_sample["timestamps"],
                window_note="This is a coarse scan over the whole video. Find the rough first-impact neighborhood.",
                failed_indices=failed_indices,
            ),
            [
                (f"Frame {idx} at {ts:.3f} seconds.", frame)
                for idx, (frame, ts) in enumerate(zip(coarse_sample["frames"], coarse_sample["timestamps"]))
            ],
            rel_path,
            "time_coarse",
            max_pixels=TIME_COARSE_MAX_PIXELS,
        )
        if not coarse_result:
            continue
        coarse_idx = validate_frame_index_prediction(coarse_result, coarse_sample["num_frames"])
        if coarse_idx is not None and coarse_idx not in failed_indices:
            break
        try:
            failed_indices.append(int(coarse_result.get("frame_index")))
        except Exception:
            pass
    if coarse_idx is None or coarse_result is None:
        return None

    coarse_time = float(coarse_sample["timestamps"][coarse_idx])
    left_time = float(coarse_sample["timestamps"][max(coarse_idx - 1, 0)])
    right_time = float(coarse_sample["timestamps"][min(coarse_idx + 1, coarse_sample["num_frames"] - 1)])
    coarse_span = max(0.75, right_time - left_time)

    refine_start = max(0.0, min(left_time, coarse_time) - 0.35 * coarse_span)
    refine_end = min(info["duration"], max(right_time, coarse_time) + 0.35 * coarse_span)
    refine_sample = sample_video_window(video_path, meta, refine_start, refine_end, TIME_REFINE_FPS, TIME_REFINE_MAX_FRAMES)

    refined_result = coarse_result
    refined_time = coarse_time
    refined_sample = coarse_sample
    refined_idx = coarse_idx
    if refine_sample and refine_sample["num_frames"] >= 3:
        print(
            f"  -> [time] refine 샘플링: {refine_sample['num_frames']}프레임 "
            f"(window={refine_start:.3f}s~{refine_end:.3f}s, fps={TIME_REFINE_FPS:.2f})"
        )
        maybe_refined = call_vllm_for_images(
            model,
            sampling_params,
            build_time_prompt_v2(
                meta,
                refine_sample["timestamps"],
                window_note=f"This is a narrow refinement window around the coarse candidate near {coarse_time:.3f}s.",
            ),
            [
                (f"Frame {idx} at {ts:.3f} seconds.", frame)
                for idx, (frame, ts) in enumerate(zip(refine_sample["frames"], refine_sample["timestamps"]))
            ],
            rel_path,
            "time_refine",
            max_pixels=TIME_REFINE_MAX_PIXELS,
        )
        maybe_refined_idx = validate_frame_index_prediction(maybe_refined, refine_sample["num_frames"]) if maybe_refined else None
        if maybe_refined_idx is not None:
            refined_candidate_aftermath = looks_like_aftermath_result(maybe_refined)
            coarse_aftermath = looks_like_aftermath_result(coarse_result)
            if not refined_candidate_aftermath or coarse_aftermath:
                refined_result = maybe_refined
                refined_time = float(refine_sample["timestamps"][maybe_refined_idx])
                refined_sample = refine_sample
                refined_idx = maybe_refined_idx

    if refined_sample["num_frames"] > 1:
        prev_refined = float(refined_sample["timestamps"][max(refined_idx - 1, 0)])
        next_refined = float(refined_sample["timestamps"][min(refined_idx + 1, refined_sample["num_frames"] - 1)])
        boundary_start = max(0.0, prev_refined - 0.10)
        boundary_end = min(info["duration"], next_refined + 0.10)
    else:
        boundary_start = max(0.0, refined_time - 0.50)
        boundary_end = min(info["duration"], refined_time + 0.50)
    boundary_sample = sample_video_window(video_path, meta, boundary_start, boundary_end, TIME_BOUNDARY_FPS, TIME_BOUNDARY_MAX_FRAMES)

    final_result = refined_result
    final_time = refined_time
    selected_stage = "time_refine" if refined_sample is refine_sample else "time_coarse"
    if boundary_sample and boundary_sample["num_frames"] >= 3:
        print(
            f"  -> [time] boundary 샘플링: {boundary_sample['num_frames']}프레임 "
            f"(window={boundary_start:.3f}s~{boundary_end:.3f}s, fps={TIME_BOUNDARY_FPS:.2f})"
        )
        maybe_boundary = call_vllm_for_images(
            model,
            sampling_params,
            build_time_prompt_v2(
                meta,
                boundary_sample["timestamps"],
                window_note=f"This is the final boundary check around {refined_time:.3f}s. Be extremely precise.",
            ),
            [
                (f"Frame {idx} at {ts:.3f} seconds.", frame)
                for idx, (frame, ts) in enumerate(zip(boundary_sample["frames"], boundary_sample["timestamps"]))
            ],
            rel_path,
            "time_boundary",
            max_pixels=TIME_BOUNDARY_MAX_PIXELS,
        )
        maybe_boundary_idx = validate_frame_index_prediction(maybe_boundary, boundary_sample["num_frames"]) if maybe_boundary else None
        if maybe_boundary_idx is not None:
            boundary_aftermath = looks_like_aftermath_result(maybe_boundary)
            refined_aftermath = looks_like_aftermath_result(refined_result)
            if not boundary_aftermath or refined_aftermath:
                final_result = maybe_boundary
                final_time = float(boundary_sample["timestamps"][maybe_boundary_idx])
                selected_stage = "time_boundary"

    extracted = extract_frame_at_time(video_path, final_time, meta, fps_override=info["fps"])
    if not extracted:
        return None
    final_time = extracted["frame_index"] / extracted["fps"]
    append_raw_log(
        RAW_LOG_PATH,
        {
            "path": rel_path,
            "stage": "time_summary",
            "coarse_time": round(coarse_time, 4),
            "refined_time": round(refined_time, 4),
            "final_accident_time": round(final_time, 4),
            "final_accident_frame": int(extracted["frame_index"]),
            "selected_stage": selected_stage,
        },
    )
    return {
        "accident_time": final_time,
        "time_frame_index": int(extracted["frame_index"]),
        "raw_time_json": final_result,
        "extracted": extracted,
    }


def predict_location_ver2(
    model,
    sampling_params,
    video_path: str,
    meta: Dict[str, str],
    rel_path: str,
    extracted: Dict[str, Any],
    accident_time: float,
) -> Optional[Dict[str, Any]]:
    context = capture_context_by_offsets(video_path, extracted["frame_index"], LOCATION_CONTEXT_OFFSETS, meta)
    if not context or len(context) < 3:
        return None
    target_item = next((item for item in context if item["offset"] == 0), None)
    if target_item is None:
        return None
    height, width = target_item["bgr"].shape[:2]
    motion_box = estimate_motion_box(context[0]["bgr"], target_item["bgr"], context[-1]["bgr"], LOCATION_MIN_FOCUS_FRACTION)
    if motion_box is None:
        motion_box = build_center_box(width, height, 0.5, 0.5, LOCATION_MIN_FOCUS_FRACTION)
    wide_box = expand_box(motion_box, width, height, scale=1.45, min_fraction=LOCATION_WIDE_MIN_FOCUS_FRACTION)
    crop_hint = (
        f"tight crop {box_to_text(motion_box, width, height)}; "
        f"wide crop {box_to_text(wide_box, width, height)}"
    )
    raw_loc = call_vllm_for_images(
        model,
        sampling_params,
        build_location_prompt_v2(meta, accident_time, crop_hint),
        [
            ("Image 1/5: Context frame before the target frame.", context[0]["image"]),
            ("Image 2/5: TARGET frame. Output coordinates must refer to this full frame.", target_item["image"]),
            ("Image 3/5: Context frame after the target frame.", context[-1]["image"]),
            ("Image 4/5: Tight zoom crop from the TARGET frame.", target_item["image"].crop(motion_box)),
            ("Image 5/5: Wider local road context crop from the TARGET frame.", target_item["image"].crop(wide_box)),
        ],
        rel_path,
        "location",
    )
    if not raw_loc:
        return None
    loc = validate_location_prediction(raw_loc)
    if not loc:
        return None
    return {"loc": loc, "raw_loc": raw_loc, "focus_box": motion_box, "wide_focus_box": wide_box}

def build_type_offsets(fps: float) -> List[int]:
    near = max(1, int(round(fps * 0.25)))
    far = max(near + 1, int(round(fps * 0.55)))
    return reduce_evenly([-far, -near, 0, near, far], 5)


def predict_type_ver2(
    model,
    sampling_params,
    video_path: str,
    meta: Dict[str, str],
    rel_path: str,
    extracted: Dict[str, Any],
    accident_time: float,
    loc: Dict[str, float],
    location_focus_box: Optional[Tuple[int, int, int, int]],
    time_raw_json: Optional[Dict[str, Any]],
    location_raw_json: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    context = capture_context_by_offsets(video_path, extracted["frame_index"], build_type_offsets(float(extracted["fps"])), meta)
    if not context:
        return None
    target_item = next((item for item in context if item["offset"] == 0), None)
    if target_item is None:
        return None
    height, width = target_item["bgr"].shape[:2]
    center_box = build_center_box(width, height, loc["center_x"], loc["center_y"], TYPE_MIN_FOCUS_FRACTION)
    focus_box = merge_boxes(center_box, location_focus_box, width, height, TYPE_MIN_FOCUS_FRACTION) if location_focus_box else center_box
    wide_focus_box = expand_box(focus_box, width, height, scale=1.35, min_fraction=TYPE_WIDE_MIN_FOCUS_FRACTION)
    crop_hint = (
        f"tight crop {box_to_text(focus_box, width, height)}; "
        f"wide crop {box_to_text(wide_focus_box, width, height)}"
    )
    next_item = next((item for item in context if item["offset"] > 0), target_item)
    labeled_images: List[Tuple[str, Image.Image]] = []
    total = len(context)
    for idx, item in enumerate(context, start=1):
        if item["offset"] == 0:
            label = f"Sequence frame {idx}/{total}: TARGET first-impact frame."
        elif item["offset"] < 0:
            label = f"Sequence frame {idx}/{total}: Context frame before target."
        else:
            label = f"Sequence frame {idx}/{total}: Context frame after target."
        labeled_images.append((label, item["image"]))
    labeled_images.append(("Crop A: TARGET frame tight focus near the collision point.", target_item["image"].crop(focus_box)))
    labeled_images.append(("Crop B: TARGET frame wider local road context.", target_item["image"].crop(wide_focus_box)))
    labeled_images.append(("Crop C: Near-post-impact frame with the same wider region.", next_item["image"].crop(wide_focus_box)))

    raw_type = call_vllm_for_images(
        model,
        sampling_params,
        build_type_prompt_v2(meta, accident_time, crop_hint),
        labeled_images,
        rel_path,
        "type",
    )
    if not raw_type:
        return None

    predicted_type = validate_type_prediction_v2(raw_type)
    if not predicted_type:
        return None

    collision_partner = "fixed_object" if predicted_type == "single" else None
    raw_partner = None
    if should_run_partner_verification(time_raw_json, location_raw_json, raw_type, predicted_type):
        raw_partner = call_vllm_for_images(
            model,
            sampling_params,
            build_partner_verification_prompt_v2(meta, accident_time, crop_hint),
            labeled_images,
            rel_path,
            "partner_verify",
        )
        partner = validate_collision_partner_prediction(raw_partner)
        partner_confidence = maybe_get_confidence(raw_partner) if raw_partner else 0.0
        if partner == "fixed_object" and partner_confidence >= PARTNER_VERIFY_MIN_CONFIDENCE:
            predicted_type = "single"
            collision_partner = "fixed_object"
        elif partner == "vehicle" and partner_confidence >= PARTNER_VERIFY_MIN_CONFIDENCE:
            if predicted_type == "single":
                # 단독 사고로 예측했으나 검증 결과 차대차 사고인 경우, 가장 무난한 sideswipe로 보정하거나
                # 혹은 파트너 정보만이라도 반영합니다. 여기서는 일관성을 위해 type도 보정합니다.
                predicted_type = "sideswipe" 
            collision_partner = "vehicle"
        elif partner in {"vehicle", "fixed_object"}:
            # 신뢰도가 낮더라도 일단 정보가 있으면 업데이트 (단, type 강제 변경은 하지 않음)
            collision_partner = partner

    return {
        "type": predicted_type,
        "collision_partner": collision_partner,
        "raw_type": raw_type,
        "raw_partner": raw_partner,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vLLM Inference in Parallel over Multiple GPUs (Ver2)")
    parser.add_argument("--gpu", type=int, required=True, help="작업을 할당할 GPU ID (예: 0 번 혹은 1 번)")
    parser.add_argument("--num_gpus", type=int, default=2, help="총 할당된 GPU 수")
    args = parser.parse_args()

    # 스크립트가 실행될 때 입력받은 번호의 단일 GPU만 쳐다보도록 시스템을 격리합니다.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    total_run_start = PROGRAM_START_TIME
    init_start = time.perf_counter()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)
    # 기존 파일 삭제 로직은 안전을 위해 제외 (다른 병렬 워커가 지우면 안되므로)
    
    metadata_rows = list(load_all_metadata(METADATA_CSV_PATH))
    active_video_dir = TEST_VIDEO_DIR if TEST_MODE else VIDEO_DIR
    video_lookup = build_video_lookup(active_video_dir)
    if TEST_MODE:
        metadata_rows = [m for m in metadata_rows if resolve_video_path(m.get("path"), video_lookup)]
        if TEST_LIMIT > 0:
            metadata_rows = metadata_rows[:TEST_LIMIT]

    print(f"모델을 로드합니다: {MODEL_NAME}")
    comp_error = ensure_runtime_compatibility()
    if comp_error:
        print(f"환경 호환성 오류: {comp_error}")
        return

    model_load_start = time.perf_counter()
    model, sampling_params = load_vllm_model()
    model_load_elapsed = time.perf_counter() - model_load_start
    init_elapsed = time.perf_counter() - init_start
    print(f"[time] 모델 로드 완료: {format_elapsed(model_load_elapsed)}")
    print(f"[time] 초기 준비 완료(프로세스 시작 후): {format_elapsed(init_elapsed)}")

    prediction_fieldnames = ["path", "accident_time", "center_x", "center_y", "type"]
    existing_paths = set()
    if SKIP_EXISTING and os.path.exists(PREDICTION_PATH):
        with open(PREDICTION_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_paths.add(row["path"])

    remaining_metadata = []
    for meta in metadata_rows:
        rp = meta.get("path")
        if not rp or (SKIP_EXISTING and rp in existing_paths):
            continue
        remaining_metadata.append(meta)

    print(f"\n[GPU {args.gpu}] 남은 총 작업 수: {len(remaining_metadata)}개 (이 중 1/{args.num_gpus}를 담당합니다.)")

    for idx, meta in enumerate(remaining_metadata, start=0):
        if idx % args.num_gpus != args.gpu:
            continue
            
        rel_path = meta.get("path")
        abs_video_path = resolve_video_path(rel_path, video_lookup)
        if not abs_video_path:
            print(f"[경보] 메타데이터와 매칭되는 비디오 파일을 찾을 수 없음: {rel_path}")
            continue

        print(f"\n[{idx+1}/{len(remaining_metadata)}] 처리 중: {rel_path}")
        matched_name = os.path.basename(abs_video_path)
        original_name = os.path.basename(rel_path)
        if matched_name != original_name:
            print(f"  -> 매칭 파일: {matched_name}")

        video_start = time.perf_counter()
        video_outcome = "중도 종료"
        try:
            time_result = predict_accident_time_ver2(model, sampling_params, abs_video_path, meta, rel_path)
            if not time_result:
                video_outcome = "사고 시점 추론 실패"
                print(f"  -> [오류] {rel_path}의 사고 시점을 찾지 못했습니다. 건너뜁니다.")
                continue

            accident_time = float(time_result["accident_time"])
            extracted = time_result["extracted"]
            print(f"  -> 최종 사고 프레임: {time_result['time_frame_index']} (t={accident_time:.4f}초)")
            print_debug_payload("time", time_result["raw_time_json"])

            frame_filename = os.path.splitext(os.path.basename(rel_path))[0]
            frame_path = os.path.join(FRAME_DIR, f"{frame_filename}_t{accident_time:.3f}.jpg")
            cv2.imwrite(frame_path, extracted["frame"])

            location_result = predict_location_ver2(
                model,
                sampling_params,
                abs_video_path,
                meta,
                rel_path,
                extracted,
                accident_time,
            )
            if not location_result:
                video_outcome = "사고 위치 추론 실패"
                print("  -> [오류] 사고 위치 추론 실패.")
                continue
            loc = location_result["loc"]
            print(f"  -> 사고 위치: ({loc['center_x']:.4f}, {loc['center_y']:.4f})")
            print_debug_payload("location", location_result["raw_loc"])

            type_result = predict_type_ver2(
                model,
                sampling_params,
                abs_video_path,
                meta,
                rel_path,
                extracted,
                accident_time,
                loc,
                location_result["focus_box"],
                time_result["raw_time_json"],
                location_result["raw_loc"],
            )
            if not type_result:
                video_outcome = "사고 유형 추론 실패"
                print("  -> [오류] 사고 유형 추론 실패.")
                continue
            acc_type = type_result["type"]
            collision_partner = type_result.get("collision_partner")
            if collision_partner == "fixed_object":
                print(f"  -> 검증 결과: 고정 사물 충돌! (type={acc_type})")
            elif collision_partner == "vehicle":
                print(f"  -> 검증 결과: 타 차량 충돌! (type={acc_type})")
            
            print(f"  -> 최종 사고 유형: {acc_type}")
            print_debug_payload("type", type_result["raw_type"])
            if type_result.get("raw_partner"):
                print_debug_payload("partner_verify", type_result["raw_partner"])

            prediction_row = {
                "path": rel_path,
                "accident_time": accident_time,
                "center_x": loc["center_x"],
                "center_y": loc["center_y"],
                "type": acc_type,
            }
            append_prediction_row(PREDICTION_PATH, prediction_row, prediction_fieldnames)
            existing_paths.add(rel_path)
            video_outcome = "성공"
            print(f"  -> submission.csv 저장 완료: {rel_path}")
        finally:
            video_elapsed = time.perf_counter() - video_start
            print(
                f"  -> [time] 영상 전체 처리 시간: {format_elapsed(video_elapsed)} "
                f"(status={video_outcome})"
            )
            import torch
            torch.cuda.empty_cache()

    if os.path.exists(PREDICTION_PATH):
        print(f"\n모든 결과가 저장되었습니다: {PREDICTION_PATH}")
    print(f"[time] 전체 실행 시간: {format_elapsed(time.perf_counter() - total_run_start)}")

if __name__ == "__main__":
    main()
