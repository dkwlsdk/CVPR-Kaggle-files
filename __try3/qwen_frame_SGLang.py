#!/usr/bin/env python3
import base64
import importlib
import io
import os
import csv
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

import cv2
import torch
from PIL import Image
from transformers import PretrainedConfig

PROGRAM_START_TIME = time.perf_counter()

# --- GPU 환경 및 보안 설정 ---
# 진정한 SGLang의 위력(Tensor Parallelism)을 활용하기 위해 GPU 0, 1 두 대를 묶어 48GB VRAM으로 구동합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# --- 모델 및 생성 제어 하이퍼파라미터 ---
MODEL_NAME = "Qwen/Qwen3.5-9B"
# MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
# MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

OUTPUT_DIR = "/root/Desktop/workspace/ja/qwen_test_output_9B/06.SGLang_test"

# 모델이 생성할 최대 답변 길이 (무한 반복 방지 및 속도 향상을 위해 2048로 제한)
MAX_NEW_TOKENS = 2048
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
MAX_VIDEO_SAMPLED_FRAMES = 64  # 정확도를 위해 64프레임으로 복구 (GPU 2개를 사용하므로 OOM 발생 안 함)
TIME_STAGE_MAX_PIXELS = MIN_PIXELS

# --- 재시도 및 검증 로직 설정 ---
MAX_TIME_RETRIES = 2           # 사고 시점 예측 실패 시 최대 재시도 횟수
# 이전 예측값과 현재 예측값이 동일한지 판단할 오차 범위 (초 단위)
TIME_MATCH_THRESHOLD = 0.1
# 모델 호출 자체의 재시도 횟수 (JSON 파싱 실패 등 대비)
MAX_CALL_RETRIES = 2

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
TEST_LIMIT = 2        # 테스트 시 처리할 최대 영상 수
# SKIP_EXISTING = True
SKIP_EXISTING = False

# GPU 메모리 예약 공간 (MiB 단위)
CUDA_MEMORY_RESERVE_MIB = 256
SGLANG_CONTEXT_LENGTH = 32768
SGLANG_MEM_FRACTION_STATIC = 0.80
SGLANG_BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000").rstrip("/")
SGLANG_HTTP_TIMEOUT = float(os.environ.get("SGLANG_HTTP_TIMEOUT", "600"))
SGLANG_FORCE_HTTP = os.environ.get("SGLANG_FORCE_HTTP", "").strip().lower() in {"1", "true", "yes"}

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

def build_sglang_sampling_params() -> Dict[str, Any]:
    return {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "skip_special_tokens": True,
    }


def read_sglang_json(
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    method: str = "GET",
) -> Dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urlrequest.urlopen(req, timeout=SGLANG_HTTP_TIMEOUT) as response:
            raw = response.read().decode("utf-8")
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"연결 실패: {exc.reason}") from exc

    parsed = json.loads(raw) if raw else {}
    if isinstance(parsed, list):
        return parsed[0] if parsed else {}
    if isinstance(parsed, dict):
        return parsed
    raise RuntimeError(f"예상하지 못한 SGLang 응답 형식: {type(parsed)}")


def probe_sglang_server() -> Dict[str, Any]:
    last_error = None
    for endpoint in ("/model_info", "/get_model_info"):
        try:
            return read_sglang_json(f"{SGLANG_BASE_URL}{endpoint}")
        except Exception as exc:
            last_error = str(exc)
    raise RuntimeError(
        "SGLang 서버 정보를 읽지 못했습니다. "
        f"endpoint={SGLANG_BASE_URL}, error={last_error}"
    )


def load_sglang_model():
    """SGLang Engine 또는 SGLang 서버 연결 정보를 로드합니다."""
    sampling_params = build_sglang_sampling_params()

    if not SGLANG_FORCE_HTTP:
        try:
            sglang_mod = importlib.import_module("sglang")
            Engine = getattr(sglang_mod, "Engine")
        except ModuleNotFoundError:
            Engine = None

        if Engine is not None:
            tensor_parallel_size = infer_tensor_parallel_size()
            print(
                f"  -> SGLang Engine 로드: model={MODEL_NAME}, tp={tensor_parallel_size}, "
                f"context_length={SGLANG_CONTEXT_LENGTH}"
            )
            engine = Engine(
                model_path=MODEL_NAME,
                trust_remote_code=True,
                tp_size=tensor_parallel_size,
                context_length=SGLANG_CONTEXT_LENGTH,
                mem_fraction_static=SGLANG_MEM_FRACTION_STATIC,
            )
            return {"mode": "engine", "engine": engine, "model_name": MODEL_NAME}, sampling_params

    server_info = probe_sglang_server()
    served_model = str(server_info.get("model_path") or server_info.get("tokenizer_path") or MODEL_NAME)
    print(f"  -> SGLang HTTP 연결: model={served_model}, endpoint={SGLANG_BASE_URL}")
    return {"mode": "http", "base_url": SGLANG_BASE_URL, "model_name": served_model}, sampling_params


def close_sglang_model(model: Dict[str, Any]) -> None:
    if model.get("mode") != "engine":
        return
    engine = model.get("engine")
    if engine is None or not hasattr(engine, "shutdown"):
        return
    try:
        engine.shutdown()
    except Exception:
        pass

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
        if parsed and ("frame_index" in parsed or "center_x" in parsed or "type" in parsed):
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

def maybe_get_confidence(result: Dict[str, Any]) -> Optional[float]:
    raw_conf = result.get("confidence")
    if raw_conf is None:
        return None
    try:
        confidence = float(raw_conf)
    except (TypeError, ValueError):
        return None
    return min(max(confidence, 0.0), 1.0)

def maybe_get_evidence(result: Dict[str, Any]) -> Optional[str]:
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

def print_debug_payload(stage: str, result: Dict[str, Any]) -> None:
    if not DEBUG_VISIBLE_EVIDENCE:
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
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

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
    """메시지 리스트를 SGLang용 ChatML 프롬프트와 이미지 리스트로 변환합니다."""
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

def pil_image_to_sglang_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def messages_to_sglang_request(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt, images = render_chatml_prompt(messages)
    return {"prompt": prompt, "images": images}


def run_sglang_request(
    model,
    sampling_params,
    messages: List[Dict[str, Any]],
    rel_path: str,
    stage: str,
    max_retries: int = MAX_CALL_RETRIES,
) -> Optional[Dict[str, Any]]:
    """SGLang으로 단일 멀티모달 요청을 실행하고 JSON 응답을 파싱합니다."""
    request_payload = messages_to_sglang_request(messages)
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            if model.get("mode") == "engine":
                generate_kwargs: Dict[str, Any] = {
                    "prompt": request_payload["prompt"],
                    "sampling_params": sampling_params,
                }
                if request_payload["images"]:
                    generate_kwargs["image_data"] = request_payload["images"]
                response = model["engine"].generate(**generate_kwargs)
                collected_text = str(response.get("text", "")).strip() if isinstance(response, dict) else ""
            else:
                http_payload: Dict[str, Any] = {
                    "text": request_payload["prompt"],
                    "sampling_params": sampling_params,
                }
                if request_payload["images"]:
                    http_payload["image_data"] = [
                        pil_image_to_sglang_base64(image) for image in request_payload["images"]
                    ]
                response = read_sglang_json(
                    f"{model['base_url']}/generate",
                    payload=http_payload,
                    method="POST",
                )
                collected_text = str(response.get("text", "")).strip()

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
                f"  -> [{stage}] SGLang 입력 축소: 첫 프레임 {frames[0].size[0]}x{frames[0].size[1]} "
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
    return run_sglang_request(model, sampling_params, messages, rel_path, stage, max_retries=max_retries)

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

    return run_sglang_request(model, sampling_params, messages, rel_path, stage, max_retries=max_retries)

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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)

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

# --- 메인 실행 프로세스 ---

def main():
    total_run_start = PROGRAM_START_TIME
    init_start = time.perf_counter()

    # 출력 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)

    # 이전에 기록된 원시 로그 삭제 (새로 시작할 경우)
    if not SKIP_EXISTING and os.path.exists(RAW_LOG_PATH):
        os.remove(RAW_LOG_PATH)

    # 영상 리스트 로드
    metadata_rows = list(load_all_metadata(METADATA_CSV_PATH))
    active_video_dir = TEST_VIDEO_DIR if TEST_MODE else VIDEO_DIR
    video_lookup = build_video_lookup(active_video_dir)
    
    # 테스트 모드 시 파일 필터링
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
    model, sampling_params = load_sglang_model()
    model_load_elapsed = time.perf_counter() - model_load_start
    init_elapsed = time.perf_counter() - init_start
    print(f"[time] 모델 로드 완료: {format_elapsed(model_load_elapsed)}")
    print(f"[time] 초기 준비 완료(프로세스 시작 후): {format_elapsed(init_elapsed)}")

    prediction_fieldnames = ["path", "accident_time", "center_x", "center_y", "type"]
    existing_paths = set()
    
    # 이전에 중단된 부분부터 다시 시작할 경우 기존 결과 로드
    if SKIP_EXISTING and os.path.exists(PREDICTION_PATH):
        with open(PREDICTION_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_paths.add(row["path"])

    # 영상별 순차 처리 루프
    for idx, meta in enumerate(metadata_rows, start=1):
        rel_path = meta.get("path")
        if not rel_path or (SKIP_EXISTING and rel_path in existing_paths):
            continue

        abs_video_path = resolve_video_path(rel_path, video_lookup)
        if not abs_video_path:
            print(f"[경보] 메타데이터와 매칭되는 비디오 파일을 찾을 수 없음: {rel_path}")
            continue

        print(f"\n[{idx}/{len(metadata_rows)}] 처리 중: {rel_path}")
        matched_name = os.path.basename(abs_video_path)
        original_name = os.path.basename(rel_path)
        if matched_name != original_name:
            print(f"  -> 매칭 파일: {matched_name}")

        video_start = time.perf_counter()
        video_outcome = "중도 종료"

        try:
            # --- [Step 1: 사고 시점(Time) 예측 및 검증] ---
            accident_time = None
            time_frame_index = None
            failed_indices: List[int] = []

            sampled_video = load_video_frames_for_vlm(abs_video_path, sample_fps=VIDEO_FPS)
            if not sampled_video:
                video_outcome = "샘플링 실패"
                print(f"  -> [오류] {rel_path}의 샘플 프레임을 만들지 못했습니다. 건너뜁니다.")
                continue

            print(
                f"  -> [time] OpenCV 샘플링: {sampled_video['num_frames']}프레임 "
                f"(sample_fps={sampled_video['sample_fps']:.2f}, raw_fps={sampled_video['raw_fps']:.2f})"
            )

            for t_attempt in range(1, MAX_TIME_RETRIES + 1):
                raw_time_json = call_qwen_for_frame_sequence(
                    model,
                    sampling_params,
                    sampled_video["frames"],
                    sampled_video["timestamps"],
                    build_time_prompt(meta, sampled_video["timestamps"], failed_indices),
                    rel_path,
                    "time",
                )
                if not raw_time_json:
                    continue

                temp_frame_index = validate_frame_index_prediction(raw_time_json, sampled_video["num_frames"])
                if temp_frame_index is not None:
                    if temp_frame_index not in failed_indices:
                        time_frame_index = temp_frame_index
                        accident_time = float(sampled_video["timestamps"][temp_frame_index])
                        break
                    print(f"  -> [time] 프레임 인덱스 {temp_frame_index}는 이미 실패했던 값이라 무시합니다.")

                try:
                    raw_idx = int(raw_time_json.get("frame_index"))
                    if raw_idx not in failed_indices:
                        failed_indices.append(raw_idx)
                except Exception:
                    pass

            if accident_time is None or time_frame_index is None:
                video_outcome = "사고 시점 추론 실패"
                print(f"  -> [오류] {rel_path}의 사고 시점을 찾지 못했습니다. 건너뜁니다.")
                continue

            print(f"  -> 최종 사고 프레임: {time_frame_index} (t={accident_time:.4f}초)")
            append_raw_log(
                RAW_LOG_PATH,
                build_raw_log_payload(
                    rel_path=rel_path,
                    stage="time",
                    attempt=t_attempt,
                    raw_output="",
                    parsed_output=raw_time_json,
                    resolved_time_frame_index=time_frame_index,
                    resolved_time_seconds=accident_time,
                ),
            )
            print_debug_payload("time", raw_time_json)

            # --- [Step 2: 핵심 프레임 추출] ---
            extracted = extract_frame_at_time(abs_video_path, accident_time, meta)
            if not extracted:
                video_outcome = "프레임 추출 실패"
                print("  -> [오류] 프레임 추출 실패.")
                continue
            
            # 추출한 프레임을 나중에 사고 위치/종류 추론에 사용하기 위해 임시 저장
            frame_filename = os.path.splitext(os.path.basename(rel_path))[0]
            frame_path = os.path.join(FRAME_DIR, f"{frame_filename}_t{accident_time:.3f}.jpg")
            cv2.imwrite(frame_path, extracted["frame"])
            abs_frame_path = os.path.abspath(frame_path)

            # --- [Step 3: 사고 위치(Location) 예측] ---
            # 비디오 대신 Step 2에서 추출한 정지 이미지를 사용하여 속도/정확도 향상
            raw_loc = call_qwen_for_media(model, sampling_params, "image", abs_frame_path, build_location_prompt(meta, accident_time), rel_path, "location")
            if not raw_loc:
                video_outcome = "사고 위치 추론 실패"
                continue
            loc = validate_location_prediction(raw_loc)
            if not loc:
                video_outcome = "사고 위치 검증 실패"
                continue
            print(f"  -> 사고 위치: ({loc['center_x']:.4f}, {loc['center_y']:.4f})")
            print_debug_payload("location", raw_loc)

            # --- [Step 4: 사고 유형(Type) 분류] ---
            raw_type = call_qwen_for_media(model, sampling_params, "image", abs_frame_path, build_type_prompt(meta, accident_time), rel_path, "type")
            if not raw_type:
                video_outcome = "사고 유형 추론 실패"
                continue
            acc_type = validate_type_prediction(raw_type)
            if not acc_type:
                video_outcome = "사고 유형 검증 실패"
                continue
            print(f"  -> 사고 유형: {acc_type}")
            print_debug_payload("type", raw_type)

            # 결과를 즉시 CSV에 append 저장
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

            # SGLang은 자체 메모리 풀을 사용하므로 여기서 torch cache를 비울 필요가 없습니다 (지연만 유발함)
            # torch.cuda.empty_cache() 제거됨

    if os.path.exists(PREDICTION_PATH):
        print(f"\n모든 결과가 저장되었습니다: {PREDICTION_PATH}")
    close_sglang_model(model)
    print(f"[time] 전체 실행 시간: {format_elapsed(time.perf_counter() - total_run_start)}")

if __name__ == "__main__":
    main()
