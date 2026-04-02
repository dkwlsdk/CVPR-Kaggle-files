#!/usr/bin/env python3
import csv
import json
import math
import os
import re
import time
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer, BitsAndBytesConfig

# --- GPU 환경 및 보안 설정 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# PyTorch 메모리 파편화 방지 설정
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# HuggingFace 인증 토큰 (비공개 모델 로드 시 필요)
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

# --- 모델 및 생성 제어 하이퍼파라미터 ---
MODEL_NAME = "Qwen/Qwen3.5-9B"
# MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# 4-bit 양자화 사용 여부 (VRAM 절약)
USE_4BIT = True
# 모델이 생성할 최대 답변 길이
MAX_NEW_TOKENS = 1536
# 답변의 창의성 제어 (0.0은 가장 결정론적인 답변)
TEMPERATURE = 0.0
# 답변 생성 시 고려할 누적 확률 범위
TOP_P = 1.0
# 동일 문구 반복 방지를 위한 가중치
REPETITION_PENALTY = 1.2

# --- 시각 분석 및 샘플링 설정 ---
# 프레임당 픽셀 수 제한 (OOM 에러 방지를 위해 중요)
MIN_PIXELS = 256 * 28 * 28     # 최소 픽셀 (약 20만)
MAX_PIXELS = 1280 * 28 * 28    # 최대 픽셀 (약 100만)
# 비디오 분석 시 초당 샘플링할 프레임 수
VIDEO_FPS = 1.0

# --- 재시도 및 검증 로직 설정 ---
MAX_TIME_RETRIES = 3           # 사고 시점 예측 실패 시 최대 재시도 횟수
# 이전 예측값과 현재 예측값이 동일한지 판단할 오차 범위 (초 단위)
TIME_MATCH_THRESHOLD = 0.1
# 모델 호출 자체의 재시도 횟수 (JSON 파싱 실패 등 대비)
MAX_CALL_RETRIES = 2

# --- 경로 및 기타 설정 ---
BASE_DIR = "/root/Desktop/workspace/ja"
VIDEO_DIR = "/root/Desktop/workspace/ja/CVPR-Kaggle-files/데이터셋/test_videos"
METADATA_CSV_PATH = "/root/Desktop/workspace/ja/CVPR-Kaggle-files/데이터셋/test_metadata.csv"
OUTPUT_DIR = "/root/Desktop/workspace/ja/qwen_test_output"

# 결과 저장 경로 설정
PREDICTION_PATH = os.path.join(OUTPUT_DIR, "predictions.csv")
RAW_LOG_PATH = os.path.join(OUTPUT_DIR, "raw_outputs.jsonl")
FRAME_DIR = os.path.join(OUTPUT_DIR, "frames")

# 사고 유형 분류 목록
VALID_TYPES = {"rear-end", "head-on", "sideswipe", "t-bone", "single"}

# 테스트 모드 설정 (True일 경우 특정 폴더의 일부 영상만 처리)
TEST_MODE = True
TEST_VIDEO_DIR = "/root/Desktop/workspace/ja/temp"
TEST_LIMIT = 2        # 테스트 시 처리할 최대 영상 수
SKIP_EXISTING = False # 이미 처리된 영상 건너뛰기 여부

# GPU 메모리 예약 공간 (MiB 단위)
CUDA_MEMORY_RESERVE_MIB = 256

# --- 하드웨어 및 모델 로드 유틸리티 ---

def get_cuda_free_bytes(device_index: int) -> Optional[int]:
    """해당 GPU 장치의 사용 가능한 메모리(바이트)를 반환합니다."""
    try:
        with torch.cuda.device(device_index):
            free_bytes, _ = torch.cuda.mem_get_info()
        return int(free_bytes)
    except Exception:
        return None

def build_quantization_config():
    """4-bit 양자화 설정을 생성합니다."""
    if not USE_4BIT:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

def build_model_load_config() -> Tuple[Dict[str, Any], List[str]]:
    """모델 로드 시 사용할 하드웨어 환경 설정을 구성합니다."""
    notes: List[str] = []
    quantization_config = build_quantization_config()
    if quantization_config:
        notes.append("4-bit 양자화가 적용되었습니다.")

    if not torch.cuda.is_available():
        notes.append("CUDA를 사용할 수 없어 CPU로 로드합니다.")
        return {"device_map": "cpu", "torch_dtype": torch.float32}, notes

    max_memory: Dict[Any, str] = {}
    for device_index in range(torch.cuda.device_count()):
        free_bytes = get_cuda_free_bytes(device_index)
        gpu_name = torch.cuda.get_device_name(device_index)
        if free_bytes is None:
            notes.append(f"cuda:{device_index} ({gpu_name}): 여유 메모리 정보를 가져올 수 없습니다.")
            continue

        # 예약 공간을 제외한 실제 사용 가능 메모리 계산
        reserve_bytes = CUDA_MEMORY_RESERVE_MIB * 1024 * 1024
        usable_mib = max(1024, (free_bytes // (1024 * 1024)) - CUDA_MEMORY_RESERVE_MIB)
        max_memory[device_index] = f"{usable_mib}MiB"
        notes.append(
            f"cuda:{device_index} ({gpu_name}): 여유={free_bytes / (1024**3):.2f} GiB, 할당 가능={usable_mib} MiB"
        )

    config = {
        "device_map": "auto",      # 레이어를 여러 GPU에 자동으로 분산 배치
        "max_memory": max_memory,
        "torch_dtype": "auto",
    }
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    return config, notes

def ensure_runtime_compatibility() -> Optional[str]:
    """필요한 라이브러리 버전을 충족하는지 확인합니다."""
    MIN_PYTHON = (3, 8)
    MIN_TRANSFORMERS = (4, 46)
    py_version = tuple(map(int, os.sys.version_info[:2]))
    if py_version < MIN_PYTHON:
        return f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ 버전이 필요합니다."
    tf_version = tuple(map(int, re.findall(r"\d+", transformers.__version__)[:2]))
    if tf_version < MIN_TRANSFORMERS:
        return f"transformers {MIN_TRANSFORMERS[0]}.{MIN_TRANSFORMERS[1]}+ 버전이 필요합니다."
    return None

def get_model_input_device(model) -> torch.device:
    """모델의 입력 데이터를 전달할 장치(Device)를 찾습니다."""
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
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 프롬프트 빌딩 섹션 ---

def build_time_prompt(metadata: Dict[str, str], failed_times: List[float] = None) -> str:
    """사고 발생 시점을 찾기 위한 비디오 분석 프롬프트를 생성합니다."""
    # 메타데이터 추출
    region = metadata.get("region", "")
    scene_layout = metadata.get("scene_layout", "")
    weather = metadata.get("weather", "")
    day_time = metadata.get("day_time", "")
    quality = metadata.get("quality", "")
    duration = metadata.get("duration", "")
    no_frames = metadata.get("no_frames", "")
    height = metadata.get("height", "")
    width = metadata.get("width", "")

    # 재시도 시 이전 오답을 피하도록 지시 추가 (모델 입력용이므로 영어로 작성)
    retry_note = ""
    if failed_times:
        retry_note = f"\n\n[CRITICAL NOTE]\nYou previously estimated accident times at {failed_times}, but those were INCORRECT. DO NOT output these times again. Find a DIFFERENT moment in the video."

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
- frame_width: {width}

Instructions:
1. Carefully analyze the ENTIRE video.
2. Find the earliest accident_time (in seconds) when a traffic accident CLEARLY BEGINS.
3. accident_time must correspond to the earliest collision moment:
   - the first frame where physical contact begins, or
   - the first frame where collision is clearly unavoidable and immediate.
4. Ignore the exact location and the accident type in this step.
5. Focus only on accurately detecting the first accident_time.

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
  "accident_time"

Output format:
Reasoning here...
{{
  "accident_time": <float>
}}
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
Reasoning here...
{{
  "center_x": <float>,
  "center_y": <float>
}}
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
Reasoning here...
{{
  "type": "<one of: rear-end, head-on, sideswipe, t-bone, single>"
}}
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
    """텍스트에서 유효한 첫 번째 JSON 객체를 추출합니다. 마크다운 처리 및 괄호 매칭 포함."""
    if not text:
        return None

    # 생각/추론 접두사 정리
    text = strip_thinking_text(text)
    
    # 1. 마크다운 코드 블록 제거 시도
    cleaned_text = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL)
    cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text.strip())
    cleaned_text = re.sub(r"```$", "", cleaned_text.strip())

    # 전체가 바로 파싱되는지 확인
    direct = try_parse_single_json(cleaned_text)
    if direct is not None:
        return direct

    # 1.5 중괄호 { } 범위를 찾아 파싱 시도
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

    # 2. 괄호 개수 매칭(Brace Counting) 방식으로 실패 대비 추출
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
    """예측된 사고 시간이 유효한 범위 내에 있는지 확인합니다."""
    try:
        accident_time = float(result["accident_time"])
    except (KeyError, TypeError, ValueError):
        return None
    
    duration_str = meta.get("duration", "0")
    try:
        duration = float(duration_str)
    except (TypeError, ValueError):
        duration = 0.0

    # 엄격한 검증 로직 (사고 시간은 0보다 크고 영상 길이보다 짧아야 함)
    if accident_time <= 0.0:
        return None
    if duration > 0 and accident_time > duration:
        return None
    # 롱 비디오인데 너무 초반(1초 미만)에 발생했다고 한 경우 필터링 (경험적 검증)
    if duration >= 3.0 and accident_time <= 1.0:
        return None
        
    return accident_time

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

def move_inputs_to_device(batch, device):
    """모델 입력 텐서들을 지정된 GPU 장치로 이동시킵니다."""
    moved = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved

def call_qwen_for_media(
    model,
    processor,
    media_type: str,
    media_path: str,
    prompt: str,
    rel_path: str,
    stage: str,
    max_retries: int = MAX_CALL_RETRIES,
    fps: float = VIDEO_FPS,
) -> Optional[Dict[str, Any]]:
    """VLM 모델을 호출하여 비디오 또는 이미지 분석 결과를 받아옵니다."""
    
    # 전송용 메시지 구조 구성
    media_dict = {"type": media_type, media_type: media_path}
    if media_type == "video":
        media_dict["fps"] = fps # 라이브러리에 FPS 힌트 제공
        
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Respond with JSON only. No reasoning. No explanation. /no_think"}],
        },
        {
            "role": "user",
            "content": [
                media_dict,
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            # 해상도 제한 설정을 포함한 전처리
            from qwen_vl_utils import process_vision_info
            # 1. 채팅 템플릿 적용 (텍스트만)
            # 2. 이미지/비디오 데이터 로드 및 픽셀 제한 적용
            image_inputs, video_inputs = process_vision_info(messages)
            processed = processor(
                text=[processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)],
                images=image_inputs,
                videos=video_inputs,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,
                padding=True,
                return_tensors="pt",
            )
            
            # 장치 이동
            processed = move_inputs_to_device(processed, get_model_input_device(model))
            
            # 스트리밍 방식의 출력 생성
            streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                **processed,
                streamer=streamer,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=processor.tokenizer.eos_token_id,
                do_sample=(TEMPERATURE > 0),
                repetition_penalty=REPETITION_PENALTY,
            )
            if TEMPERATURE > 0:
                generation_kwargs["temperature"] = TEMPERATURE
                generation_kwargs["top_p"] = TOP_P
            
            # 쓰레드를 이용한 비동기식 생성 시작
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            print(f"  -> [{stage}] (시도 {attempt}/{max_retries}) 모델 출력: ", end="", flush=True)
            collected_text = ""
            for new_text in streamer:
                print(new_text, end="", flush=True)
                collected_text += new_text
            thread.join()
            print()
            
            # 로그 기록 및 JSON 추출
            append_raw_log(RAW_LOG_PATH, {"path": rel_path, "stage": stage, "attempt": attempt, "raw_output": collected_text})
            parsed = extract_first_json_object(collected_text)
            if parsed is not None:
                return parsed
                
            last_error = f"JSON 파싱 실패 (시도 {attempt})"
        except Exception as e:
            last_error = f"에러 발생: {str(e)}"
            
        time.sleep(1.0) # 재시도 전 대기
        
    print(f"    [오류] Qwen 요청 최종 실패: {last_error}")
    return None

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

# --- 메인 실행 프로세스 ---

def main():
    # 출력 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)

    # 이전에 기록된 원시 로그 삭제 (새로 시작할 경우)
    if not SKIP_EXISTING and os.path.exists(RAW_LOG_PATH):
        os.remove(RAW_LOG_PATH)

    # 영상 리스트 로드
    metadata_rows = list(load_all_metadata(METADATA_CSV_PATH))
    active_video_dir = TEST_VIDEO_DIR if TEST_MODE else VIDEO_DIR
    
    # 테스트 모드 시 파일 필터링
    if TEST_MODE:
        available_files = set(os.listdir(active_video_dir))
        metadata_rows = [m for m in metadata_rows if os.path.basename(m["path"]) in available_files]
        if TEST_LIMIT > 0:
            metadata_rows = metadata_rows[:TEST_LIMIT]

    print(f"모델을 로드합니다: {MODEL_NAME}")
    comp_error = ensure_runtime_compatibility()
    if comp_error:
        print(f"환경 호환성 오류: {comp_error}")
        return

    # 모델 세부 설정 및 로드
    model_config, notes = build_model_load_config()
    for note in notes:
        print(f"  -> {note}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        **model_config
    )

    predictions = []
    existing_paths = set()
    
    # 이전에 중단된 부분부터 다시 시작할 경우 기존 결과 로드
    if SKIP_EXISTING and os.path.exists(PREDICTION_PATH):
        with open(PREDICTION_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_paths.add(row["path"])
                predictions.append(row)

    # 영상별 순차 처리 루프
    for idx, meta in enumerate(metadata_rows, start=1):
        rel_path = meta.get("path")
        if not rel_path or (SKIP_EXISTING and rel_path in existing_paths):
            continue

        video_path = os.path.join(active_video_dir, os.path.basename(rel_path))
        if not os.path.exists(video_path):
            print(f"[경보] 파일을 찾을 수 없음: {video_path}")
            continue

        abs_video_path = os.path.abspath(video_path)
        print(f"\n[{idx}/{len(metadata_rows)}] 처리 중: {rel_path}")

        # --- [Step 1: 사고 시점(Time) 예측 및 검증] ---
        accident_time = None
        failed_times = [] # 틀렸던 시간을 기록하여 재시도 시 모델에게 전달
        
        for t_attempt in range(1, MAX_TIME_RETRIES + 1):
            raw_time_json = call_qwen_for_media(
                model, processor, "video", abs_video_path, 
                build_time_prompt(meta, failed_times), rel_path, "time",
                fps=VIDEO_FPS
            )
            if not raw_time_json:
                continue
                
            temp_time = validate_time_prediction(raw_time_json, meta)
            if temp_time is not None:
                # 이미 시도했던 틀린 시간값과 너무 비슷한지 체크
                is_repeated = False
                for ft in failed_times:
                    if abs(temp_time - ft) <= TIME_MATCH_THRESHOLD:
                        is_repeated = True
                        break
                
                if not is_repeated:
                    accident_time = temp_time
                    break # 유효한 값 발견 시 루프 탈출
                else:
                    print(f"  -> [time] 예측값 {temp_time:.2f}초는 이미 실패했던 값과 비슷하여 무시합니다.")
            
            # 실패한 시간 기록
            try:
                raw_val = float(raw_time_json.get("accident_time", 0))
                if raw_val not in failed_times:
                    failed_times.append(raw_val)
            except:
                pass
                
        if accident_time is None:
            print(f"  -> [오류] {rel_path}의 사고 시점을 찾지 못했습니다. 건너뜁니다.")
            continue
            
        print(f"  -> 최종 사고 시점: {accident_time:.4f}초")

        # --- [Step 2: 핵심 프레임 추출] ---
        extracted = extract_frame_at_time(abs_video_path, accident_time, meta)
        if not extracted:
            print("  -> [오류] 프레임 추출 실패.")
            continue
        
        # 추출한 프레임을 나중에 사고 위치/종류 추론에 사용하기 위해 임시 저장
        frame_filename = os.path.splitext(os.path.basename(rel_path))[0]
        frame_path = os.path.join(FRAME_DIR, f"{frame_filename}_t{accident_time:.3f}.jpg")
        cv2.imwrite(frame_path, extracted["frame"])
        abs_frame_path = os.path.abspath(frame_path)

        # --- [Step 3: 사고 위치(Location) 예측] ---
        # 비디오 대신 Step 2에서 추출한 정지 이미지를 사용하여 속도/정확도 향상
        raw_loc = call_qwen_for_media(model, processor, "image", abs_frame_path, build_location_prompt(meta, accident_time), rel_path, "location")
        if not raw_loc: continue
        loc = validate_location_prediction(raw_loc)
        if not loc: continue
        print(f"  -> 사고 위치: ({loc['center_x']:.4f}, {loc['center_y']:.4f})")

        # --- [Step 4: 사고 유형(Type) 분류] ---
        raw_type = call_qwen_for_media(model, processor, "image", abs_frame_path, build_type_prompt(meta, accident_time), rel_path, "type")
        if not raw_type: continue
        acc_type = validate_type_prediction(raw_type)
        if not acc_type: continue
        print(f"  -> 사고 유형: {acc_type}")

        # 결과 저장 리스트에 추가
        predictions.append({
            "path": rel_path,
            "accident_time": accident_time,
            "center_x": loc["center_x"],
            "center_y": loc["center_y"],
            "type": acc_type,
        })

        # 비디오 하나 처리할 때마다 캐시 비워서 메모리 관리
        torch.cuda.empty_cache()

    # 최종 CSV 파일 저장
    if predictions:
        fieldnames = ["path", "accident_time", "center_x", "center_y", "type"]
        with open(PREDICTION_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(predictions)
        print(f"\n모든 결과가 저장되었습니다: {PREDICTION_PATH}")

if __name__ == "__main__":
    main()
