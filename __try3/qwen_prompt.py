#!/usr/bin/env python3
import os
from typing import Dict, List

# ==========================================
# 1. 하이퍼 파라미터 설정 (Hyperparameters)
# ==========================================

# 사용할 모델 경로 또는 ID
MODEL_NAME = "Qwen/Qwen3.5-9B"
# MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# 4-bit 양자화 사용 여부 (VRAM 절약)
USE_4BIT = True
# 모델이 생성할 최대 답변 길이
MAX_NEW_TOKENS = 2048
# 답변의 창의성 제어 (0.0은 가장 결정론적인 답변)
TEMPERATURE = 0.0
# 답변 생성 시 고려할 누적 확률 범위
TOP_P = 1.0
# 동일 문구 반복 방지를 위한 가중치
REPETITION_PENALTY = 1.3

# --- 시각 분석 및 샘플링 설정 ---
# 프레임당 픽셀 수 제한 (OOM 에러 방지를 위해 중요)
MAX_PIXELS = 256 * 28 * 28
# MAX_PIXELS = 1280 * 28 * 28
# MAX_PIXELS = 768 * 28 * 28
# MAX_PIXELS = 424 * 28 * 28
MIN_PIXELS = 192 * 28 * 28
# 비디오 분석 시 초당 샘플링할 프레임 수
VIDEO_FPS = 1.0

# --- 재시도 및 검증 로직 설정 ---
MAX_TIME_RETRIES = 3
TIME_MATCH_THRESHOLD = 0.1
MAX_CALL_RETRIES = 2

# ==========================================
# 2. 파일 경로 지정 (Paths)
# ==========================================

BASE_DIR = "/root/Desktop/workspace/ja"
VIDEO_DIR = "/root/Desktop/workspace/ja/CVPR-Kaggle-files/데이터셋/test_videos"
METADATA_CSV_PATH = "/root/Desktop/workspace/ja/CVPR-Kaggle-files/데이터셋/test_metadata.csv"
OUTPUT_DIR = "/root/Desktop/workspace/ja/qwen_test_output_9B/02.video"

# 결과 저장 경로 설정
PREDICTION_PATH = os.path.join(OUTPUT_DIR, "predictions.csv")
RAW_LOG_PATH = os.path.join(OUTPUT_DIR, "raw_outputs.jsonl")
FRAME_DIR = os.path.join(OUTPUT_DIR, "frames")

# 사고 유형 분류 목록
VALID_TYPES = {"rear-end", "head-on", "sideswipe", "t-bone", "single"}

# 테스트 모드 설정
TEST_MODE = False
TEST_VIDEO_DIR = "/root/Desktop/workspace/ja/temp"
# TEST_VIDEO_DIR = "/root/Desktop/workspace/ja/CVPR-Kaggle-files/데이터셋/test_videos"
TEST_LIMIT = 0
SKIP_EXISTING = False
CUDA_DEVICE = "1"
CUDA_MEMORY_RESERVE_MIB = 256

# ==========================================
# 3. 프롬프트 빌딩 (Prompt Building)
# ==========================================

def build_time_prompt(metadata: Dict[str, str], failed_times: List[float] = None) -> str:
    """사고 발생 시점을 찾기 위한 비디오 분석 프롬프트를 생성합니다."""
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
- The FINAL output MUST be exactly one JSON object. DO NOT use a list [].
- "accident_time" MUST be a single FLOAT (e.g., 12.34). DO NOT include arithmetic expressions or units.
- Include a brief reasoning in English for your choice inside the JSON under the key "reasoning".
- No markdown, no code blocks, no text before or after the JSON.
- The JSON must contain exactly these keys:
  "reasoning", "accident_time"

Output format:
{{
  "reasoning": "<brief explanation in English>",
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
- The FINAL output MUST be exactly one JSON object. DO NOT use a list [].
- "center_x" and "center_y" MUST be single FLOATs between 0.0 and 1.0.
- Include a brief reasoning in English for your choice inside the JSON under the key "reasoning".
- No markdown, no code blocks, no text before or after the JSON.
- The JSON must contain exactly these keys:
  "reasoning", "center_x", "center_y"

Output format:
{{
  "reasoning": "<brief explanation in English>",
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
- The FINAL output MUST be exactly one JSON object. DO NOT use a list [].
- Include a brief reasoning in English for your choice inside the JSON under the key "reasoning".
- No markdown, no code blocks, no text before or after the JSON.
- The JSON must contain exactly these keys:
  "reasoning", "type"

Output format:
{{
  "reasoning": "<brief explanation in English>",
  "type": "<one of: rear-end, head-on, sideswipe, t-bone, single>"
}}
"""
    return prompt.strip()
