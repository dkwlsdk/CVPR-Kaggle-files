#!/usr/bin/env python3

import argparse
import importlib.util
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    BitsAndBytesConfig,
)
from transformers.utils import import_utils

def _extract_config_path_from_argv() -> Optional[Path]:
    for i, token in enumerate(sys.argv):
        if token == "--config" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
        if token.startswith("--config="):
            return Path(token.split("=", 1)[1])
    return None


def _resolve_config_path(base_dir: Path) -> Path:
    override = _extract_config_path_from_argv()
    config_dir = base_dir / "config"
    if override is None:
        return (config_dir / "config.py").resolve()
    if override.is_absolute():
        return override.resolve()
    cwd_candidate = (Path.cwd() / override).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    local_candidate = (base_dir / override).resolve()
    if local_candidate.exists():
        return local_candidate
    return (config_dir / override).resolve()


def _load_module(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_dir = Path(__file__).resolve().parent
config_path = _resolve_config_path(base_dir)
if not config_path.exists():
    raise FileNotFoundError(f"Config file not found: {config_path}")

os.environ["QWEN3_CONFIG_PATH"] = str(config_path)

_cfg = _load_module("_local_config", config_path)
metadata_module = _load_module("_local_metadata_utils", base_dir / "utils" / "metadata_utils.py")
compute_adaptive_fps = metadata_module.compute_adaptive_fps
load_metadata = metadata_module.load_metadata
parser_module = _load_module("_local_output_parser", base_dir / "utils" / "output_parser.py")
parse_output = parser_module.parse_output
prompt_module = _load_module("_local_prompt_loader", base_dir / "utils" / "prompt_loader.py")
PromptLoader = prompt_module.PromptLoader

DEFAULT_DURATION = _cfg.DEFAULT_DURATION
DO_SAMPLE = _cfg.DO_SAMPLE
INFERENCE_BATCH_SIZE = _cfg.INFERENCE_BATCH_SIZE
MAX_FRAMES = _cfg.MAX_FRAMES
MAX_NEW_TOKENS = _cfg.MAX_NEW_TOKENS
MAX_PIXELS = _cfg.MAX_PIXELS
METADATA_CSV = _cfg.METADATA_CSV
MIN_FRAMES = _cfg.MIN_FRAMES
MIN_PIXELS = _cfg.MIN_PIXELS
MODEL_ID = _cfg.MODEL_ID
OUTPUT_DIR = Path("./outputs") / config_path.stem
SUBMISSION_CSV = OUTPUT_DIR / "submission.csv"
SYSTEM_PROMPT_FILE = _cfg.SYSTEM_PROMPT_FILE
TEMPERATURE = _cfg.TEMPERATURE
TOP_P = _cfg.TOP_P
USER_PROMPT_FILE = _cfg.USER_PROMPT_FILE
VIDEO_DIR = _cfg.VIDEO_DIR

cfg_mod: Any = _cfg
JSON_ONLY_OUTPUT = getattr(cfg_mod, "JSON_ONLY_OUTPUT", False)
DEBUG_JSON = OUTPUT_DIR / "debug_results.json"
DEBUG_CSV = OUTPUT_DIR / "debug_results.csv"

# 로깅 설정: 시간, 로그 레벨, 메시지 포맷 지정
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def load_model(model_id: str = MODEL_ID) -> Tuple[Any, Any]:
    """
    지정된 model_id를 기반으로 모델과 프로세서를 로드하는 함수.
    로컬 GPU 환경에 맞춰 어텐션 구현 방식 및 양자화 설정을 자동으로 조정합니다.
    """
    # Flash Attention 지원 여부 확인 및 설정
    if getattr(import_utils, "is_flash_attn_3_available", lambda: False)():
        attn_implementation = "flash_attention_3"
    elif import_utils.is_flash_attn_2_available():
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa" # 기본 PyTorch 스케일드 닷 프로덕트 어텐션

    logger.info("Loading model: %s", model_id)
    logger.info("Attention backend: %s", attn_implementation)
    
    requested_model_id = model_id
    # 모델 설정(config) 파일 로드
    model_config = AutoConfig.from_pretrained(requested_model_id, trust_remote_code=True)
    model_quant_config = getattr(model_config, "quantization_config", None)

    # 양자화 메서드 식별
    quant_method = ""
    if isinstance(model_quant_config, dict):
        quant_method = str(model_quant_config.get("quant_method", "")).lower()
    elif model_quant_config is not None:
        quant_method = str(getattr(model_quant_config, "quant_method", "")).lower()

    # 모델 ID나 양자화 설정에 'fp8'이 포함되어 있는지 여부
    fp8_requested = "fp8" in requested_model_id.lower() or "fp8" in quant_method
    use_bnb_4bit = model_quant_config is None

    # GPU가 FP8을 지원하는지 확인 (Compute Capability 8.9 이상 필요, ex: RTX 40 시리즈, H100 등)
    if fp8_requested and torch.cuda.is_available():
        cc_major, cc_minor = torch.cuda.get_device_capability(0)
        # 8.9 미만 기기인 경우 (FP8 미지원)
        if (cc_major, cc_minor) < (8, 9):
            # 모델 아이디에서 마지막 '-fp8' 혹은 '_fp8' 제거하여 4bit 방식 폴백 시도
            fallback_model_id = re.sub(r"(?i)[-_]fp8$", "", requested_model_id)
            if fallback_model_id == requested_model_id:
                logger.warning(
                    "FP8 fallback could not derive a non-FP8 model ID from %s; using default base model ID.",
                    requested_model_id,
                )
                fallback_model_id = "Qwen/Qwen3-VL-8B-Instruct"
            logger.warning(
                "FP8 model requested but GPU capability is %s.%s (< 8.9). Falling back to %s with 4bit.",
                cc_major,
                cc_minor,
                fallback_model_id,
            )
            model_id = fallback_model_id
            use_bnb_4bit = True # 4비트 양자화로 대체
        else:
            # GPU가 FP8 지원할 경우
            model_id = requested_model_id
            use_bnb_4bit = False

    # 모델 로드를 위한 추가 인자 설정
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.float16, # FP16 가중치 사용
        "device_map": "auto",         # 다중 GPU 환경 시 자동 분배
        "attn_implementation": attn_implementation,
        "trust_remote_code": True,
    }

    # bitsandbytes(bnb) 4비트 양자화를 사용하는 경우 설정
    if use_bnb_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", # 최적화된 NF4 포맷 사용
        )
        logger.info("Quantization: runtime bitsandbytes 4bit")
    else:
        logger.info("Quantization: model-native config (%s)", type(model_quant_config).__name__)

    # Qwen2-VL 모델 불러오기
    model_cls = getattr(transformers, "Qwen2VLForConditionalGeneration")
    model = model_cls.from_pretrained(
        model_id,
        **model_kwargs,
    )
    # 모델에 맞는 프로세서(이미지/비디오 전처리 및 토크나이저 역할) 불러오기
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # 토크나이저의 패딩 토큰 설정
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = "left" # 자동 생성 작업에 유리한 좌측 패딩 사용

    model.eval() # 추론 모드로 변경 (학습 비활성화)
    return model, processor


def build_messages(
    *,
    video_path: str,
    duration: float,
    fps: float,
    prompt_loader: Any,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    단일 비디오에 대해 모델에 입력할 메시지 템플릿을 생성합니다.
    시스템 프롬프트와 사용자 프롬프트, 그리고 비디오 메타데이터를 포함합니다.
    """
    # 샘플 프레임 수를 반올림으로 계산해 시간 분해능 손실을 줄입니다.
    sampled_frames = max(MIN_FRAMES, min(MAX_FRAMES, int(round(duration * fps))))
    
    # 유저 프롬프트에 비디오 길이, fps, 프레임수 정보 렌더링
    user_prompt = prompt_loader.render_user_prompt(
        duration=duration, fps=fps, n_frames=sampled_frames
    )

    # VLM(시각-언어 모델) 입력 포맷에 맞춘 대화 메시지 구성
    messages = [
        {"role": "system", "content": prompt_loader.system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                    "min_frames": MIN_FRAMES,
                    "max_frames": MAX_FRAMES,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    return messages, sampled_frames


def generate_raw_batch(
    model: Any, processor: Any, messages_batch: List[List[Dict[str, Any]]]
) -> List[str]:
    """
    모델을 사용해 메시지 배치의 결과(raw text)를 한 번에 생성(추론)하는 함수.
    """
    # 배치의 메시지들을 모델이 인식할 수 있는 텍스트 형태(채팅 템플릿 적용)로 변환
    text_inputs = [
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for messages in messages_batch
    ]

    # qwen_vl_utils를 이용해 메시지 내 비디오 및 이미지 데이터 전처리
    vision_outputs = process_vision_info(messages_batch)
    vision_tuple = vision_outputs if isinstance(vision_outputs, tuple) else tuple(vision_outputs)
    image_inputs, video_inputs = vision_tuple[0], vision_tuple[1]

    processor_kwargs: Dict[str, Any] = {
        "text": text_inputs,
        "padding": True,
        "return_tensors": "pt",
    }
    if image_inputs is not None:
        processor_kwargs["images"] = image_inputs
    if video_inputs is not None:
        processor_kwargs["videos"] = video_inputs

    # 전처리된 입력을 모델이 사용하는 장치(GPU)로 이동
    inputs = processor(**processor_kwargs).to(model.device)
    
    # 그라디언트 계산 없이 텍스트 생성 (추론 단계)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    # 생성된 토큰에서 입력 부분(프롬프트) 제외
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    
    # 디코딩하여 사람이 읽을 수 있는 문자열 리스트로 반환
    return [
        text.strip()
        for text in processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    ]


def infer_batch(
    model: Any,
    processor: Any,
    batch_paths: List[str],
    batch_metas: List[Optional[Dict[str, Any]]],
    prompt_loader: Any,
) -> List[Dict[str, Any]]:
    """
    여러 비디오 영상을 배치 단위로 모델에 전달하여 추론하고, 결과를 파싱합니다.
    """
    messages_batch: List[List[Dict[str, Any]]] = []
    prepared: List[Tuple[str, float]] = []

    # 각 비디오 및 메타데이터에 대해 입력 생성 준비
    for video_path, meta in zip(batch_paths, batch_metas):
        # 메타데이터 유무에 따른 정보 설정
        if meta:
            duration = float(meta["duration"])
            no_frames = int(meta["no_frames"])
            height = int(meta["height"])
            width = int(meta["width"])
        else:
            duration = DEFAULT_DURATION
            no_frames = 0
            height = 720
            width = 1280

        # 해당 영상에 적용할 적응형 FPS 계산
        fps = compute_adaptive_fps(duration, no_frames, height, width)
        
        # 모델 입력 포맷 생성을 위해 build_messages 호출
        messages, sampled_frames = build_messages(
            video_path=video_path,
            duration=duration,
            fps=fps,
            prompt_loader=prompt_loader,
        )
        messages_batch.append(messages)
        prepared.append((video_path, duration))

        # 현재 처리되는 비디오 정보 로깅 (콘솔 출력용)
        logger.info(
            "%s | duration=%.1fs %dx%d -> fps=%.2f (~%d frames)",
            Path(video_path).name,
            duration,
            width,
            height,
            fps,
            sampled_frames,
        )

    # 텍스트 답변 생성 결과 도출 (비디오 원본 입력 후 언어 모델 결과 반환)
    raws = generate_raw_batch(model, processor, messages_batch)
    
    results: List[Dict[str, Any]] = []
    # 생성된 원본 결과를 파싱하여(예: JSON 추출) 구조화된 딕셔너리로 저장
    for (video_path, duration), raw in zip(prepared, raws):
        results.append(parse_output(raw, video_path, duration, VIDEO_DIR))
    return results


def save_outputs(records: List[Dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with DEBUG_JSON.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info("Saved: %s", DEBUG_JSON)

    if JSON_ONLY_OUTPUT:
        return

    import csv

    debug_fields = ["path", "time", "center_x", "center_y", "type", "confidence", "why", "raw", "issues"]
    with DEBUG_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=debug_fields)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k, "") for k in debug_fields})

    submission_fields = ["path", "accident_time", "center_x", "center_y", "type"]
    with SUBMISSION_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=submission_fields)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "path": row.get("path", ""),
                    "accident_time": round(float(row.get("time", 0.0) or 0.0), 2),
                    "center_x": round(float(row.get("center_x", 0.5) or 0.5), 3),
                    "center_y": round(float(row.get("center_y", 0.5) or 0.5), 3),
                    "type": row.get("type", "single"),
                }
            )

    logger.info("Saved: %s", SUBMISSION_CSV)
    logger.info("Saved: %s", DEBUG_CSV)


def run(video_dir: str, metadata_csv: str, model_id: str, limit: int) -> None:
    """
    메타데이터 로드, 모델 초기화, 배치 추론 루프 및 결과 저장 파이프라인 전체를 관리합니다.
    """
    metadata: Dict[str, Dict[str, Any]] = {}
    
    # 메타데이터 CSV 파일이 있으면 로드하고, 없으면 기본값으로 진행 안내
    if Path(metadata_csv).exists():
        metadata = load_metadata(metadata_csv)
        logger.info("Loaded metadata entries: %d", len(metadata))
    else:
        logger.warning("Metadata CSV not found: %s (fallback defaults)", metadata_csv)

    metadata_stripped: Dict[str, Dict[str, Any]] = {}
    for k, v in metadata.items():
        name = Path(str(k)).name
        stripped = re.sub(r"^\d{4}_", "", name)
        metadata_stripped[f"videos/{stripped}"] = v

    # 시스템 및 사용자 프롬프트 로드
    prompt_loader = PromptLoader(SYSTEM_PROMPT_FILE, USER_PROMPT_FILE)
    
    # 지정된 Qwen3-VL 등 비전 인공지능 모델과 프로세서 로드
    model, processor = load_model(model_id)

    # 폴더 내 모든 mp4 비디오 리스트 수집 후 정렬
    video_paths = sorted([str(p) for p in Path(video_dir).glob("*.mp4")])
    # limit이 0보다 크면 그 개수만큼만 추론 (테스트용 등)
    if limit > 0:
        video_paths = video_paths[:limit]
    if not video_paths:
        raise FileNotFoundError(f"No .mp4 files found in: {video_dir}")

    records: List[Dict[str, Any]] = []
    # tdqm을 통한 진행 바(progress bar) 설정
    pbar = tqdm(total=len(video_paths), desc="Inference", unit="video")
    try:
        # 지정된 BATCH_SIZE(배치 크기) 만큼씩 끊어서 순회 처리
        for start in range(0, len(video_paths), INFERENCE_BATCH_SIZE):
            batch_paths = video_paths[start : start + INFERENCE_BATCH_SIZE]
            batch_metas: List[Optional[Dict[str, Any]]] = []
            
            # 각 비디오의 메타데이터 조회
            for p in batch_paths:
                meta_key = f"videos/{Path(p).name}"
                stripped_name = re.sub(r"^\d{4}_", "", Path(p).name)
                stripped_key = f"videos/{stripped_name}"
                batch_metas.append(metadata.get(meta_key) or metadata_stripped.get(stripped_key))

            # 배치 추론 함수 호출 (결과값 받기)
            batch_results = infer_batch(
                model, processor, batch_paths, batch_metas, prompt_loader
            )
            records.extend(batch_results)
            pbar.update(len(batch_paths)) # 진행 바 갱신

            # 현재 배치 처리 결과 로그로 출력
            for result in batch_results:
                logger.info(
                    "-> %s | %.2fs (%.3f, %.3f) | %s | conf=%.2f | issues=%s",
                    Path(result["path"]).name,
                    result["time"],
                    result["center_x"],
                    result["center_y"],
                    result["type"],
                    result["confidence"],
                    result["issues"], # 파싱 오류 여부 등을 표시
                )
    finally:
        pbar.close() # 작업 도중 Exception이 발생해도 프로그레스 바는 안전하게 종료

    # 전체 결과 저장 처리
    save_outputs(records)


def parse_args() -> argparse.Namespace:
    """
    터미널 커맨드라인 인터페이스(CLI)용 인수 설정 함수.
    """
    parser = argparse.ArgumentParser(
        description="Traffic accident inference with Qwen3-VL-8B-FP8"
    )
    # 비디오 디렉토리 지정 (기본값 설정 파일에서)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    # 메타데이터 CSV 파일 지정
    parser.add_argument("--metadata_csv", type=str, default=METADATA_CSV)
    # 사용할 모델의 ID (또는 경로) 지정
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    # 사용할 설정 파일 지정 (config.py, config_2.py 등)
    parser.add_argument("--config", type=str, default=str(config_path))
    # 총 추론할 비디오 개수 제한 (0이면 전체 동작)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    # 메인 실행 함수
    args = parse_args()
    run(
        video_dir=args.video_dir,
        metadata_csv=args.metadata_csv,
        model_id=args.model_id,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
