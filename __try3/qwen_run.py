#!/usr/bin/env python3
import csv
import json
import os
import re
import time
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

# qwen_main에서 설정과 프롬프트를 가져옵니다.
import qwen_prompt as cfg

# --- 환경 및 보안 설정 기초 (엔진 시작 전 필수 설정) ---
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_DEVICE
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

import cv2
import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer, BitsAndBytesConfig

# --- 하드웨어 및 모델 로드 엔진 ---

def get_cuda_free_bytes(device_index: int) -> Optional[int]:
    try:
        with torch.cuda.device(device_index):
            free_bytes, _ = torch.cuda.mem_get_info()
        return int(free_bytes)
    except Exception:
        return None

def build_quantization_config(use_4bit: bool):
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

def build_model_load_config(use_4bit: bool, reserve_mib: int) -> Tuple[Dict[str, Any], List[str]]:
    notes: List[str] = []
    quantization_config = build_quantization_config(use_4bit)
    if quantization_config:
        notes.append("4-bit 양자화가 적용되었습니다.")

    if not torch.cuda.is_available():
        notes.append("CUDA를 사용할 수 없어 CPU로 로드합니다.")
        return {"device_map": "cpu", "torch_dtype": torch.float32}, notes

    max_memory: Dict[Any, str] = {}
    for device_index in range(torch.cuda.device_count()):
        free_bytes = get_cuda_free_bytes(device_index)
        gpu_name = torch.cuda.get_device_name(device_index)
        if free_bytes is None: continue
        usable_mib = max(1024, (free_bytes // (1024 * 1024)) - reserve_mib)
        max_memory[device_index] = f"{usable_mib}MiB"
        notes.append(f"cuda:{device_index} ({gpu_name}): 여유={free_bytes / (1024**3):.2f} GiB, 할당 가능={usable_mib} MiB")

    config = {"device_map": "auto", "max_memory": max_memory, "torch_dtype": "auto"}
    if quantization_config is not None: config["quantization_config"] = quantization_config
    return config, notes

def ensure_runtime_compatibility() -> Optional[str]:
    MIN_PYTHON, MIN_TRANSFORMERS = (3, 8), (4, 46)
    py_version = tuple(map(int, os.sys.version_info[:2]))
    if py_version < MIN_PYTHON: return f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ 필요."
    tf_version = tuple(map(int, re.findall(r"\d+", transformers.__version__)[:2]))
    if tf_version < MIN_TRANSFORMERS: return f"transformers {MIN_TRANSFORMERS[0]}.{MIN_TRANSFORMERS[1]}+ 필요."
    return None

def get_model_input_device(model) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for mapped_device in hf_device_map.values():
            if isinstance(mapped_device, int): return torch.device(f"cuda:{mapped_device}")
            if isinstance(mapped_device, str) and mapped_device not in {"cpu", "disk"}: return torch.device(mapped_device)
    model_device = getattr(model, "device", None)
    if model_device: return torch.device(model_device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 보조 유틸리티 ---

def strip_thinking_text(text: str) -> str:
    if not text: return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^(?:The user wants|I understand|Okay|Let me|Based on).*?:\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()

def try_parse_single_json(candidate: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except: return None

def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text: return None
    text = strip_thinking_text(text)
    
    # 0. 전처리: 주석 제거 및 NaN/Infinity 처리
    # // 주석 제거
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    # NaN -> null, Infinity -> 큰 숫자로 치환
    text = re.sub(r"\bNaN\b", "null", text)
    text = re.sub(r"\bInfinity\b", "999999", text)
    
    cleaned = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip())
    
    # 1. 직접 파싱 시도
    parsed = try_parse_single_json(cleaned)
    if parsed: return parsed
    
    # 2. 리스트 형태인 경우([ {} ]) 첫 번째 원소 추출 시도
    if cleaned.startswith("["):
        try:
            # 리스트 끝을 찾기 위한 시도
            end_idx = cleaned.rfind("]")
            if end_idx != -1:
                trimmed_list = cleaned[:end_idx+1]
                arr = json.loads(trimmed_list)
                if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], dict):
                    return arr[0]
        except: pass

    # 3. 중괄호 찾기 (가장 확실한 방법)
    try:
        s, e = cleaned.find('{'), cleaned.rfind('}')
        if s != -1 and e != -1 and e > s:
            parsed = try_parse_single_json(cleaned[s:e+1])
            if parsed: return parsed
    except: pass
    return None

def validate_time_prediction(result: Dict[str, Any], meta: Dict[str, str]) -> Tuple[Optional[float], Optional[str]]:
    """사고 시점을 검증합니다. 결과와 에러 메시지를 함께 반환합니다."""
    try:
        if not result or "accident_time" not in result:
            return None, "JSON에 'accident_time' 키가 없음"
        
        raw_val = result["accident_time"]
        try:
            accident_time = float(raw_val)
        except (ValueError, TypeError):
            return None, f"accident_time이 숫자가 아님: {raw_val}"

        duration = float(meta.get("duration", 0))
        if accident_time <= 0.0:
            return None, f"사고 시점이 0이하임: {accident_time}"
        if duration > 0 and accident_time > duration:
            return None, f"사고 시점이 영상 길이({duration}s)를 초과함: {accident_time}"
        if duration >= 3.0 and accident_time <= 1.0:
            return None, f"롱 비디오인데 사고가 너무 초반(1초 이하)에 발생함: {accident_time}"
        
        return accident_time, None
    except Exception as e:
        return None, f"검증 중 예외 발생: {str(e)}"

def validate_location_prediction(result: Dict[str, Any]) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """좌표를 검증합니다. (0,0)은 실패로 처리합니다."""
    try:
        if not result or "center_x" not in result or "center_y" not in result:
            return None, "JSON에 'center_x' 또는 'center_y' 키가 없음"
            
        x, y = float(result["center_x"]), float(result["center_y"])
        if x == 0.0 and y == 0.0:
            return None, "좌표가 (0,0)임 (실패 처리)"
            
        return {"center_x": min(max(x, 0.0), 1.0), "center_y": min(max(y, 0.0), 1.0)}, None
    except Exception as e:
        return None, f"좌표 변환 실패: {str(e)}"

def validate_type_prediction(result: Dict[str, Any], valid_types: set) -> Tuple[Optional[str], Optional[str]]:
    """사고 유형을 검증합니다."""
    try:
        if not result or "type" not in result:
            return None, "JSON에 'type' 키가 없음"
            
        t = str(result["type"]).strip()
        if t not in valid_types:
            return None, f"유효하지 않은 사고 유형: {t}"
            
        return t, None
    except Exception as e:
        return None, f"유형 검증 실패: {str(e)}"

# --- 실행 엔진 ---

def call_qwen_for_media(model, processor, media_type, media_path, prompt, rel_path, stage):
    media_dict = {
        "type": media_type, 
        media_type: media_path,
        "min_pixels": cfg.MIN_PIXELS,
        "max_pixels": cfg.MAX_PIXELS
    }
    if media_type == "video": media_dict["fps"] = cfg.VIDEO_FPS
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Respond with JSON including a brief reasoning for your analysis."}]},
        {"role": "user", "content": [media_dict, {"type": "text", "text": prompt}]}
    ]
    for attempt in range(1, cfg.MAX_CALL_RETRIES + 1):
        try:
            from qwen_vl_utils import process_vision_info
            imgs, vids = process_vision_info(messages)
            processed = processor(
                text=[processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)],
                images=imgs, videos=vids, padding=True, return_tensors="pt"
            )
            processed = move_inputs_to_device(processed, get_model_input_device(model))
            streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(**processed, streamer=streamer, max_new_tokens=cfg.MAX_NEW_TOKENS, pad_token_id=processor.tokenizer.eos_token_id, do_sample=(cfg.TEMPERATURE > 0), repetition_penalty=cfg.REPETITION_PENALTY)
            if cfg.TEMPERATURE > 0:
                gen_kwargs["temperature"] = cfg.TEMPERATURE
                gen_kwargs["top_p"] = cfg.TOP_P
            Thread(target=model.generate, kwargs=gen_kwargs).start()
            print(f"  -> [{stage}] ({attempt}/{cfg.MAX_CALL_RETRIES}) 출력: ", end="", flush=True)
            res = ""
            for t in streamer:
                print(t, end="", flush=True)
                res += t
            print()
            with open(cfg.RAW_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"path": rel_path, "stage": stage, "attempt": attempt, "raw_output": res}, ensure_ascii=False) + "\n")
            parsed = extract_first_json_object(res)
            if parsed: return parsed
        except Exception as e: print(f"  에러: {e}")
        time.sleep(1.0)
    return None

def move_inputs_to_device(batch, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

def extract_frame(video_path, time_s, meta):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        try: fps = float(meta["no_frames"]) / float(meta["duration"])
        except: return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_s * fps))
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000.0)
        ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def normalize_meta(row):
    row = dict(row)
    if "scene_layout" not in row and "scene_layoutm" in row: row["scene_layout"] = row["scene_layoutm"]
    return row

def load_meta(path):
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f): yield normalize_meta(row)

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

# ==========================================
# 4. 메인 실행 루프 (Execution Runner)
# ==========================================

def run_pipeline():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.FRAME_DIR, exist_ok=True)
    if not cfg.SKIP_EXISTING and os.path.exists(cfg.RAW_LOG_PATH): os.remove(cfg.RAW_LOG_PATH)

    metadata = list(load_meta(cfg.METADATA_CSV_PATH))
    v_dir = cfg.TEST_VIDEO_DIR if cfg.TEST_MODE else cfg.VIDEO_DIR
    video_lookup = build_video_lookup(v_dir)
    if cfg.TEST_MODE:
        metadata = [m for m in metadata if resolve_video_path(m.get("path"), video_lookup)][:cfg.TEST_LIMIT]

    print(f"모델 로드: {cfg.MODEL_NAME}")
    err = ensure_runtime_compatibility()
    if err:
        print(f"환경 오류: {err}")
        return

    m_cfg, notes = build_model_load_config(cfg.USE_4BIT, cfg.CUDA_MEMORY_RESERVE_MIB)
    for n in notes: print(f"  -> {n}")

    proc = AutoProcessor.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True, **m_cfg)

    results = []
    done_paths = set()
    if cfg.SKIP_EXISTING and os.path.exists(cfg.PREDICTION_PATH):
        with open(cfg.PREDICTION_PATH, "r") as f:
            for r in csv.DictReader(f): done_paths.add(r["path"]); results.append(r)

    for idx, meta in enumerate(metadata, 1):
        rel = meta.get("path")
        if not rel or (cfg.SKIP_EXISTING and rel in done_paths): continue
        v_path = resolve_video_path(rel, video_lookup)
        if not v_path:
            print(f"\n[{idx}/{len(metadata)}] {rel}")
            print("  -> [건너뜀] 메타데이터와 매칭되는 비디오 파일을 찾지 못했습니다.")
            continue

        print(f"\n[{idx}/{len(metadata)}] {rel}")
        matched_name = os.path.basename(v_path)
        original_name = os.path.basename(rel)
        if matched_name != original_name:
            print(f"  -> 매칭 파일: {matched_name}")
        
        # Step 1: Time
        acc_t, fails = None, []
        for t_retry in range(1, cfg.MAX_TIME_RETRIES + 1):
            retry_tag = f"(재시도 {t_retry}/{cfg.MAX_TIME_RETRIES})"
            raw = call_qwen_for_media(model, proc, "video", v_path, cfg.build_time_prompt(meta, fails), rel, f"time {retry_tag}")
            if not raw:
                print(f"    -> [time] {retry_tag} 실패: 모델 응답 없음 또는 JSON 추출 실패")
                continue
                
            val, err_msg = validate_time_prediction(raw, meta)
            if val is not None:
                if any(abs(val - f) <= cfg.TIME_MATCH_THRESHOLD for f in fails):
                    print(f"    -> [time] {retry_tag} 실패: 이전 실패값({val}s)과 너무 유사함")
                else:
                    acc_t = val; break
            else:
                print(f"    -> [time] {retry_tag} 검증 실패: {err_msg}")
            
            if raw and "accident_time" in raw:
                try: fails.append(float(raw["accident_time"]))
                except: pass
        
        if acc_t is None:
            print(f"  -> [Step 1 중단] {rel}의 유효한 사고 시점을 찾지 못했습니다.")
            continue
        print(f"  -> 사고 시점 확정: {acc_t:.4f}s")

        # Step 2: Frame
        img = extract_frame(v_path, acc_t, meta)
        if img is None:
            print(f"  -> [Step 2 에러] {rel} 프레임 추출 실패")
            continue
        f_name = os.path.splitext(os.path.basename(rel))[0]
        f_path = os.path.abspath(os.path.join(cfg.FRAME_DIR, f"{f_name}_t{acc_t:.3f}.jpg"))
        cv2.imwrite(f_path, img)

        # Step 3: Location
        loc = None
        for l_retry in range(1, cfg.MAX_CALL_RETRIES + 1):
            retry_tag = f"(재시도 {l_retry}/{cfg.MAX_CALL_RETRIES})"
            raw_l = call_qwen_for_media(model, proc, "image", f_path, cfg.build_location_prompt(meta, acc_t), rel, f"location {retry_tag}")
            if not raw_l:
                print(f"    -> [location] {retry_tag} 실패: 응답 없음")
                continue
            
            val_l, err_msg = validate_location_prediction(raw_l)
            if val_l:
                loc = val_l; break
            else:
                print(f"    -> [location] {retry_tag} 검증 실패: {err_msg}")
                
        if not loc:
            print(f"  -> [Step 3 중단] {rel}의 유효한 위치를 찾지 못했습니다.")
            continue
        print(f"  -> 위치 확정: ({loc['center_x']:.4f}, {loc['center_y']:.4f})")

        # Step 4: Type
        acc_type = None
        for ty_retry in range(1, cfg.MAX_CALL_RETRIES + 1):
            retry_tag = f"(재시도 {ty_retry}/{cfg.MAX_CALL_RETRIES})"
            raw_ty = call_qwen_for_media(model, proc, "image", f_path, cfg.build_type_prompt(meta, acc_t), rel, f"type {retry_tag}")
            if not raw_ty:
                print(f"    -> [type] {retry_tag} 실패: 응답 없음")
                continue
                
            val_ty, err_msg = validate_type_prediction(raw_ty, cfg.VALID_TYPES)
            if val_ty:
                acc_type = val_ty; break
            else:
                print(f"    -> [type] {retry_tag} 검증 실패: {err_msg}")
                
        if not acc_type:
            print(f"  -> [Step 4 중단] {rel}의 유효한 유형을 찾지 못했습니다.")
            continue
        print(f"  -> 유형 확정: {acc_type}")

        results.append({"path": rel, "accident_time": acc_t, "center_x": loc["center_x"], "center_y": loc["center_y"], "type": acc_type})
        torch.cuda.empty_cache()

    if results:
        with open(cfg.PREDICTION_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "accident_time", "center_x", "center_y", "type"])
            w.writeheader()
            w.writerows(results)
        print(f"\n결과 저장 완료: {cfg.PREDICTION_PATH}")

if __name__ == "__main__":
    run_pipeline()
