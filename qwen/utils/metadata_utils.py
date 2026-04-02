import csv
import importlib.util
import os
from pathlib import Path
from typing import Any, Dict

base_dir = Path(__file__).resolve().parents[1]
config_override = os.environ.get("QWEN3_CONFIG_PATH")
config_file = Path(config_override).resolve() if config_override else (base_dir / "config" / "config.py").resolve()
config_spec = importlib.util.spec_from_file_location("_local_config", config_file)
if config_spec is None or config_spec.loader is None:
    raise ImportError("cannot load config.py")
_cfg = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(_cfg)

DEFAULT_DURATION = _cfg.DEFAULT_DURATION
MAX_FRAMES = _cfg.MAX_FRAMES
MIN_FRAMES = _cfg.MIN_FRAMES
TARGET_FPS = _cfg.TARGET_FPS


def to_float_or(value: Any, default: float) -> float:
    # 값이 None이면 기본값(default)을 반환하고, 그렇지 않으면 실수형(float)으로 변환하여 반환합니다.
    return float(default if value is None else value)


def to_int_or(value: Any, default: int) -> int:
    # 값이 None이면 기본값(default)을 반환하고, 그렇지 않으면 정수형(int)으로 변환하여 반환합니다.
    return int(default if value is None else value)


def load_metadata(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """Load metadata csv to {path: metadata} dict."""
    output: Dict[str, Dict[str, Any]] = {}
    # CSV 파싱을 pandas 대신 표준 라이브러리로 처리해 환경별 타입 이슈를 줄입니다.
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = str(row.get("path", "")).strip()
            if not key:
                continue
            # 각 행의 데이터를 파싱하여 딕셔너리에 저장합니다. 값이 없을 경우 기본값을 사용합니다.
            output[key] = {
                "duration": to_float_or(row.get("duration"), DEFAULT_DURATION),
                "no_frames": to_int_or(row.get("no_frames"), 0),
                "height": to_int_or(row.get("height"), 720),
                "width": to_int_or(row.get("width"), 1280),
                "quality": str(row.get("quality", "unknown")),
            }
    return output


def compute_adaptive_fps(
    duration: float, no_frames: int, height: int, width: int
) -> float:
    """Metadata-based adaptive fps with stable bounds."""
    # 비디오의 지속 시간에 따라 모델 추론에 적합한 가변 FPS(초당 프레임 수)를 계산합니다.
    # 사용하지 않는 매개변수들은 무시합니다.
    del no_frames, height, width

    # 영상 길이가 0 이하인 경우 기본 대상 FPS(TARGET_FPS)를 반환합니다.
    if duration <= 0:
        return TARGET_FPS

    fps = TARGET_FPS
    # 영상 추출 시 예상되는 전체 프레임 수(duration * fps)가 최대 프레임 수(MAX_FRAMES)를 초과하는 경우
    # 최대 프레임 수에 맞게 FPS를 낮춥니다.
    if duration * fps > MAX_FRAMES:
        fps = MAX_FRAMES / duration
    # 영상 추출 시 예상되는 전체 프레임 수가 최소 프레임 수(MIN_FRAMES)보다 적은 경우
    # 최소 프레임 수를 맞추되, 목표 FPS(TARGET_FPS) 상한은 유지합니다.
    if duration * fps < MIN_FRAMES:
        fps = min(MIN_FRAMES / duration, TARGET_FPS)

    # 최종 FPS는 0.5 ~ TARGET_FPS 범위로 제한합니다.
    return round(max(0.5, min(TARGET_FPS, fps)), 2)
