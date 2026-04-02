import json
import importlib.util
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

base_dir = Path(__file__).resolve().parents[1]
config_override = os.environ.get("QWEN3_CONFIG_PATH")
config_file = Path(config_override).resolve() if config_override else (base_dir / "config" / "config.py").resolve()
config_spec = importlib.util.spec_from_file_location("_local_config", config_file)
if config_spec is None or config_spec.loader is None:
    raise ImportError("cannot load config.py")
_cfg = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(_cfg)

VALID_TYPES = _cfg.VALID_TYPES


TYPE_ALIASES = {
    "rear end": "rear-end",
    "rear_end": "rear-end",
    "rear-ended": "rear-end",
    "head on": "head-on",
    "head_on": "head-on",
    "side swipe": "sideswipe",
    "side-swipe": "sideswipe",
    "tbone": "t-bone",
    "t bone": "t-bone",
}


TIME_KEY_RE = re.compile(r'"(?:accident_time|time)"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
CENTER_X_RE = re.compile(r'"center_x"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
CENTER_Y_RE = re.compile(r'"center_y"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
CONFIDENCE_RE = re.compile(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
TYPE_RE = re.compile(r'"type"\s*:\s*"([^"]+)"')
REASONING_RE = re.compile(r'"reasoning"\s*:\s*"([^"]*)"')
WHY_RE = re.compile(r'"why"\s*:\s*"([^"]*)"')
TEXT_TIME_RE = re.compile(r'(?i)\bt\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*s\b|\b([0-9]+(?:\.[0-9]+)?)\s*(?:sec|secs|second|seconds)\b')


@dataclass
class Prediction:
    path: str
    time: float
    center_x: float
    center_y: float
    type: str
    confidence: float
    why: str
    raw: str
    issues: str
    parse_stage: str
    time_source: str
    fallback_used: int


def extract_json(text: str) -> Dict[str, Any]:
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    plain = re.search(r"(\{.*?\})", text, re.DOTALL)
    if plain:
        return json.loads(plain.group(1))
    raise ValueError("json not found")


def repair_json_text(text: str) -> str:
    repaired = text.strip()
    repaired = re.sub(r"^```json", "", repaired, flags=re.IGNORECASE).strip()
    repaired = re.sub(r"^```", "", repaired).strip()
    repaired = re.sub(r"```$", "", repaired).strip()
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"([{,]\s*)'([^']+)'\s*:", r'\1"\2":', repaired)
    repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)
    return repaired


def try_extract_json(clean: str) -> Tuple[Dict[str, Any], str, bool]:
    try:
        return extract_json(clean), "strict", True
    except (ValueError, json.JSONDecodeError):
        pass

    repaired = repair_json_text(clean)
    try:
        return extract_json(repaired), "repair", True
    except (ValueError, json.JSONDecodeError):
        return {}, "failed", False


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def salvage_from_text(clean: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    m = TIME_KEY_RE.search(clean)
    if m:
        out["accident_time"] = safe_float(m.group(1))

    mx = CENTER_X_RE.search(clean)
    my = CENTER_Y_RE.search(clean)
    if mx and my:
        out["center_x"] = safe_float(mx.group(1))
        out["center_y"] = safe_float(my.group(1))

    mc = CONFIDENCE_RE.search(clean)
    if mc:
        out["confidence"] = safe_float(mc.group(1))

    mt = TYPE_RE.search(clean)
    if mt:
        out["type"] = mt.group(1)

    mr = REASONING_RE.search(clean)
    if mr:
        out["reasoning"] = mr.group(1)
    else:
        mw = WHY_RE.search(clean)
        if mw:
            out["why"] = mw.group(1)

    return out


def extract_time_from_reasoning(text: str) -> float:
    m = TEXT_TIME_RE.search(text)
    if not m:
        return -1.0
    g1 = m.group(1)
    g2 = m.group(2)
    return safe_float(g1 if g1 is not None else g2)


def normalize_type(value: Any) -> str:
    if not isinstance(value, str):
        return "single"
    text = TYPE_ALIASES.get(value.strip().lower(), value.strip().lower())
    text = text.replace("_", "-").strip()
    if not text:
        return "single"
    if VALID_TYPES and text not in VALID_TYPES:
        return "single"
    return text


def normalize_coordinate(raw: Any) -> Tuple[float, float]:
    if not isinstance(raw, list) or not raw:
        return 0.5, 0.5
    if isinstance(raw[0], list):
        points = [p for p in raw if isinstance(p, list) and len(p) >= 2]
        if not points:
            return 0.5, 0.5
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    if len(raw) >= 2:
        return float(raw[0]), float(raw[1])
    return 0.5, 0.5


def detect_issues(
    clean: str,
    parsed_ok: bool,
    time_value: float,
    center_x: float,
    center_y: float,
    confidence: float,
) -> List[str]:
    issues: List[str] = []
    lowered = clean.lower()
    if not parsed_ok:
        issues.append("parse_failed")
    if "no crash" in lowered or "no accident" in lowered or "not detectable" in lowered:
        issues.append("forbidden_no_crash")
    if time_value <= 0.01:
        issues.append("time_zero")
    if abs(center_x - 0.5) < 1e-6 and abs(center_y - 0.5) < 1e-6:
        issues.append("center_default")
    if confidence < 0.2:
        issues.append("low_confidence")
    return issues


def parse_output(
    raw: str, video_path: str, duration: float, video_root: str
) -> Dict[str, Any]:
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    data, parse_stage, parsed_ok = try_extract_json(clean)
    if not parsed_ok:
        data = salvage_from_text(clean)
        if data:
            parse_stage = "salvage"

    fallback_used = 0
    fallback_reason = ""

    time_source = "json"
    if "accident_time" in data or "time" in data:
        time_value = safe_float(data.get("accident_time", data.get("time", 0.0)))
    else:
        reason_text = str(data.get("reasoning", data.get("why", clean))).strip()
        reason_time = extract_time_from_reasoning(reason_text)
        if reason_time >= 0.0:
            time_value = reason_time
            time_source = "reasoning"
            fallback_used = 1
            fallback_reason = "time_from_reasoning"
        elif duration > 0.0:
            time_value = duration * 0.15
            time_source = "duration_ratio_15"
            fallback_used = 1
            fallback_reason = "no_time_found"
        else:
            time_value = 0.0
            time_source = "zero_fallback"
            fallback_used = 1
            fallback_reason = "invalid_duration"

    if not parsed_ok and time_source == "json":
        time_source = "salvage"
        fallback_used = 1
        if not fallback_reason:
            fallback_reason = "json_parse_failed"

    time_value = max(0.0, min(duration, time_value))

    if "center_x" in data and "center_y" in data:
        center_x = safe_float(data.get("center_x", 0.5) or 0.5)
        center_y = safe_float(data.get("center_y", 0.5) or 0.5)
    else:
        center_x, center_y = normalize_coordinate(data.get("coordinate", [0.5, 0.5]))
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))

    confidence = safe_float(data.get("confidence", 0.0) or 0.0)
    confidence = max(0.0, min(1.0, confidence))

    collision_type = normalize_type(data.get("type", "single"))
    why = str(data.get("reasoning", data.get("why", ""))).strip()

    issues = detect_issues(clean, parsed_ok, time_value, center_x, center_y, confidence)
    if fallback_reason:
        issues.append(f"fallback:{fallback_reason}")

    video_root_path = Path(video_root).parent.resolve()
    resolved_path = Path(video_path).resolve()
    try:
        relative_path = resolved_path.relative_to(video_root_path)
    except ValueError:
        relative_path = resolved_path.name

    prediction = Prediction(
        path=str(relative_path),
        time=round(time_value, 2),
        center_x=round(center_x, 3),
        center_y=round(center_y, 3),
        type=collision_type,
        confidence=round(confidence, 3),
        why=why,
        raw=clean,
        issues=",".join(issues),
        parse_stage=parse_stage,
        time_source=time_source,
        fallback_used=fallback_used,
    )
    return asdict(prediction)
