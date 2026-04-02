import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    FLOW_SAMPLE_FPS,
    METADATA_CSV,
    MIN_PEAK_SEP_SEC,
    TOPK_PEAKS,
    TRACK_SAMPLE_FPS,
    VEHICLE_CLASSES,
    YOLO_CONF,
    YOLO_IOU,
    YOLO_MODEL,
)
from utils import LOGGER, get_stage_paths, get_video_info, read_metadata, save_json, write_csv

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import torch
except Exception:
    torch = None


_YOLO_MODEL_CACHE = None
_YOLO_DEVICE_CACHE = None


def get_yolo_model():
    global _YOLO_MODEL_CACHE
    if YOLO is None:
        return None
    if _YOLO_MODEL_CACHE is None:
        _YOLO_MODEL_CACHE = YOLO(YOLO_MODEL)
    return _YOLO_MODEL_CACHE


def get_yolo_device():
    global _YOLO_DEVICE_CACHE
    if _YOLO_DEVICE_CACHE is not None:
        return _YOLO_DEVICE_CACHE

    if torch is not None and torch.cuda.is_available():
        _YOLO_DEVICE_CACHE = 0
        LOGGER.info("[stage1] YOLO device fixed to CUDA:0")
    else:
        _YOLO_DEVICE_CACHE = "cpu"
        LOGGER.warning("[stage1] CUDA not available. YOLO will run on CPU.")
    return _YOLO_DEVICE_CACHE


def moving_average(arr: np.ndarray, k: int = 5) -> np.ndarray:
    if len(arr) == 0:
        return arr
    k = max(1, int(k))
    pad = k // 2
    arr_pad = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(arr_pad, kernel, mode="valid")


def robust_norm(arr: np.ndarray) -> np.ndarray:
    if len(arr) == 0:
        return arr.astype(np.float32)
    p95 = float(np.percentile(arr, 95))
    if p95 <= 1e-6:
        p95 = float(arr.max()) + 1e-6
    return np.clip(arr / p95, 0.0, 1.0).astype(np.float32)


def sample_frames(video_path: str, sample_fps: float) -> List[Tuple[int, float, np.ndarray]]:
    info = get_video_info(video_path)
    native_fps = info["fps"]
    if native_fps <= 0:
        raise RuntimeError(f"Bad fps: {video_path}")
    step = max(1, int(round(native_fps / sample_fps)))
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % step == 0:
            sec = idx / native_fps
            frames.append((idx, sec, frame))
        idx += 1
    cap.release()
    return frames


def optical_flow_scores_from_samples(samples: List[Tuple[int, float, np.ndarray]]) -> List[Dict[str, Any]]:
    rows = []
    prev_gray = None
    prev_edges = None
    for frame_idx, sec, frame in samples:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 100, 200)
        flow_score = 0.0
        edge_change = 0.0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_score = float(np.mean(mag) + 0.5 * np.percentile(mag, 95))
            if prev_edges is not None:
                edge_change = float(np.mean(cv2.absdiff(prev_edges, edge)) / 255.0)
        rows.append({
            "frame_idx": frame_idx,
            "sec": sec,
            "flow_score": flow_score,
            "edge_change_score": edge_change,
        })
        prev_gray = gray
        prev_edges = edge
    return rows


def optical_flow_scores(
    video_path: str,
    sample_fps: float = FLOW_SAMPLE_FPS,
    samples: Optional[List[Tuple[int, float, np.ndarray]]] = None,
) -> List[Dict[str, Any]]:
    if samples is None:
        samples = sample_frames(video_path, sample_fps)
    return optical_flow_scores_from_samples(samples)


def _bbox_area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def tracking_scores(
    video_path: str,
    sample_fps: float = TRACK_SAMPLE_FPS,
    samples: Optional[List[Tuple[int, float, np.ndarray]]] = None,
) -> List[Dict[str, Any]]:
    if samples is None:
        samples = sample_frames(video_path, sample_fps)

    model = get_yolo_model()
    if model is None:
        LOGGER.warning("ultralytics not installed. tracking score will be zeros.")
        return [{
            "frame_idx": frame_idx,
            "sec": sec,
            "track_score": 0.0,
            "single_score": 0.0,
            "pair_score": 0.0,
            "vehicle_count": 0,
        } for frame_idx, sec, _ in samples]

    device = get_yolo_device()
    prev_state: Dict[int, Dict[str, float]] = {}
    rows = []

    for frame_idx, sec, frame in samples:
        track_score = 0.0
        single_score = 0.0
        pair_score = 0.0
        vehicle_count = 0
        current_state: Dict[int, Dict[str, float]] = {}
        detections: List[Tuple[int, np.ndarray, float, float, float]] = []

        try:
            result = model.track(
                source=frame,
                persist=True,
                verbose=False,
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                device=device,
            )[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                cls = result.boxes.cls.detach().cpu().numpy().astype(int)
                tids = result.boxes.id
                tids = tids.detach().cpu().numpy().astype(int) if tids is not None else np.arange(len(boxes))
                for box, c, tid in zip(boxes, cls, tids):
                    if c not in VEHICLE_CLASSES:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    area = _bbox_area(box)
                    vehicle_count += 1
                    current_state[int(tid)] = {"cx": cx, "cy": cy, "area": area}
                    detections.append((int(tid), box, cx, cy, area))

                    if tid in prev_state:
                        prev = prev_state[tid]
                        motion = float(((cx - prev["cx"]) ** 2 + (cy - prev["cy"]) ** 2) ** 0.5)
                        area_jump = abs(area - prev["area"]) / max(prev["area"], 1.0)
                        track_score += motion
                        single_score += 0.6 * motion + 0.4 * area_jump * 100.0

                for i in range(len(detections)):
                    _, box1, cx1, cy1, area1 = detections[i]
                    x11, y11, x12, y12 = box1.tolist()
                    for j in range(i + 1, len(detections)):
                        _, box2, cx2, cy2, area2 = detections[j]
                        x21, y21, x22, y22 = box2.tolist()

                        dx = cx1 - cx2
                        dy = cy1 - cy2
                        dist = float((dx * dx + dy * dy) ** 0.5)

                        ix1 = max(x11, x21)
                        iy1 = max(y11, y21)
                        ix2 = min(x12, x22)
                        iy2 = min(y12, y22)
                        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        union = area1 + area2 - inter + 1e-6
                        iou = inter / union
                        pair_score += max(0.0, 120.0 * iou + max(0.0, 80.0 - dist) / 80.0)

        except Exception as e:
            LOGGER.warning(f"tracking failed on {video_path} @ {sec:.3f}s: {e}")

        prev_state = current_state
        rows.append({
            "frame_idx": frame_idx,
            "sec": sec,
            "track_score": track_score,
            "single_score": single_score,
            "pair_score": pair_score,
            "vehicle_count": vehicle_count,
        })
    return rows


def pick_top_peaks(times: np.ndarray, scores: np.ndarray, topk: int, min_sep_sec: float) -> List[Tuple[float, float]]:
    order = np.argsort(scores)[::-1]
    selected: List[Tuple[float, float]] = []
    for idx in order:
        t = float(times[idx])
        s = float(scores[idx])
        if s <= 0:
            continue
        if all(abs(t - st) >= min_sep_sec for st, _ in selected):
            selected.append((t, s))
        if len(selected) >= topk:
            break
    selected.sort(key=lambda x: x[0])
    return selected


def _same_fps(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


def stage1_for_video(video_path: str) -> Dict[str, Any]:
    if _same_fps(FLOW_SAMPLE_FPS, TRACK_SAMPLE_FPS):
        shared_samples = sample_frames(video_path, FLOW_SAMPLE_FPS)
        flow_rows = optical_flow_scores_from_samples(shared_samples)
        track_rows = tracking_scores(video_path, TRACK_SAMPLE_FPS, samples=shared_samples)
    else:
        flow_samples = sample_frames(video_path, FLOW_SAMPLE_FPS)
        track_samples = sample_frames(video_path, TRACK_SAMPLE_FPS)
        flow_rows = optical_flow_scores_from_samples(flow_samples)
        track_rows = tracking_scores(video_path, TRACK_SAMPLE_FPS, samples=track_samples)

    n = min(len(flow_rows), len(track_rows))
    if n == 0:
        return {
            "candidate_times": [],
            "candidate_scores": [],
            "best_candidate_time": None,
            "best_candidate_score": None,
            "series": {},
        }

    secs = np.array([flow_rows[i]["sec"] for i in range(n)], dtype=np.float32)
    flow = np.array([flow_rows[i]["flow_score"] for i in range(n)], dtype=np.float32)
    edge = np.array([flow_rows[i]["edge_change_score"] for i in range(n)], dtype=np.float32)
    track = np.array([track_rows[i]["track_score"] for i in range(n)], dtype=np.float32)
    single = np.array([track_rows[i]["single_score"] for i in range(n)], dtype=np.float32)
    pair = np.array([track_rows[i]["pair_score"] for i in range(n)], dtype=np.float32)
    count = np.array([track_rows[i]["vehicle_count"] for i in range(n)], dtype=np.float32)

    flow_s = moving_average(flow, 5)
    edge_s = moving_average(edge, 5)
    track_s = moving_average(track, 5)
    single_s = moving_average(single, 5)
    pair_s = moving_average(pair, 5)

    flow_n = robust_norm(flow_s)
    edge_n = robust_norm(edge_s)
    track_n = robust_norm(track_s)
    single_n = robust_norm(single_s)
    pair_n = robust_norm(pair_s)

    single_gate = (count <= 1.5).astype(np.float32)
    multi_gate = (count >= 1.5).astype(np.float32)

    fused = (
        0.34 * flow_n +
        0.16 * edge_n +
        0.18 * track_n +
        0.17 * (single_n * single_gate + 0.35 * single_n * multi_gate) +
        0.15 * (pair_n * multi_gate + 0.20 * pair_n * single_gate)
    ).astype(np.float32)

    peaks = pick_top_peaks(secs, fused, TOPK_PEAKS, MIN_PEAK_SEP_SEC)

    best_candidate_time = None
    best_candidate_score = None
    if len(fused) > 0:
        best_idx = int(np.argmax(fused))
        best_candidate_time = round(float(secs[best_idx]), 3)
        best_candidate_score = round(float(fused[best_idx]), 6)

    return {
        "candidate_times": [round(t, 3) for t, _ in peaks],
        "candidate_scores": [round(s, 6) for _, s in peaks],
        "best_candidate_time": best_candidate_time,
        "best_candidate_score": best_candidate_score,
        "series": {
            "sec": secs.tolist(),
            "flow": flow.tolist(),
            "edge": edge.tolist(),
            "track": track.tolist(),
            "single": single.tolist(),
            "pair": pair.tolist(),
            "vehicle_count": count.tolist(),
            "fused": fused.tolist(),
        },
    }


def save_stage1_outputs(rel_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
    paths = get_stage_paths(rel_path)
    cands = result["candidate_times"]
    row = {
        "path": rel_path,
        "best_candidate_time": result.get("best_candidate_time", ""),
        "best_candidate_score": result.get("best_candidate_score", ""),
        "candidate_1": cands[0] if len(cands) > 0 else "",
        "candidate_2": cands[1] if len(cands) > 1 else "",
        "candidate_3": cands[2] if len(cands) > 2 else "",
    }

    with open(paths["stage1_scores_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time", "flow_score", "edge_change_score", "track_score",
                "single_score", "pair_score", "vehicle_count", "fused_score",
            ],
        )
        writer.writeheader()
        series = result.get("series", {})
        for vals in zip(
            series.get("sec", []), series.get("flow", []), series.get("edge", []), series.get("track", []),
            series.get("single", []), series.get("pair", []), series.get("vehicle_count", []), series.get("fused", [])
        ):
            t, fl, ed, tr, sg, pr, vc, fu = vals
            writer.writerow({
                "time": round(float(t), 3),
                "flow_score": round(float(fl), 6),
                "edge_change_score": round(float(ed), 6),
                "track_score": round(float(tr), 6),
                "single_score": round(float(sg), 6),
                "pair_score": round(float(pr), 6),
                "vehicle_count": round(float(vc), 3),
                "fused_score": round(float(fu), 6),
            })

    save_json(paths["stage1_candidates_json"], {
        "path": rel_path,
        "best_candidate_time": result.get("best_candidate_time"),
        "best_candidate_score": result.get("best_candidate_score"),
        "candidate_times": result["candidate_times"],
        "candidate_scores": result["candidate_scores"],
    })
    save_json(paths["stage1_summary_json"], {
        "path": rel_path,
        "note": "Candidates are heuristic hints only. Single-vehicle crashes are explicitly supported.",
        "best_candidate_time": result.get("best_candidate_time"),
        "best_candidate_score": result.get("best_candidate_score"),
        "candidate_times": result["candidate_times"],
        "candidate_scores": result["candidate_scores"],
        "num_points": len(result.get("series", {}).get("sec", [])),
        "scores_csv": str(paths["stage1_scores_csv"]),
    })
    return row


def process_one_video(rel_path: str, video_path: str) -> Dict[str, Any]:
    LOGGER.info(f"[stage1] {rel_path}")
    result = stage1_for_video(video_path)
    row = save_stage1_outputs(rel_path, result)
    return {
        **row,
        "candidate_times": result["candidate_times"],
        "candidate_scores": result["candidate_scores"],
        "best_candidate_time": result.get("best_candidate_time"),
        "best_candidate_score": result.get("best_candidate_score"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=str(METADATA_CSV))
    parser.add_argument("--videos-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows_out = []
    metas = list(read_metadata(Path(args.metadata)))
    if args.limit is not None:
        metas = metas[: max(0, args.limit)]

    for meta in metas:
        rel_path = meta.get("path", "")
        if not rel_path:
            continue
        video_path = str(Path(args.videos_root) / rel_path) if args.videos_root else str(Path(args.metadata).parent / rel_path)
        try:
            out = process_one_video(rel_path, video_path)
            rows_out.append({
                k: out.get(k, "")
                for k in ["path", "best_candidate_time", "best_candidate_score", "candidate_1", "candidate_2", "candidate_3"]
            })
        except Exception as e:
            LOGGER.exception(f"stage1 failed: {rel_path} | {e}")
            rows_out.append({
                "path": rel_path,
                "best_candidate_time": "",
                "best_candidate_score": "",
                "candidate_1": "",
                "candidate_2": "",
                "candidate_3": "",
            })

    final_csv = Path(args.metadata).parent / "outputs" / "stage1_candidate_times.csv"
    write_csv(final_csv, rows_out, ["path", "best_candidate_time", "best_candidate_score", "candidate_1", "candidate_2", "candidate_3"])
    LOGGER.info(f"saved: {final_csv}")


if __name__ == "__main__":
    main()
