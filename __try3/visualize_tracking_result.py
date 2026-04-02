import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2


TRACKING_JSON_PATH = Path(
    "/root/Desktop/workspace/ja/CVPR-Kaggle-files/Detection_Dataset/test_tracking/_FjROQyb1C8_1_00.json"
)
VIDEO_PATH = Path(
    "/root/Desktop/workspace/ja/CVPR-Kaggle-files/데이터셋/test_videos/_FjROQyb1C8_1_00.mp4"
)
OUTPUT_PATH = Path(
    "/root/Desktop/workspace/ja/qwen_single_video_test_output/tracking_single test/_FjROQyb1C8_1_00_tracking_overlay.mp4"
)
TRAIL_LENGTH = 20
DRAW_UNTRACKED = False


def load_tracking_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"Invalid tracking JSON, missing frames list: {json_path}")
    return data


def build_frame_map(data: dict) -> Dict[int, List[dict]]:
    frame_map: Dict[int, List[dict]] = {}
    for frame in data.get("frames", []):
        frame_idx = int(frame.get("frame_idx", 0))
        frame_map[frame_idx] = frame.get("detections", [])
    return frame_map


def color_for_track(track_id: Optional[int]) -> Tuple[int, int, int]:
    if track_id is None:
        return (180, 180, 180)

    seed = int(track_id) * 2654435761 % 0xFFFFFF
    b = 80 + (seed & 0x7F)
    g = 80 + ((seed >> 8) & 0x7F)
    r = 80 + ((seed >> 16) & 0x7F)
    return (int(b), int(g), int(r))


def draw_label(frame, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - text_h - baseline - 6)
    cv2.rectangle(frame, (x, top), (x + text_w + 8, y), color, -1)
    cv2.putText(
        frame,
        text,
        (x + 4, y - 4),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_header(frame, frame_idx: int, total_frames: int, fps: float, num_detections: int) -> None:
    current_time = frame_idx / fps if fps > 0 else 0.0
    lines = [
        f"frame: {frame_idx + 1}/{total_frames}",
        f"time: {current_time:.2f}s",
        f"detections: {num_detections}",
    ]

    x, y = 16, 18
    line_h = 26
    box_w = 250
    box_h = 18 + line_h * len(lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (24, 24, 24), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x + 12, y + 28 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def visualize_tracking(
    tracking_json_path: Path,
    video_path: Path,
    output_path: Path,
    trail_length: int,
    draw_untracked: bool,
) -> Path:
    data = load_tracking_json(tracking_json_path)
    frame_map = build_frame_map(data)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS for video: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video: {output_path}")

    history: Dict[int, Deque[Tuple[int, int]]] = defaultdict(lambda: deque(maxlen=trail_length))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        json_frame_idx = frame_idx + 1
        detections = frame_map.get(json_frame_idx, [])

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None and not draw_untracked:
                continue

            bbox = det.get("bbox_xyxy", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            color = color_for_track(track_id)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, center, 3, color, -1)

            if track_id is not None:
                track_id = int(track_id)
                history[track_id].append(center)
                points = list(history[track_id])
                for i in range(1, len(points)):
                    cv2.line(frame, points[i - 1], points[i], color, 2)

            class_name = str(det.get("class_name", "obj"))
            confidence = float(det.get("confidence", 0.0))
            track_text = "untracked" if track_id is None else f"id {track_id}"
            label = f"{track_text} | {class_name} | {confidence:.2f}"
            draw_label(frame, label, x1, y1, color)

        draw_header(frame, frame_idx, total_frames, fps, len(detections))
        writer.write(frame)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"processing {frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()
    return output_path


def main() -> None:
    tracking_json_path = TRACKING_JSON_PATH
    video_path = VIDEO_PATH
    output_path = OUTPUT_PATH

    if not tracking_json_path.exists():
        raise FileNotFoundError(f"Tracking JSON not found: {tracking_json_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    saved_path = visualize_tracking(
        tracking_json_path=tracking_json_path,
        video_path=video_path,
        output_path=output_path,
        trail_length=max(1, TRAIL_LENGTH),
        draw_untracked=DRAW_UNTRACKED,
    )
    print(f"saved overlay video: {saved_path}")


if __name__ == "__main__":
    main()
