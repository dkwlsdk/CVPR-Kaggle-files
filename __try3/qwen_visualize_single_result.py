import argparse
import json
import os
from pathlib import Path
from typing import Optional

import cv2


DEFAULT_RESULT_JSON = "/root/Desktop/workspace/ja/qwen_single_video_test_output/_FjROQyb1C8_1_00.json"


def load_result(result_json_path: str) -> dict:
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("status") == "failed":
        stage = data.get("failed_stage", "unknown")
        error = data.get("error", "unknown error")
        raise ValueError(f"result.json indicates failure at stage '{stage}': {error}")

    result = data.get("result")
    if not isinstance(result, dict):
        raise ValueError("Missing 'result' object in result.json")

    return data


def resolve_output_path(output_path: str, video_path: str) -> str:
    if output_path:
        return output_path

    video_stem = Path(video_path).stem
    return str(Path(DEFAULT_RESULT_JSON).parent / f"{video_stem}_qwen_overlay.mp4")


def draw_overlay(
    frame,
    center_px: int,
    center_py: int,
    current_time: float,
    accident_time: float,
    accident_type: str,
    is_accident_frame: bool,
) -> None:
    if is_accident_frame:
        marker_color = (0, 0, 255)
        ring_radius = 36
        thickness = 3
    elif current_time >= accident_time:
        marker_color = (0, 140, 255)
        ring_radius = 28
        thickness = 2
    else:
        marker_color = (0, 255, 255)
        ring_radius = 22
        thickness = 2

    cv2.circle(frame, (center_px, center_py), ring_radius, marker_color, thickness)
    cv2.line(frame, (center_px - 16, center_py), (center_px + 16, center_py), marker_color, 2)
    cv2.line(frame, (center_px, center_py - 16), (center_px, center_py + 16), marker_color, 2)
    cv2.circle(frame, (center_px, center_py), 4, marker_color, -1)

    info_lines = [
        f"Current time: {current_time:.2f}s",
        f"Pred accident_time: {accident_time:.2f}s",
        f"Pred type: {accident_type}",
        f"Pred center: ({center_px}, {center_py})",
    ]
    if is_accident_frame:
        info_lines.append("Predicted impact frame")

    box_x, box_y = 16, 18
    line_height = 28
    box_w = 360
    box_h = 18 + line_height * len(info_lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for idx, text in enumerate(info_lines):
        text_y = box_y + 28 + idx * line_height
        cv2.putText(
            frame,
            text,
            (box_x + 12, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def annotate_key_frame(frame, center_px: int, center_py: int, accident_time: float, accident_type: str) -> None:
    draw_overlay(
        frame=frame,
        center_px=center_px,
        center_py=center_py,
        current_time=accident_time,
        accident_time=accident_time,
        accident_type=accident_type,
        is_accident_frame=True,
    )


def visualize_result(result_json_path: str, output_path: Optional[str] = None) -> str:
    data = load_result(result_json_path)

    video_path = data["video_path"]
    result = data["result"]

    accident_time = float(result["accident_time"])
    center_x = float(result["center_x"])
    center_y = float(result["center_y"])
    accident_type = str(result.get("type", "unknown"))

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS for video: {video_path}")

    accident_frame_index = int(round(accident_time * fps))
    accident_frame_index = max(0, min(accident_frame_index, max(0, total_frames - 1)))

    center_px = max(0, min(int(round(center_x * width)), max(0, width - 1)))
    center_py = max(0, min(int(round(center_y * height)), max(0, height - 1)))

    output_path = resolve_output_path(output_path, video_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter: {output_path}")

    key_frame_path = str(Path(output_path).with_name(f"{Path(output_path).stem}_keyframe.jpg"))
    key_frame_saved = False

    print(f"Video path: {video_path}")
    print(f"Result JSON: {result_json_path}")
    print(f"Output video: {output_path}")
    print(f"Accident frame index: {accident_frame_index}")
    print(f"Overlay pixel center: ({center_px}, {center_py})")

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_index / fps
        is_accident_frame = frame_index == accident_frame_index

        draw_overlay(
            frame=frame,
            center_px=center_px,
            center_py=center_py,
            current_time=current_time,
            accident_time=accident_time,
            accident_type=accident_type,
            is_accident_frame=is_accident_frame,
        )

        if is_accident_frame and not key_frame_saved:
            key_frame = frame.copy()
            annotate_key_frame(key_frame, center_px, center_py, accident_time, accident_type)
            cv2.imwrite(key_frame_path, key_frame)
            key_frame_saved = True

        writer.write(frame)
        frame_index += 1

        if frame_index % 50 == 0:
            print(f"  processing: {frame_index}/{total_frames} frames")

    cap.release()
    writer.release()

    print(f"Saved overlay video to: {output_path}")
    if key_frame_saved:
        print(f"Saved key frame image to: {key_frame_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay Qwen single-video result.json predictions on the original video."
    )
    parser.add_argument(
        "--result-json",
        default=DEFAULT_RESULT_JSON,
        help="Path to qwen_single_video_test_output/result.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output mp4 path. Defaults to qwen_single_video_test_output/<video>_qwen_overlay.mp4",
    )
    args = parser.parse_args()

    visualize_result(args.result_json, args.output)


if __name__ == "__main__":
    main()
