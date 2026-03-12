"""
① json_to_tracking_csv.py
YOLO_SIM JSON 파일들을 TRACKING CSV 형식으로 변환한다.
- track_id=null인 탐지 결과는 자동 필터링
- 출력: SIM_TRACKING/{accident_type}/{video_name}.csv
"""
import os
import json
import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── 경로 설정 ──────────────────────────────────────────────
YOLO_SIM_DIR  = r"C:\Users\echin\Desktop\CVPR Kaggle\YOLO_SIM"
OUTPUT_DIR    = r"C:\Users\echin\Desktop\CVPR Kaggle\SIM_TRACKING"
IMAGE_WIDTH   = 1920
IMAGE_HEIGHT  = 1080

def convert_single_json(json_path: Path, output_dir: str) -> bool:
    """JSON 1개를 TRACKING 호환 CSV로 변환한다."""
    accident_type = json_path.parent.name  # 폴더명 = 사고 유형 (head-on 등)
    video_name    = json_path.stem         # 확장자 없는 파일 이름

    # 출력 경로 (사고 유형별 하위 폴더 유지)
    out_subdir = os.path.join(output_dir, accident_type)
    os.makedirs(out_subdir, exist_ok=True)
    out_path = os.path.join(out_subdir, f"{video_name}.csv")

    if os.path.exists(out_path):
        return True  # 이미 변환된 파일은 건너뜀

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] JSON 읽기 실패 {json_path.name}: {e}")
        return False

    rows = []
    null_skipped = 0

    for frame in data.get("frames", []):
        frame_idx = frame.get("frame_idx", 0)
        for det in frame.get("detections", []):
            track_id = det.get("track_id")

            # ── 핵심: track_id=null 필터링 ──
            if track_id is None:
                null_skipped += 1
                continue

            bbox = det.get("bbox_xyxy", [0, 0, 0, 0])
            rows.append({
                "frame_idx" : frame_idx,
                "track_id"  : int(track_id),
                "x1"        : bbox[0],
                "y1"        : bbox[1],
                "x2"        : bbox[2],
                "y2"        : bbox[3],
                "class_id"  : det.get("class_id", 0),
                "video_name": video_name,
            })

    if not rows:
        return False  # 탐지 결과 없음

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    if null_skipped > 0:
        print(f"  [INFO] {video_name}: null track_id {null_skipped}개 제외됨")

    return True


def main():
    json_files = sorted(Path(YOLO_SIM_DIR).rglob("*.json"))
    print(f"변환 대상 JSON: {len(json_files)}개")

    success, fail = 0, 0
    for json_path in tqdm(json_files, desc="JSON → CSV 변환"):
        ok = convert_single_json(json_path, OUTPUT_DIR)
        if ok:
            success += 1
        else:
            fail += 1

    print(f"\n완료: 성공={success}, 실패={fail}")
    print(f"출력 폴더: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
