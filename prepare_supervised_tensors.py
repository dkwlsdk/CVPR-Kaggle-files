"""
③ prepare_supervised_tensors.py
SIM_FEATURES/intra CSV + labels.csv → 슬라이딩 윈도우 PyTorch Tensor 생성
- labels.csv의 accident_frame 기준으로 정상(0) / 사고(1) 라벨 부착
- 사고 구간: [accident_frame - 30, 영상 끝] 전부 label=1
- 출력: SUPERVISED_TENSORS/X_train.pt, y_train.pt, meta.csv
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

# ── 경로 설정 ──────────────────────────────────────────────
INTRA_FEATURE_DIR = r"C:\Users\echin\Desktop\CVPR Kaggle\SIM_FEATURES\intra"
LABELS_CSV        = r"C:\Users\echin\Desktop\CVPR Kaggle\데이터셋\sim_dataset\labels.csv"
OUTPUT_DIR        = r"C:\Users\echin\Desktop\CVPR Kaggle\SUPERVISED_TENSORS"

# ── 하이퍼파라미터 ──────────────────────────────────────────
WINDOW_SIZE       = 30     # 슬라이딩 윈도우 크기 (프레임)
ACCIDENT_PRE_FRAMES = 30   # 사고 발생 N프레임 전부터 사고 구간으로 처리

# 수집할 피처 목록 (intra + inter 결합)
INTRA_COLS = ["velocity", "acceleration", "direction", "direction_change", "curvature", "traj_direction"]
INTER_COLS = ["relative_angle", "trajectory_angle_diff", "ttc", "distance"]
FEATURE_COLS = INTRA_COLS + INTER_COLS

# 사고 유형 → 정수 라벨 맵핑
TYPE_MAP = {
    "normal"   : 0,
    "head-on"  : 1,
    "rear-end" : 2,
    "sideswipe": 3,
    "t-bone"   : 4,
    "single"   : 5,
}

def load_labels(labels_csv: str) -> pd.DataFrame:
    """labels.csv 로드 및 video_name 키 생성."""
    df = pd.read_csv(labels_csv)
    df["video_name"] = df["rgb_path"].apply(
        lambda p: Path(p).parts[1] if len(Path(p).parts) > 1 else Path(p).stem
    )
    return df

def merge_inter_features(intra_df: pd.DataFrame, inter_df: pd.DataFrame) -> pd.DataFrame:
    """
    각 track_id에 대해 해당 프레임에서 가장 가까운 차량과의 인터랙션 피처(inter)를 결합한다.
    """
    if inter_df is None or inter_df.empty:
        for col in INTER_COLS:
            intra_df[col] = 0.0
            if col == "ttc": intra_df[col] = 9999.0
            if col == "distance": intra_df[col] = 1000.0  # 기본 먼 거리
        return intra_df

    merged_rows = []
    
    # 프레임별 처리
    for f_idx, f_intra in intra_df.groupby("frame_idx"):
        f_inter = inter_df[inter_df["frame_idx"] == f_idx]
        
        for _, i_row in f_intra.iterrows():
            tid = i_row["track_id"]
            
            # 해당 track_id가 포함된 inter 행 찾기
            rel_inter = f_inter[(f_inter["track_A"] == tid) | (f_inter["track_B"] == tid)]
            
            if not rel_inter.empty:
                # 여러 차량이 있으면 가장 가까운 것 선택
                closest = rel_inter.loc[rel_inter["distance"].idxmin()]
                for col in INTER_COLS:
                    i_row[col] = closest[col]
            else:
                # 주변 차량 없음
                for col in INTER_COLS:
                    i_row[col] = 0.0
                    if col == "ttc": i_row[col] = 9999.0
                    if col == "distance": i_row[col] = 1000.0
            
            merged_rows.append(i_row)
            
    return pd.DataFrame(merged_rows)

def create_windows(intra_df: pd.DataFrame, inter_df: pd.DataFrame, accident_frame: int, accident_type: str):
    """
    영상 피처에서 슬라이딩 윈도우를 생성하고 라벨을 부착한다.
    """
    # 1. 인터랙션 피처 결합
    df = merge_inter_features(intra_df, inter_df)

    # 2. 누락값 채우기
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

    # 3. track_id별로 시계열 생성 후 통합
    X_list, y_list, t_list = [], [], []
    accident_class = TYPE_MAP.get(accident_type, 0)
    saf_start = accident_frame - ACCIDENT_PRE_FRAMES

    for track_id, grp in df.groupby("track_id"):
        grp = grp.sort_values("frame_idx")
        if len(grp) < WINDOW_SIZE:
            continue

        data = grp[FEATURE_COLS].values  # shape: (T, N_FEATURES)

        for start in range(len(data) - WINDOW_SIZE + 1):
            seq = data[start : start + WINDOW_SIZE] 
            last_frame = grp["frame_idx"].iloc[start + WINDOW_SIZE - 1]

            if last_frame >= saf_start:
                label = 1
                type_label = accident_class
            else:
                label = 0
                type_label = TYPE_MAP["normal"]

            X_list.append(seq)
            y_list.append(label)
            t_list.append(type_label)

    return X_list, y_list, t_list


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    labels_df = load_labels(LABELS_CSV)
    print(f"labels.csv 로드 완료: {len(labels_df)}개 영상")

    # video_name을 키로 빠른 조회
    label_map = {}
    for _, row in labels_df.iterrows():
        label_map[row["video_name"]] = {
            "accident_frame": int(row["accident_frame"]),
            "type"          : str(row["type"]),
        }

    csv_files = glob.glob(os.path.join(INTRA_FEATURE_DIR, "**", "*.csv"), recursive=True)
    print(f"피처 CSV: {len(csv_files)}개")

    all_X, all_y, all_t = [], [], []
    skipped = 0

    for csv_path in tqdm(csv_files, desc="텐서 생성"):
        video_name = Path(csv_path).stem

        # labels.csv에서 매칭
        info = label_map.get(video_name)
        if info is None:
            matched = [k for k in label_map if video_name in k or k in video_name]
            if not matched:
                skipped += 1
                continue
            info = label_map[matched[0]]

        try:
            # intra 로드
            intra_df = pd.read_csv(csv_path)
            if intra_df.empty or len(intra_df) < WINDOW_SIZE:
                skipped += 1
                continue

            # inter 로드 (선택적)
            inter_path = os.path.join(os.path.dirname(os.path.dirname(csv_path)), "inter", f"{video_name}.csv")
            inter_df = None
            if os.path.exists(inter_path):
                inter_df = pd.read_csv(inter_path)

            X_l, y_l, t_l = create_windows(intra_df, inter_df, info["accident_frame"], info["type"])
            all_X.extend(X_l)
            all_y.extend(y_l)
            all_t.extend(t_l)
        except Exception as e:
            print(f"[ERROR] {video_name}: {e}")
            skipped += 1

    if not all_X:
        print("생성된 시퀀스가 없습니다.")
        return

    X = np.array(all_X, dtype=np.float32)  # (N, 30, 5)
    y = np.array(all_y, dtype=np.float32)  # (N,)
    t = np.array(all_t, dtype=np.int64)    # (N,)

    print(f"\n데이터셋 크기: X={X.shape}, y={y.shape}")
    print(f"정상 샘플: {(y==0).sum()}, 사고 샘플: {(y==1).sum()}")

    # ── Z-Score 정규화 ──
    N, W, F = X.shape
    X_flat = X.reshape(-1, F)
    mean = X_flat.mean(axis=0)
    std  = X_flat.std(axis=0)
    std[std == 0] = 1e-6
    X_scaled = ((X_flat - mean) / std).reshape(N, W, F)

    # ── 저장 ──
    torch.save(torch.tensor(X_scaled), os.path.join(OUTPUT_DIR, "X_train.pt"))
    torch.save(torch.tensor(y),        os.path.join(OUTPUT_DIR, "y_train.pt"))
    torch.save(torch.tensor(t),        os.path.join(OUTPUT_DIR, "t_train.pt"))
    np.save(os.path.join(OUTPUT_DIR, "scaler_mean.npy"), mean)
    np.save(os.path.join(OUTPUT_DIR, "scaler_std.npy"),  std)

    print(f"\n저장 완료: {OUTPUT_DIR}")
    print(f"스킵된 영상: {skipped}개")


if __name__ == "__main__":
    main()
