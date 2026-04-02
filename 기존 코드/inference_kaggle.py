"""
⑤ inference_kaggle.py
훈련된 LSTM 분류기로 FEATURES/intra + inter 데이터에 추론하여
Kaggle 제출 형식의 submission.csv를 생성한다.

출력 형식:
  path,accident_time,center_x,center_y,type
  videos/-2UPLUV7JLg_00.mp4,25.5,0.308,0.211,head-on
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from typing import Optional

# ── 경로 설정 보강 ──────────────────────────────────────────
def find_path(target_rel_path, description="File/Dir"):
    """
    스크립트 위치, 현재 작업 디렉토리, 상위 디렉토리 등에서 대상을 찾습니다.
    """
    possible_roots = [
        os.path.dirname(os.path.abspath(__file__)), # 스크립트 위치
        os.getcwd(),                                 # 현재 작업 디렉토리
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), # 상위 디렉토리
        "/root/Desktop/workspace/ja/CVPR-Kaggle-files",
        "/root/Desktop/workspace/ja"
    ]
    
    for root in possible_roots:
        full_path = os.path.join(root, target_rel_path)
        if os.path.exists(full_path):
            print(f"Found {description}: {full_path}")
            return full_path
            
    # 못 찾은 경우 기본값 반환 (이후 에러 발생 시 경로 확인 용도)
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), target_rel_path)
    print(f"Warning: Could not find {description} in common locations. Using default: {default_path}")
    return default_path

INTRA_DIR   = find_path(os.path.join("FEATURES", "intra"), "Intra Features Dir")
INTER_DIR   = find_path(os.path.join("FEATURES", "inter"), "Inter Features Dir")
TENSOR_DIR  = find_path("SUPERVISED_TENSORS", "Tensor Dir")
MODEL_PATH  = find_path(os.path.join("models", "lstm_sim_classifier.pth"), "Model File")
OUTPUT_CSV  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission_dl.csv")

# ── 하이퍼파라미터 (학습 시와 동일하게) ──────────────────────
WINDOW_SIZE  = 30
HIDDEN_DIM   = 128
NUM_LAYERS   = 2
NUM_CLASSES  = 6
DROPOUT      = 0.3
# 수집할 피처 목록 (학습 시와 동일하게)
INTRA_COLS = ["velocity", "acceleration", "direction", "direction_change", "curvature", "traj_direction"]
INTER_COLS = ["relative_angle", "trajectory_angle_diff", "ttc", "distance"]
FEATURE_COLS = INTRA_COLS + INTER_COLS
THRESHOLD    = 0.5   # 사고 판별 확률 임계값

IMAGE_WIDTH  = 1920
IMAGE_HEIGHT = 1080

# 정수 → 사고 유형 문자열
TYPE_NAMES = {0: "normal", 1: "head-on", 2: "rear-end",
              3: "sideswipe", 4: "t-bone", 5: "single"}
DEFAULT_FPS = 30.0


# ── 모델 클래스 (train_lstm_classifier.py와 동일) ──────────────
class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout     = nn.Dropout(dropout)
        self.binary_head = nn.Linear(hidden_dim, 1)
        self.multi_head  = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        feat = self.dropout(hidden[-1])
        return self.binary_head(feat).squeeze(1), self.multi_head(feat)


def load_scaler():
    """학습 시 저장한 Z-Score 파라미터 로드."""
    mean = np.load(os.path.join(TENSOR_DIR, "scaler_mean.npy"))
    std  = np.load(os.path.join(TENSOR_DIR, "scaler_std.npy"))
    return mean, std


def normalize(X: np.ndarray, mean, std) -> np.ndarray:
    N, W, F = X.shape
    X_flat = X.reshape(-1, F)
    X_norm = (X_flat - mean) / std
    return X_norm.reshape(N, W, F)


def merge_inter_features(intra_df: pd.DataFrame, inter_df: pd.DataFrame) -> pd.DataFrame:
    """각 track_id에 대해 해당 프레임에서 가장 가까운 차량과의 인터랙션 피처 결합."""
    if inter_df is None or inter_df.empty:
        for col in INTER_COLS:
            intra_df[col] = 0.0
            if col == "ttc": intra_df[col] = 9999.0
            if col == "distance": intra_df[col] = 1000.0
        return intra_df

    merged_rows = []
    for f_idx, f_intra in intra_df.groupby("frame_idx"):
        f_inter = inter_df[inter_df["frame_idx"] == f_idx]
        for _, i_row in f_intra.iterrows():
            tid = i_row["track_id"]
            rel_inter = f_inter[(f_inter["track_A"] == tid) | (f_inter["track_B"] == tid)]
            if not rel_inter.empty:
                closest = rel_inter.loc[rel_inter["distance"].idxmin()]
                for col in INTER_COLS: i_row[col] = closest[col]
            else:
                for col in INTER_COLS:
                    i_row[col] = 0.0
                    if col == "ttc": i_row[col] = 9999.0
                    if col == "distance": i_row[col] = 1000.0
            merged_rows.append(i_row)
    return pd.DataFrame(merged_rows)


def sliding_windows(intra_df: pd.DataFrame, inter_df: pd.DataFrame) -> tuple:
    """intra + inter 결합 피처에서 슬라이딩 윈도우 생성."""
    # 1. 인터랙션 피처 결합
    df = merge_inter_features(intra_df, inter_df)

    # 2. 누락값 채우기
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

    X_list, frames_list = [], []
    for _, grp in df.groupby("track_id"):
        grp = grp.sort_values("frame_idx")
        if len(grp) < WINDOW_SIZE:
            continue
        data = grp[FEATURE_COLS].values
        frame_idxs = grp["frame_idx"].values
        for start in range(len(data) - WINDOW_SIZE + 1):
            X_list.append(data[start:start + WINDOW_SIZE])
            frames_list.append(frame_idxs[start + WINDOW_SIZE - 1])

    return np.array(X_list, dtype=np.float32), np.array(frames_list, dtype=int)


def get_center_xy(intra_df: pd.DataFrame, inter_df: pd.DataFrame,
                  peak_frame: int, predicted_type: str) -> tuple:
    """
    사고 유형에 따라 center_x, center_y를 산출한다.
    - single 사고: intra에서 가속도 이상이 가장 큰 차량의 중심점
    - 나머지    : inter에서 거리가 가장 가까운 두 차량의 중심점 평균
    """
    # single 또는 inter 데이터가 없는 경우
    peak_inter = pd.DataFrame()
    if inter_df is not None and not inter_df.empty:
        peak_inter = inter_df[inter_df["frame_idx"] == peak_frame]

    if predicted_type == "single" or peak_inter.empty:
        # intra에서 해당 프레임, 가속도 이상 최대 차량
        peak_intra = intra_df[intra_df["frame_idx"] == peak_frame]
        if peak_intra.empty:
            return 0.5, 0.5
        row = peak_intra.loc[peak_intra["acceleration"].abs().idxmax()]
        cx = (row["x1"] + row["x2"]) / 2 / IMAGE_WIDTH
        cy = (row["y1"] + row["y2"]) / 2 / IMAGE_HEIGHT
        return float(cx), float(cy)
    else:
        # 거리 최소 쌍의 두 차량 중심점 평균
        closest = peak_inter.loc[peak_inter["distance"].idxmin()]
        # intra에서 두 track의 중심 좌표 조회
        def get_cx_cy(track_id):
            rows = intra_df[(intra_df["frame_idx"] == peak_frame) &
                            (intra_df["track_id"] == track_id)]
            if rows.empty:
                return 0.5 * IMAGE_WIDTH, 0.5 * IMAGE_HEIGHT
            r = rows.iloc[0]
            return (r["x1"] + r["x2"]) / 2, (r["y1"] + r["y2"]) / 2

        cx_a, cy_a = get_cx_cy(int(closest["track_A"]))
        cx_b, cy_b = get_cx_cy(int(closest["track_B"]))
        return ((cx_a + cx_b) / 2) / IMAGE_WIDTH, ((cy_a + cy_b) / 2) / IMAGE_HEIGHT


def infer_single_video(video_name: str, model, device, mean, std) -> Optional[dict]:
    """영상 1개에 대해 추론하여 submission 행을 반환한다."""
    intra_path = os.path.join(INTRA_DIR, f"{video_name}.csv")
    inter_path = os.path.join(INTER_DIR, f"{video_name}.csv")

    if not os.path.exists(intra_path):
        return None

    intra_df = pd.read_csv(intra_path)
    inter_df = pd.read_csv(inter_path) if os.path.exists(inter_path) else None

    X_raw, frame_idxs = sliding_windows(intra_df, inter_df)
    if len(X_raw) == 0:
        return None

    # 정규화
    X_norm = normalize(X_raw, mean, std)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)

    with torch.no_grad():
        bin_logit, multi_logit = model(X_tensor)
        probs  = torch.sigmoid(bin_logit).cpu().numpy()    # (N,)
        types  = multi_logit.argmax(dim=1).cpu().numpy()   # (N,)

    # 사고 확률이 최대인 윈도우의 마지막 프레임
    peak_idx   = int(np.argmax(probs))
    peak_prob  = float(probs[peak_idx])
    peak_frame = int(frame_idxs[peak_idx])
    peak_type  = TYPE_NAMES.get(int(types[peak_idx]), "rear-end")

    if peak_prob < THRESHOLD:
        # 사고 미탐지: 가장 높은 확률 프레임으로 일단 기록 (대회 요건)
        pass

    # FPS 추정 (intra_df에 fps 컬럼이 없으므로 기본값 또는 파싱)
    fps = DEFAULT_FPS

    accident_time = round(peak_frame / fps, 1)
    center_x, center_y = get_center_xy(intra_df, inter_df, peak_frame, peak_type)

    return {
        "path"         : f"videos/{video_name}.mp4",
        "accident_time": accident_time,
        "center_x"     : round(center_x, 3),
        "center_y"     : round(center_y, 3),
        "type"         : peak_type,
    }


def main():
    # ── 모델 로드 ──
    print("모델 로드 중...")
    mean, std = load_scaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    n_features = len(FEATURE_COLS)
    model = LSTMClassifier(n_features, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    print(f"모델 로드 완료: {MODEL_PATH}")

    # ── 전체 영상 추론 ──
    intra_files = glob.glob(os.path.join(INTRA_DIR, "*.csv"))
    print(f"\n추론 대상 영상: {len(intra_files)}개")

    rows = []
    for intra_path in tqdm(intra_files, desc="추론 중"):
        video_name = Path(intra_path).stem
        result = infer_single_video(video_name, model, device, mean, std)
        if result:
            rows.append(result)

    # ── submission.csv 저장 ──
    if not rows:
        print("결과 없음.")
        return

    submission = pd.DataFrame(rows, columns=["path", "accident_time", "center_x", "center_y", "type"])
    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"\nsubmission.csv 저장 완료: {OUTPUT_CSV}")
    print(f"총 {len(submission)}개 영상 처리됨")
    print("\n--- 미리보기 ---")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
