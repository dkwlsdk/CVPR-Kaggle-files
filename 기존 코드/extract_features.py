import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from functools import partial
from tqdm import tqdm

def calculate_intra_track_features(df):
    """
    하나의 track_id 내에서 개별적으로 구해야 할 정보: 
    속도(Velocity), 가속도(Acceleration), 이동방향(Direction), 방향변화량(Jerk / Curvature)을 계산합니다.
    """
    df = df.copy()
    df['cx'] = (df['x1'] + df['x2']) / 2.0
    df['cy'] = (df['y1'] + df['y2']) / 2.0
    df['dx'] = df.groupby('track_id')['cx'].diff()
    df['dy'] = df.groupby('track_id')['cy'].diff()
    df['dframes'] = df.groupby('track_id')['frame_idx'].diff()
    df['velocity'] = np.sqrt(df['dx']**2 + df['dy']**2) / df['dframes']
    df['direction'] = np.arctan2(df['dy'], df['dx'])
    df['acceleration'] = df.groupby('track_id')['velocity'].diff() / df['dframes']
    diff_dir = df.groupby('track_id')['direction'].diff()
    df['direction_change'] = np.arctan2(np.sin(diff_dir), np.cos(diff_dir))
    
    # ── 추가: 궤적 방향성 (과거 20프레임 이동 벡터 평균) ──
    # 단순 순간 각도가 아닌, 전체적인 진행 방향을 파악하기 위함 (sideswipe vs t-bone 구분용)
    df['rolling_dx'] = df.groupby('track_id')['dx'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    df['rolling_dy'] = df.groupby('track_id')['dy'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    df['traj_direction'] = np.arctan2(df['rolling_dy'], df['rolling_dx'])

    feature_cols = ['velocity', 'acceleration', 'direction', 'direction_change', 'traj_direction']
    df[feature_cols] = df[feature_cols].fillna(0.0)
    df['curvature'] = df.groupby('track_id')['direction_change'].apply(lambda x: x.abs().rolling(window=3, min_periods=1).mean()).reset_index(level=0, drop=True)
    df['curvature'] = df['curvature'].fillna(0.0)
    return df

def process_single_csv(csv_path, output_dir):
    try:
        out_intra_path = os.path.join(output_dir, "intra", os.path.basename(csv_path))
        out_inter_path = os.path.join(output_dir, "inter", os.path.basename(csv_path))
        
        # 파일이 이미 존재하면 건너뛰기 (이어하기 지원)
        if os.path.exists(out_intra_path):
            return True

        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        
        df = calculate_intra_track_features(df)
        
        frames = sorted(df['frame_idx'].unique())
        inter_features = []
        
        for f_idx in frames:
            f_data = df[df['frame_idx'] == f_idx]
            tracks = f_data['track_id'].unique()
            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    trackA = tracks[i]
                    trackB = tracks[j]
                    rowA = f_data[f_data['track_id'] == trackA].iloc[0]
                    rowB = f_data[f_data['track_id'] == trackB].iloc[0]
                    dist = np.sqrt((rowA['cx'] - rowB['cx'])**2 + (rowA['cy'] - rowB['cy'])**2)
                    inter_features.append({
                        'video_name': rowA['video_name'],
                        'frame_idx': f_idx,
                        'track_A': trackA,
                        'track_B': trackB,
                        'distance': dist,
                        'dir_A': rowA['direction'],
                        'dir_B': rowB['direction'],
                        'traj_dir_A': rowA['traj_direction'],
                        'traj_dir_B': rowB['traj_direction'],
                        'dx_A': rowA['dx'], 'dy_A': rowA['dy'],
                        'dx_B': rowB['dx'], 'dy_B': rowB['dy']
                    })
        
        inter_df = pd.DataFrame(inter_features)
        if not inter_df.empty:
            inter_df['pair'] = inter_df.apply(lambda r: f"{int(min(r['track_A'], r['track_B']))}_{int(max(r['track_A'], r['track_B']))}", axis=1)
            inter_df = inter_df.sort_values(['pair', 'frame_idx'])
            inter_df['dframes'] = inter_df.groupby('pair')['frame_idx'].diff()
            
            # 접근 속도 (Approach Speed)
            inter_df['approach_speed'] = -(inter_df.groupby('pair')['distance'].diff()) / inter_df['dframes']
            inter_df['approach_speed'] = inter_df['approach_speed'].fillna(0.0)
            
            # 1. TTC (Time To Collision)
            inter_df['ttc'] = np.where(inter_df['approach_speed'] > 0, 
                                       inter_df['distance'] / inter_df['approach_speed'], 
                                       9999.0)
                                       
            # 2. Relative Angle (순간 각도 차이)
            diff_rad = np.abs(np.arctan2(np.sin(inter_df['dir_A'] - inter_df['dir_B']), 
                                         np.cos(inter_df['dir_A'] - inter_df['dir_B'])))
            inter_df['relative_angle'] = np.degrees(diff_rad)

            # 3. Trajectory Angle Diff (궤적 진행 방향 차이 - sideswipe vs t-bone 구분)
            traj_diff_rad = np.abs(np.arctan2(np.sin(inter_df['traj_dir_A'] - inter_df['traj_dir_B']), 
                                             np.cos(inter_df['traj_dir_A'] - inter_df['traj_dir_B'])))
            inter_df['trajectory_angle_diff'] = np.degrees(traj_diff_rad)
            
            # 4. Relative Velocity Vector
            inter_df['v_rel'] = np.sqrt((inter_df['dx_A'] - inter_df['dx_B'])**2 + 
                                        (inter_df['dy_A'] - inter_df['dy_B'])**2) / inter_df['dframes']
            inter_df['v_rel'] = inter_df['v_rel'].fillna(0.0)
        else:
            inter_df = pd.DataFrame(columns=['video_name', 'frame_idx', 'track_A', 'track_B', 'distance', 'approach_speed', 'ttc', 'relative_angle', 'trajectory_angle_diff', 'v_rel', 'pair'])

        os.makedirs(os.path.join(output_dir, "intra"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "inter"), exist_ok=True)
        out_intra_path = os.path.join(output_dir, "intra", os.path.basename(csv_path))
        out_inter_path = os.path.join(output_dir, "inter", os.path.basename(csv_path))
        df.to_csv(out_intra_path, index=False)
        if not inter_df.empty:
            inter_df.to_csv(out_inter_path, index=False)
        return True
    except Exception as e:
        print(f"Error {os.path.basename(csv_path)}: {e}")
        return False

def extract_features_for_video(csv_path, output_dir):
    return process_single_csv(csv_path, output_dir)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tracking_dir = os.path.join(base_dir, "TRACKING")
    feature_dir = os.path.join(base_dir, "FEATURES")
    
    csv_files = glob.glob(os.path.join(tracking_dir, "*.csv"))
    print(f"Feature 추출을 시작합니다: 총 {len(csv_files)}개의 비디오 시퀀스")
    
    # Multiprocessing Pool
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    func = partial(process_single_csv, output_dir=feature_dir)
    
    results = []
    with tqdm(total=len(csv_files)) as pbar:
        for i, _ in enumerate(pool.imap_unordered(func, csv_files)):
            pbar.update()

    pool.close()
    pool.join()
    print("Feature 추출 스크립트 실행이 완료되었습니다.")

if __name__ == "__main__":
    main()
