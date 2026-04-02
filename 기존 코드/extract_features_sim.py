"""
extract_features_sim.py
SIM_TRACKING 폴더의 CSV들에 대해 피처를 추출한다.
기존 extract_features.py의 로직을 그대로 사용하되
입력/출력 경로만 SIM 전용으로 변경.
"""
import os
import sys
import glob
import multiprocessing
from functools import partial
from tqdm import tqdm

# 기존 extract_features 모듈 재사용
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import process_single_csv

SIM_TRACKING_DIR = r"C:\Users\echin\Desktop\CVPR Kaggle\SIM_TRACKING"
SIM_FEATURES_DIR = r"C:\Users\echin\Desktop\CVPR Kaggle\SIM_FEATURES"


def main():
    csv_files = glob.glob(os.path.join(SIM_TRACKING_DIR, "**", "*.csv"), recursive=True)
    print(f"피처 추출 대상: {len(csv_files)}개 CSV")

    os.makedirs(os.path.join(SIM_FEATURES_DIR, "intra"), exist_ok=True)
    os.makedirs(os.path.join(SIM_FEATURES_DIR, "inter"), exist_ok=True)

    func = partial(process_single_csv, output_dir=SIM_FEATURES_DIR)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    with tqdm(total=len(csv_files), desc="피처 추출") as pbar:
        for _ in pool.imap_unordered(func, csv_files):
            pbar.update()

    pool.close()
    pool.join()
    print(f"완료. 출력: {SIM_FEATURES_DIR}")


if __name__ == "__main__":
    main()
