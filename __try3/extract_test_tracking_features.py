from __future__ import annotations

import argparse
import json
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR / "CVPR-Kaggle-files"
DEFAULT_INPUT_DIR = PROJECT_DIR / "Detection_Dataset" / "test_tracking"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "Detection_Dataset" / "test_features"
DEFAULT_GEOMETRY_DIR = PROJECT_DIR / "Detection_Dataset" / "geometry"
DEFAULT_SMOOTHING_WINDOW = 5

INTRA_COLUMNS = [
    "video_name",
    "frame_idx",
    "original_track_id",
    "track_id",
    "track_id_source",
    "class_id",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "dx",
    "dy",
    "dframes",
    "velocity",
    "direction",
    "acceleration",
    "jerk",
    "direction_change",
    "rolling_dx",
    "rolling_dy",
    "traj_direction",
    "curvature",
]

INTER_COLUMNS = [
    "video_name",
    "frame_idx",
    "track_A",
    "track_B",
    "distance",
    "dir_A",
    "dir_B",
    "traj_dir_A",
    "traj_dir_B",
    "dx_A",
    "dy_A",
    "dx_B",
    "dy_B",
    "pair",
    "dframes",
    "approach_speed",
    "ttc",
    "relative_angle",
    "trajectory_angle_diff",
    "v_rel",
]


@dataclass
class TrackState:
    track_id: int
    canonical_track_id: int
    last_frame_idx: int
    last_bbox: Tuple[float, float, float, float]
    last_cx: float
    last_cy: float
    last_class_id: Optional[int]
    source_type: str


def load_tracking_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames")
    if not isinstance(frames, list):
        return {"video_name": json_path.stem, "frames": []}
    if not data.get("video_name"):
        data["video_name"] = json_path.stem
    return data


def compute_bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    return (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0


def compute_planar_homography(
    image_points: Sequence[Sequence[float]],
    ground_points: Sequence[Sequence[float]],
) -> np.ndarray:
    if len(image_points) != len(ground_points) or len(image_points) < 4:
        raise ValueError("At least four image_points/ground_points pairs are required.")

    rows: List[List[float]] = []
    for src, dst in zip(image_points, ground_points):
        x, y = float(src[0]), float(src[1])
        u, v = float(dst[0]), float(dst[1])
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])

    matrix = np.asarray(rows, dtype=np.float64)
    _, _, vh = np.linalg.svd(matrix)
    homography = vh[-1].reshape(3, 3)
    if np.isclose(homography[2, 2], 0.0):
        raise ValueError("Homography normalization failed because H[2,2] is zero.")
    homography /= homography[2, 2]
    return homography


def load_geometry_transform(video_name: str, geometry_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    if geometry_dir is None:
        return None

    config_path = geometry_dir / f"{video_name}.json"
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    homography_data = payload.get("homography")
    if homography_data is not None:
        homography = np.asarray(homography_data, dtype=np.float64)
        if homography.shape != (3, 3):
            raise ValueError(f"Invalid homography shape for {video_name}: {homography.shape}")
    else:
        image_points = payload.get("image_points")
        ground_points = payload.get("ground_points")
        if not isinstance(image_points, list) or not isinstance(ground_points, list):
            raise ValueError(
                f"Geometry config for {video_name} must contain either homography or image_points/ground_points."
            )
        homography = compute_planar_homography(image_points, ground_points)

    if not np.all(np.isfinite(homography)):
        raise ValueError(f"Homography for {video_name} contains non-finite values.")
    if np.isclose(homography[2, 2], 0.0):
        raise ValueError(f"Homography for {video_name} has zero normalization term.")

    homography = homography / homography[2, 2]
    return {
        "matrix": homography,
        "source": payload.get("source", config_path.name),
        "path": str(config_path),
    }


def project_points_with_homography(
    points: np.ndarray,
    homography: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    if homography is None:
        valid = np.ones(len(points), dtype=bool)
        return points.astype(np.float64, copy=True), valid

    ones = np.ones((len(points), 1), dtype=np.float64)
    homogeneous = np.concatenate([points.astype(np.float64), ones], axis=1)
    projected = homogeneous @ homography.T
    denom = projected[:, 2]
    valid = np.isfinite(denom) & (np.abs(denom) > 1e-9)

    output = np.full((len(points), 2), np.nan, dtype=np.float64)
    if np.any(valid):
        output[valid, 0] = projected[valid, 0] / denom[valid]
        output[valid, 1] = projected[valid, 1] / denom[valid]
    return output, valid


def smooth_grouped_series(series: pd.Series, window: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    if window <= 1 or len(numeric) <= 1:
        return numeric
    return numeric.rolling(window=window, min_periods=1, center=True).mean()


def bbox_diag(bbox: Sequence[float]) -> float:
    width = float(bbox[2]) - float(bbox[0])
    height = float(bbox[3]) - float(bbox[1])
    return float(np.hypot(width, height))


def iou_xyxy(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def find_track_id(track_id: int, parent_map: Dict[int, int]) -> int:
    parent = parent_map.setdefault(track_id, track_id)
    if parent != track_id:
        parent_map[track_id] = find_track_id(parent, parent_map)
    return parent_map[track_id]


def merge_track_ids(child_id: int, parent_id: int, parent_map: Dict[int, int]) -> int:
    child_root = find_track_id(child_id, parent_map)
    parent_root = find_track_id(parent_id, parent_map)
    if child_root != parent_root:
        parent_map[child_root] = parent_root
    return parent_root


def rank_track_candidates(
    det: dict,
    active_tracks: Dict[int, TrackState],
    parent_map: Dict[int, int],
    frame_idx: int,
    allowed_sources: Optional[Iterable[str]] = None,
    excluded_track_ids: Optional[Iterable[int]] = None,
) -> List[dict]:
    allowed = set(allowed_sources) if allowed_sources is not None else None
    excluded = {find_track_id(track_id, parent_map) for track_id in (excluded_track_ids or [])}
    det_bbox = det["bbox"]
    det_diag = bbox_diag(det_bbox)
    det_class_id = det.get("class_id")
    candidates: List[dict] = []

    for state in active_tracks.values():
        track_id = find_track_id(state.track_id, parent_map)
        if track_id in excluded:
            continue
        if allowed is not None and state.source_type not in allowed:
            continue

        frame_gap = frame_idx - state.last_frame_idx
        if frame_gap <= 0 or frame_gap > 2:
            continue

        overlap = iou_xyxy(state.last_bbox, det_bbox)
        center_distance = float(np.hypot(det["cx"] - state.last_cx, det["cy"] - state.last_cy))
        distance_limit = max(35.0, 0.6 * max(bbox_diag(state.last_bbox), det_diag))
        if overlap < 0.30 and center_distance > distance_limit:
            continue

        class_same = int(state.last_class_id == det_class_id) if state.last_class_id is not None else 0
        candidates.append(
            {
                "track_id": track_id,
                "iou": overlap,
                "center_distance": center_distance,
                "class_same": class_same,
            }
        )

    candidates.sort(key=lambda c: (-c["iou"], c["center_distance"], -c["class_same"]))
    return candidates


def ensure_dataframe_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = pd.Series(dtype="float64")
    return df.loc[:, columns]


def resolve_track_ids_from_json(data: dict) -> pd.DataFrame:
    video_name = data.get("video_name", "unknown")
    frames = data.get("frames", [])
    active_tracks: Dict[int, TrackState] = {}
    parent_map: Dict[int, int] = {}
    detection_rows: List[dict] = []
    next_synthetic_id = -1

    for frame in frames:
        frame_idx = int(frame.get("frame_idx", 0))
        detections = frame.get("detections", [])
        valid_detections = []

        for det in detections:
            bbox = det.get("bbox_xyxy", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [float(v) for v in bbox]
            cx, cy = compute_bbox_center((x1, y1, x2, y2))
            valid_detections.append(
                {
                    "original_track_id": det.get("track_id"),
                    "class_id": det.get("class_id"),
                    "class_name": det.get("class_name"),
                    "bbox": (x1, y1, x2, y2),
                    "cx": cx,
                    "cy": cy,
                }
            )

        real_detections = [det for det in valid_detections if det["original_track_id"] is not None]
        null_detections = [det for det in valid_detections if det["original_track_id"] is None]
        used_track_ids = set()

        for det in real_detections:
            real_track_id = int(det["original_track_id"])
            parent_map.setdefault(real_track_id, real_track_id)
            canonical_track_id = find_track_id(real_track_id, parent_map)

            synthetic_candidates = rank_track_candidates(
                det,
                active_tracks,
                parent_map,
                frame_idx,
                allowed_sources={"synthetic"},
                excluded_track_ids=used_track_ids,
            )
            if synthetic_candidates:
                synthetic_track_id = synthetic_candidates[0]["track_id"]
                canonical_track_id = merge_track_ids(synthetic_track_id, canonical_track_id, parent_map)
                active_tracks.pop(synthetic_track_id, None)

            active_tracks[canonical_track_id] = TrackState(
                track_id=canonical_track_id,
                canonical_track_id=canonical_track_id,
                last_frame_idx=frame_idx,
                last_bbox=det["bbox"],
                last_cx=det["cx"],
                last_cy=det["cy"],
                last_class_id=det.get("class_id"),
                source_type="tracker",
            )
            used_track_ids.add(canonical_track_id)

            detection_rows.append(
                {
                    "video_name": video_name,
                    "frame_idx": frame_idx,
                    "original_track_id": real_track_id,
                    "provisional_track_id": canonical_track_id,
                    "class_id": det.get("class_id"),
                    "class_name": det.get("class_name"),
                    "x1": det["bbox"][0],
                    "y1": det["bbox"][1],
                    "x2": det["bbox"][2],
                    "y2": det["bbox"][3],
                    "cx": det["cx"],
                    "cy": det["cy"],
                }
            )

        for det in null_detections:
            real_candidates = rank_track_candidates(
                det,
                active_tracks,
                parent_map,
                frame_idx,
                allowed_sources={"tracker"},
                excluded_track_ids=used_track_ids,
            )
            synthetic_candidates = rank_track_candidates(
                det,
                active_tracks,
                parent_map,
                frame_idx,
                allowed_sources={"synthetic"},
                excluded_track_ids=used_track_ids,
            )

            matched_track_id: Optional[int] = None
            source_type = "synthetic"
            if real_candidates:
                matched_track_id = real_candidates[0]["track_id"]
                source_type = "tracker"
            elif synthetic_candidates:
                matched_track_id = synthetic_candidates[0]["track_id"]

            if matched_track_id is None:
                matched_track_id = next_synthetic_id
                parent_map[matched_track_id] = matched_track_id
                next_synthetic_id -= 1

            canonical_track_id = find_track_id(matched_track_id, parent_map)
            active_tracks[canonical_track_id] = TrackState(
                track_id=canonical_track_id,
                canonical_track_id=canonical_track_id,
                last_frame_idx=frame_idx,
                last_bbox=det["bbox"],
                last_cx=det["cx"],
                last_cy=det["cy"],
                last_class_id=det.get("class_id"),
                source_type=source_type,
            )
            used_track_ids.add(canonical_track_id)

            detection_rows.append(
                {
                    "video_name": video_name,
                    "frame_idx": frame_idx,
                    "original_track_id": None,
                    "provisional_track_id": canonical_track_id,
                    "class_id": det.get("class_id"),
                    "class_name": det.get("class_name"),
                    "x1": det["bbox"][0],
                    "y1": det["bbox"][1],
                    "x2": det["bbox"][2],
                    "y2": det["bbox"][3],
                    "cx": det["cx"],
                    "cy": det["cy"],
                }
            )

    if not detection_rows:
        return pd.DataFrame(columns=INTRA_COLUMNS)

    df = pd.DataFrame(detection_rows)
    df["track_id"] = df["provisional_track_id"].apply(lambda tid: find_track_id(int(tid), parent_map))
    df["track_id_source"] = np.where(
        df["original_track_id"].notna(),
        "tracker",
        np.where(df["track_id"] > 0, "synthetic_merged", "synthetic"),
    )
    df = df.drop(columns=["provisional_track_id"])
    df = df.sort_values(["track_id", "frame_idx", "x1", "y1"]).reset_index(drop=True)
    return df


def calculate_intra_track_features(
    df: pd.DataFrame,
    geometry_transform: Optional[Dict[str, Any]] = None,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=INTRA_COLUMNS)

    df = df.copy().sort_values(["track_id", "frame_idx", "x1", "y1"]).reset_index(drop=True)
    df["cx"] = (df["x1"] + df["x2"]) / 2.0
    df["cy"] = (df["y1"] + df["y2"]) / 2.0

    raw_points = df[["cx", "cy"]].to_numpy(dtype=np.float64, copy=True)
    homography = None if geometry_transform is None else geometry_transform["matrix"]
    projected_points, projected_valid = project_points_with_homography(raw_points, homography)
    df["motion_x_raw"] = projected_points[:, 0]
    df["motion_y_raw"] = projected_points[:, 1]
    if geometry_transform is not None:
        df["motion_x_raw"] = df["motion_x_raw"].where(projected_valid, df["cx"])
        df["motion_y_raw"] = df["motion_y_raw"].where(projected_valid, df["cy"])

    df["motion_x"] = df.groupby("track_id")["motion_x_raw"].transform(
        lambda x: smooth_grouped_series(x, smoothing_window)
    )
    df["motion_y"] = df.groupby("track_id")["motion_y_raw"].transform(
        lambda x: smooth_grouped_series(x, smoothing_window)
    )

    raw_dx = df.groupby("track_id")["motion_x"].diff()
    raw_dy = df.groupby("track_id")["motion_y"].diff()
    raw_dframes = df.groupby("track_id")["frame_idx"].diff()
    safe_dframes = raw_dframes.where(raw_dframes > 0)

    velocity = np.sqrt(raw_dx**2 + raw_dy**2) / safe_dframes
    direction = np.arctan2(raw_dy, raw_dx)
    acceleration = df.assign(_velocity=velocity).groupby("track_id")["_velocity"].diff() / safe_dframes
    jerk = df.assign(_acceleration=acceleration).groupby("track_id")["_acceleration"].diff() / safe_dframes
    diff_dir = df.assign(_direction=direction).groupby("track_id")["_direction"].diff()
    direction_change = np.arctan2(np.sin(diff_dir), np.cos(diff_dir))

    df["dx"] = raw_dx.fillna(0.0)
    df["dy"] = raw_dy.fillna(0.0)
    df["dframes"] = raw_dframes.fillna(0.0)
    df["velocity"] = velocity.fillna(0.0)
    df["direction"] = direction.fillna(0.0)
    df["acceleration"] = acceleration.fillna(0.0)
    df["jerk"] = jerk.fillna(0.0)
    df["direction_change"] = direction_change.fillna(0.0)

    df["rolling_dx"] = df.groupby("track_id")["dx"].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )
    df["rolling_dy"] = df.groupby("track_id")["dy"].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )
    df["traj_direction"] = np.arctan2(df["rolling_dy"], df["rolling_dx"])
    df["traj_direction"] = df["traj_direction"].fillna(0.0)

    df["curvature"] = (
        df.groupby("track_id")["direction_change"]
        .apply(lambda x: x.abs().rolling(window=3, min_periods=1).mean())
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    fill_zero_cols = [
        "dx",
        "dy",
        "dframes",
        "velocity",
        "direction",
        "acceleration",
        "jerk",
        "direction_change",
        "rolling_dx",
        "rolling_dy",
        "traj_direction",
        "curvature",
    ]
    df[fill_zero_cols] = df[fill_zero_cols].fillna(0.0)
    return df


def calculate_inter_track_features(intra_df: pd.DataFrame) -> pd.DataFrame:
    if intra_df.empty:
        return pd.DataFrame(columns=INTER_COLUMNS)

    inter_features: List[dict] = []
    frames = sorted(intra_df["frame_idx"].unique())

    for frame_idx in frames:
        frame_data = intra_df[intra_df["frame_idx"] == frame_idx]
        frame_data = frame_data.sort_values(["track_id", "x1", "y1"]).drop_duplicates("track_id", keep="last")
        tracks = frame_data["track_id"].tolist()
        if len(tracks) < 2:
            continue

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                row_a = frame_data.iloc[i]
                row_b = frame_data.iloc[j]
                distance = float(np.hypot(row_a["motion_x"] - row_b["motion_x"], row_a["motion_y"] - row_b["motion_y"]))
                inter_features.append(
                    {
                        "video_name": row_a["video_name"],
                        "frame_idx": frame_idx,
                        "track_A": int(row_a["track_id"]),
                        "track_B": int(row_b["track_id"]),
                        "distance": distance,
                        "dir_A": row_a["direction"],
                        "dir_B": row_b["direction"],
                        "traj_dir_A": row_a["traj_direction"],
                        "traj_dir_B": row_b["traj_direction"],
                        "dx_A": row_a["dx"],
                        "dy_A": row_a["dy"],
                        "dx_B": row_b["dx"],
                        "dy_B": row_b["dy"],
                    }
                )

    if not inter_features:
        return pd.DataFrame(columns=INTER_COLUMNS)

    inter_df = pd.DataFrame(inter_features)
    inter_df["pair"] = inter_df.apply(
        lambda row: f"{int(min(row['track_A'], row['track_B']))}_{int(max(row['track_A'], row['track_B']))}",
        axis=1,
    )
    inter_df = inter_df.sort_values(["pair", "frame_idx"]).reset_index(drop=True)

    raw_dframes = inter_df.groupby("pair")["frame_idx"].diff()
    safe_dframes = raw_dframes.where(raw_dframes > 0)
    approach_speed = -(inter_df.groupby("pair")["distance"].diff()) / safe_dframes
    relative_angle_rad = np.abs(
        np.arctan2(np.sin(inter_df["dir_A"] - inter_df["dir_B"]), np.cos(inter_df["dir_A"] - inter_df["dir_B"]))
    )
    trajectory_angle_rad = np.abs(
        np.arctan2(
            np.sin(inter_df["traj_dir_A"] - inter_df["traj_dir_B"]),
            np.cos(inter_df["traj_dir_A"] - inter_df["traj_dir_B"]),
        )
    )
    v_rel = np.sqrt((inter_df["dx_A"] - inter_df["dx_B"]) ** 2 + (inter_df["dy_A"] - inter_df["dy_B"]) ** 2) / safe_dframes

    inter_df["dframes"] = raw_dframes.fillna(0.0)
    inter_df["approach_speed"] = approach_speed.fillna(0.0)
    inter_df["ttc"] = np.where(inter_df["approach_speed"] > 0, inter_df["distance"] / inter_df["approach_speed"], 9999.0)
    inter_df["relative_angle"] = np.degrees(relative_angle_rad)
    inter_df["trajectory_angle_diff"] = np.degrees(trajectory_angle_rad)
    inter_df["v_rel"] = v_rel.fillna(0.0)

    return ensure_dataframe_columns(inter_df, INTER_COLUMNS)


def write_empty_csvs(video_name: str, output_dir: Path) -> None:
    intra_path = output_dir / "intra" / f"{video_name}.csv"
    inter_path = output_dir / "inter" / f"{video_name}.csv"
    intra_path.parent.mkdir(parents=True, exist_ok=True)
    inter_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=INTRA_COLUMNS).to_csv(intra_path, index=False)
    pd.DataFrame(columns=INTER_COLUMNS).to_csv(inter_path, index=False)


def process_single_json(
    json_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    geometry_dir: Optional[Path] = None,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> bool:
    video_name = json_path.stem
    intra_path = output_dir / "intra" / f"{video_name}.csv"
    inter_path = output_dir / "inter" / f"{video_name}.csv"

    if not overwrite and intra_path.exists() and inter_path.exists():
        return True

    output_dir.joinpath("intra").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("inter").mkdir(parents=True, exist_ok=True)

    try:
        data = load_tracking_json(json_path)
        resolved_df = resolve_track_ids_from_json(data)
        if resolved_df.empty:
            write_empty_csvs(video_name, output_dir)
            return True

        geometry_transform = load_geometry_transform(video_name, geometry_dir)
        intra_df = calculate_intra_track_features(
            resolved_df,
            geometry_transform=geometry_transform,
            smoothing_window=smoothing_window,
        )
        inter_df = calculate_inter_track_features(intra_df)

        ensure_dataframe_columns(intra_df, INTRA_COLUMNS).to_csv(intra_path, index=False)
        ensure_dataframe_columns(inter_df, INTER_COLUMNS).to_csv(inter_path, index=False)
        return True
    except Exception as exc:
        print(f"Error processing {json_path.name}: {exc}")
        write_empty_csvs(video_name, output_dir)
        return False


def _process_worker(args: Tuple[Path, Path, bool, Optional[Path], int]) -> bool:
    json_path, output_dir, overwrite, geometry_dir, smoothing_window = args
    return process_single_json(
        json_path,
        output_dir,
        overwrite=overwrite,
        geometry_dir=geometry_dir,
        smoothing_window=smoothing_window,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract intra/inter features from test_tracking JSON files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing tracking JSON files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to store intra/inter CSV files.")
    parser.add_argument(
        "--geometry-dir",
        type=Path,
        default=DEFAULT_GEOMETRY_DIR,
        help="Directory containing optional per-video geometry JSON files for homography projection.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=DEFAULT_SMOOTHING_WINDOW,
        help="Centered rolling window size used to smooth projected bbox centers before differentiating.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    geometry_dir = args.geometry_dir if args.geometry_dir and args.geometry_dir.exists() else None
    json_files = sorted(input_dir.glob("*.json"))

    print(f"Feature extraction start: {len(json_files)} JSON files")
    if not json_files:
        return

    num_workers = min(multiprocessing.cpu_count(), 16)
    print(f"Using {num_workers} parallel workers")
    if geometry_dir is None:
        print("Geometry dir: not found or disabled; using smoothed bbox-center pixel motion.")
    else:
        print(f"Geometry dir: {geometry_dir}")
    print(f"Smoothing window: {max(1, int(args.smoothing_window))}")

    tasks = [
        (json_path, output_dir, args.overwrite, geometry_dir, max(1, int(args.smoothing_window)))
        for json_path in json_files
    ]
    success = 0

    with multiprocessing.Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(_process_worker, tasks), total=len(tasks), desc="JSON feature extraction"):
            if result:
                success += 1

    print(f"Finished: {success}/{len(json_files)} files processed")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
