import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, cast


Metadata = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time correction postprocess for submission CSV")
    _ = parser.add_argument("--input_csv", type=str, required=True)
    _ = parser.add_argument("--metadata_csv", type=str, required=True)
    _ = parser.add_argument("--output_csv", type=str, required=True)
    _ = parser.add_argument("--edge_ratio", type=float, default=0.04)
    _ = parser.add_argument("--shrink_alpha", type=float, default=0.35)
    return parser.parse_args()


def load_metadata(metadata_csv: str) -> Dict[str, Metadata]:
    out: Dict[str, Metadata] = {}
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = str(row.get("path", "")).strip()
            if not path:
                continue
            duration = float(row.get("duration", 0.0) or 0.0)
            no_frames = float(row.get("no_frames", 0.0) or 0.0)
            if duration <= 0.0 or no_frames <= 0.0:
                continue
            fps = no_frames / duration
            out[path] = (duration, fps)
    return out


def clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def snap_to_frame_grid(value_sec: float, fps: float) -> float:
    if fps <= 0.0:
        return value_sec
    return round(value_sec * fps) / fps


def correct_time(raw_time: float, duration: float, fps: float, edge_ratio: float, shrink_alpha: float) -> float:
    t = clamp(raw_time, 0.0, duration)
    t = snap_to_frame_grid(t, fps)

    edge_margin = min(duration * 0.25, max(2.0 / fps if fps > 0 else 0.0, duration * edge_ratio))
    left_anchor = edge_margin
    right_anchor = max(left_anchor, duration - edge_margin)

    if t < left_anchor:
        t = (1.0 - shrink_alpha) * t + shrink_alpha * left_anchor
    elif t > right_anchor:
        t = (1.0 - shrink_alpha) * t + shrink_alpha * right_anchor

    t = snap_to_frame_grid(t, fps)
    t = clamp(t, 0.0, duration)
    return round(t, 2)


def main() -> None:
    args = parse_args()
    metadata_csv = cast(str, args.metadata_csv)
    input_csv = cast(str, args.input_csv)
    output_csv = cast(str, args.output_csv)
    edge_ratio = cast(float, args.edge_ratio)
    shrink_alpha = cast(float, args.shrink_alpha)

    metadata = load_metadata(metadata_csv)

    input_path = Path(input_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    changed = 0
    total = 0

    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("input CSV has no header")

        rows: List[Dict[str, str]] = []
        for row in reader:
            total += 1
            path = str(row.get("path", "")).strip()
            meta = metadata.get(path)
            if meta is None:
                rows.append(dict(row))
                continue

            duration, fps = meta
            before = float(row.get("accident_time", 0.0) or 0.0)
            after = correct_time(
                raw_time=before,
                duration=duration,
                fps=fps,
                edge_ratio=edge_ratio,
                shrink_alpha=shrink_alpha,
            )
            if abs(after - before) > 1e-9:
                changed += 1
            row["accident_time"] = f"{after:.2f}".rstrip("0").rstrip(".")
            rows.append(dict(row))

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"written: {output_path}")
    print(f"rows: {total}")
    print(f"time_changed: {changed}")


if __name__ == "__main__":
    main()
