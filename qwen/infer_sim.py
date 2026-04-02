#!/usr/bin/env python3

import argparse
import importlib.util
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm


def _load_core() -> Any:
    base_dir = Path(__file__).resolve().parent
    module_path = base_dir / "infer.py"
    spec = importlib.util.spec_from_file_location("_infer_core", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load infer core: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = _load_core()


def _build_metadata_lookup(metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for key, value in metadata.items():
        key_str = str(key).replace("\\", "/").strip()
        if not key_str:
            continue
        lookup[key_str] = value
        name = Path(key_str).name
        lookup[f"videos/{name}"] = value
        stripped = re.sub(r"^\d{4}_", "", name)
        lookup[f"videos/{stripped}"] = value
    return lookup


def _resolve_meta_for_video(video_path: str, video_root: str, lookup: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    root = Path(video_root).resolve()
    p = Path(video_path).resolve()
    try:
        rel = p.relative_to(root).as_posix()
    except ValueError:
        rel = p.name
    stripped_name = re.sub(r"^\d{4}_", "", Path(video_path).name)
    keys = [
        f"videos/{rel}",
        rel,
        f"videos/{Path(video_path).name}",
        f"videos/{stripped_name}",
        Path(video_path).name,
    ]
    for k in keys:
        if k in lookup:
            return lookup[k]
    return None


def run(video_dir: str, metadata_csv: str, model_id: str, limit: int) -> None:
    metadata: Dict[str, Dict[str, Any]] = {}
    if Path(metadata_csv).exists():
        metadata = core.load_metadata(metadata_csv)
        core.logger.info("Loaded metadata entries: %d", len(metadata))
    else:
        core.logger.warning("Metadata CSV not found: %s (fallback defaults)", metadata_csv)

    metadata_lookup = _build_metadata_lookup(metadata)
    prompt_loader = core.PromptLoader(core.SYSTEM_PROMPT_FILE, core.USER_PROMPT_FILE)
    model, processor = core.load_model(model_id)

    video_paths = sorted(str(p) for p in Path(video_dir).rglob("*.mp4"))
    if limit > 0:
        video_paths = video_paths[:limit]
    if not video_paths:
        raise FileNotFoundError(f"No .mp4 files found recursively in: {video_dir}")

    core.VIDEO_DIR = video_dir

    records: List[Dict[str, Any]] = []
    pbar = tqdm(total=len(video_paths), desc="Inference", unit="video")
    try:
        for start in range(0, len(video_paths), core.INFERENCE_BATCH_SIZE):
            batch_paths = video_paths[start : start + core.INFERENCE_BATCH_SIZE]
            batch_metas = [_resolve_meta_for_video(p, video_dir, metadata_lookup) for p in batch_paths]
            batch_results = core.infer_batch(model, processor, batch_paths, batch_metas, prompt_loader)
            records.extend(batch_results)
            pbar.update(len(batch_paths))
            for result in batch_results:
                core.logger.info(
                    "-> %s | %.2fs (%.3f, %.3f) | %s | conf=%.2f | issues=%s",
                    Path(result["path"]).name,
                    result["time"],
                    result["center_x"],
                    result["center_y"],
                    result["type"],
                    result["confidence"],
                    result["issues"],
                )
    finally:
        pbar.close()

    core.save_outputs(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursive traffic accident inference for nested video folders")
    parser.add_argument("--video_dir", type=str, default=core.VIDEO_DIR)
    parser.add_argument("--metadata_csv", type=str, default=core.METADATA_CSV)
    parser.add_argument("--model_id", type=str, default=core.MODEL_ID)
    parser.add_argument("--config", type=str, default=str(core.config_path))
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        video_dir=args.video_dir,
        metadata_csv=args.metadata_csv,
        model_id=args.model_id,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
