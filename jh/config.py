from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
VIDEOS_DIR = DATASET_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "outputs"
LOGS_DIR = OUTPUT_DIR / "logs"

METADATA_CSV = DATASET_DIR / "test_metadata.csv"
FINAL_CSV = OUTPUT_DIR / "predictions.csv"
STAGE1_CSV = OUTPUT_DIR / "stage1_candidate_times.csv"
STAGE2_CSV = OUTPUT_DIR / "stage2_fixed_times.csv"
STAGE3_CSV = OUTPUT_DIR / "stage3_types.csv"
STAGE4_CSV = OUTPUT_DIR / "stage4_locations.csv"

RAW_JSONL = LOGS_DIR / "qwen_raw_outputs.jsonl"
RUN_LOG = LOGS_DIR / "pipeline.log"

MODEL_NAME = "Qwen/Qwen3.5-9B"
VALID_TYPES = {"rear-end", "head-on", "sideswipe", "t-bone", "single"}

TOPK_PEAKS = 3
MIN_PEAK_SEP_SEC = 1.0
FLOW_SAMPLE_FPS = 5.0
TRACK_SAMPLE_FPS = 5.0

FIX_TIME_TARGET_FPS = 2.0
FIX_TIME_MIN_FRAMES = 8
FIX_TIME_MAX_FRAMES = 248

FIX_CLIP_SEC = 4.0

TYPE_TARGET_FPS = 6.0
TYPE_MIN_FRAMES = 16
TYPE_MAX_FRAMES = 32

MAX_NEW_TOKENS = 320
TEMPERATURE = 0.2
TOP_P = 0.9

YOLO_MODEL = "yolo26x.pt"
YOLO_CONF = 0.20
YOLO_IOU = 0.50
VEHICLE_CLASSES = {2, 3, 5, 7}


def ensure_dirs() -> None:
    for d in [DATASET_DIR, VIDEOS_DIR, OUTPUT_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def safe_video_name(rel_path: str) -> str:
    rel = Path(rel_path).with_suffix("")
    return "__".join(rel.parts)


def get_video_output_dir(rel_path: str) -> Path:
    d = OUTPUT_DIR / safe_video_name(rel_path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_stage_dir(rel_path: str, stage_name: str) -> Path:
    d = get_video_output_dir(rel_path) / stage_name
    d.mkdir(parents=True, exist_ok=True)
    return d
