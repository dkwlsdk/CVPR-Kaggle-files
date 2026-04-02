from pathlib import Path

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

# 사용자의 데이터셋 경로로 설정
VIDEO_DIR = "/root/Desktop/workspace/ja/CVPR-Kaggle-files/데이터셋/test_videos"
METADATA_CSV = "/root/Desktop/workspace/ja/CVPR-Kaggle-files/데이터셋/test_metadata.csv"

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
SYSTEM_PROMPT_FILE = PROMPTS_DIR / "system_prompt.txt"
USER_PROMPT_FILE = PROMPTS_DIR / "user_prompt.txt"
JSON_ONLY_OUTPUT = True

TARGET_FPS = 1.0
MIN_FRAMES = 4
MAX_FRAMES = 32

MIN_PIXELS = 28 * 28
MAX_PIXELS = 224 * 224

MAX_NEW_TOKENS = 256
DO_SAMPLE = False
TEMPERATURE = None
TOP_P = None

DEFAULT_DURATION = 10.0
INFERENCE_BATCH_SIZE = 1

VALID_TYPES = ["head-on", "rear-end", "sideswipe", "t-bone", "single"]

RUN_TAG = (
    f"{Path(USER_PROMPT_FILE).stem}"
    f"_fps{str(TARGET_FPS).replace('.', 'p')}"
    f"_mf{MAX_FRAMES}"
)
OUTPUT_DIR = Path("./outputs") / RUN_TAG
SUBMISSION_CSV = OUTPUT_DIR / "submission.csv"
DEBUG_JSON = OUTPUT_DIR / "debug_results.json"
DEBUG_CSV = OUTPUT_DIR / "debug_results.csv"
