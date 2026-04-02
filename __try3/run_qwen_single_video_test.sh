#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/opt/conda/envs/qwen35/bin/python"
SCRIPT_PATH="/root/Desktop/workspace/ja/qwen_single_video_test.py"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing interpreter: $PYTHON_BIN" >&2
  exit 1
fi

exec "$PYTHON_BIN" -u "$SCRIPT_PATH"
