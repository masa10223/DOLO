#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ===== Runtime parameters (override with env vars) =====
RUN_NAME="${RUN_NAME:-foundation_v1}"
MODE="${MODE:-all}"                    # manifest | yolo | train | all
LOAD_MODEL_PATH="${LOAD_MODEL_PATH:-./yolo26x-pose.pt}"
ANNOTATIONS_ROOT="${ANNOTATIONS_ROOT:-../annotations}"
FOUNDATION_ROOT="${FOUNDATION_ROOT:-../annotations/foundation}"
TIFFS_ROOT="${TIFFS_ROOT:-/inthdd/tsutsumi/drosophila/DOLO/annotations/overfittings/tiffs}"
TARGET_SIZE="${TARGET_SIZE:-360}"              # augmentation target size per video for train
VAL_RATIO="${VAL_RATIO:-0.2}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-40}"
WORKERS="${WORKERS:-16}"
CACHE_MODE="${CACHE_MODE:-disk}"        # disk | ram | false
SAVE_PERIOD="${SAVE_PERIOD:--1}"        # -1 disables epoch checkpoints; 1 saves every epoch
FRACTION="${FRACTION:-1.0}"             # Ultralytics dataset fraction [0,1]
DATA_YAML="${DATA_YAML:-}"             # empty => use ${FOUNDATION_ROOT}/yamls/${RUN_NAME}.yaml
RESUME="${RESUME:-0}"                  # 1 = resume from weights/<run_name>/weights/last.pt (Ultralytics)
RESUME_ALLOW_STRIPPED="${RESUME_ALLOW_STRIPPED:-0}"  # 1 = emulate resume when last.pt is stripped
RESUME_STRIPPED_RUN_SUFFIX="${RESUME_STRIPPED_RUN_SUFFIX:-_resume_from_stripped}"
DATASETS_DIR="${DATASETS_DIR:-/cellpose/scripts}"
GPU_IDS="${GPU_IDS:-1 2 3}"              # space-separated, e.g. "0 1"

LOG_DIR="${FOUNDATION_ROOT}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_${RUN_NAME}_${TIMESTAMP}.out"

read -r -a GPU_ARRAY <<< "$GPU_IDS"

CMD=(
  python3 foundation_pipeline.py
  --mode "$MODE"
  --run_name "$RUN_NAME"
  --load_model_path "$LOAD_MODEL_PATH"
  --annotations_root "$ANNOTATIONS_ROOT"
  --foundation_root "$FOUNDATION_ROOT"
  --tiffs_root "$TIFFS_ROOT"
  --target_size "$TARGET_SIZE"
  --val_ratio "$VAL_RATIO"
  --seed "$SEED"
  --epochs "$EPOCHS"
  --batch "$BATCH"
  --workers "$WORKERS"
  --cache_mode "$CACHE_MODE"
  --save_period "$SAVE_PERIOD"
  --fraction "$FRACTION"
  --datasets_dir "$DATASETS_DIR"
)

if [ -n "$DATA_YAML" ]; then
  CMD+=(--data_yaml "$DATA_YAML")
fi
if [ "${RESUME}" = "1" ]; then
  CMD+=(--resume)
fi
if [ "${RESUME_ALLOW_STRIPPED}" = "1" ]; then
  CMD+=(--resume_allow_stripped)
fi
if [ "${NO_MICRO_BATCH:-0}" = "1" ]; then
  CMD+=(--no_micro_batch)
fi
CMD+=(--resume_stripped_run_suffix "$RESUME_STRIPPED_RUN_SUFFIX")

CMD+=(--gpu "${GPU_ARRAY[@]}")

echo "Running: ${CMD[*]}"
echo "Log file: $LOG_FILE"
nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
echo "Started. PID=$!"

# ===== Smoke run procedure =====
# 1) Manifest only (I/O + split check)
#    MODE=manifest RUN_NAME=foundation_v1_smoke ./foundation_train.sh
#
# 2) Training smoke (requires existing YAML)
#    MODE=train EPOCHS=5 RUN_NAME=foundation_v1_smoke DATA_YAML=../annotations/foundation/yamls/foundation_v1.yaml ./foundation_train.sh
#
# 3) YOLO data + YAML only (manifests must exist)
#    MODE=yolo RUN_NAME=foundation_v1 ./foundation_train.sh
#
# 4) Full (manifest → YOLO data → train)
#    MODE=all EPOCHS=5 RUN_NAME=foundation_v1_smoke ./foundation_train.sh
#
# 5) Resume training after NCCL timeout / crash (same RUN_NAME, same epochs)
#    MODE=train RESUME=1 RUN_NAME=foundation_v1 EPOCHS=50 ./foundation_train.sh
#    Checkpoint: annotations/foundation/weights/<RUN_NAME>/weights/last.pt
