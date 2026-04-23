#!/bin/bash
###############################################################################
# Evaluate a fine-tuned checkpoint on the test split
#
# Usage:
#   bash scripts/omi/run_eval.sh [checkpoint_step] [version]
#   bash scripts/omi/run_eval.sh 5000          # default = v1
#   bash scripts/omi/run_eval.sh 2000 v2       # evaluate v2
#
# fairseq2 creates a sweep sub-directory (ws_1.XXXXXXXX/) under the output
# dir, so we auto-detect it instead of hard-coding the path.
###############################################################################
STEP="${1:-5000}"
VERSION="${2:-v1}"

PROJECT_DIR="/home/data/projets-aps/projet6/MedBridge-AI2"
ENV_DIR="/home/data/projets-aps/projet6/env_aps_sanogo"
RUN_DIR="${PROJECT_DIR}/outputs/checkpoints/med_asr_ewe_${VERSION}"
DATA_DIR="/home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet_omni_v6/version=0/corpus=general/split=test/language=ewe_Latn"

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"
set -uo pipefail

# fairseq2 asset cards (common.assets.extra_paths also works, but belt-and-suspenders)
export FAIRSEQ2_USER_ASSET_CARD_DIRS="${PROJECT_DIR}/custom_cards"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# ── Auto-detect fairseq2 sweep sub-directory (ws_1.XXXXXXXX) ──────────────
# fairseq2 creates ws_1.<hash>/ subdirs. There may be several (failed runs).
# We search ALL ws_* dirs for the requested step and pick the one that exists.
MODEL_PATH="${RUN_DIR}/checkpoints/step_${STEP}/model"
if [ ! -d "${MODEL_PATH}" ]; then
    for SWEEP_DIR in "${RUN_DIR}"/ws_*; do
        CANDIDATE="${SWEEP_DIR}/checkpoints/step_${STEP}/model"
        if [ -d "${CANDIDATE}" ]; then
            MODEL_PATH="${CANDIDATE}"
            break
        fi
    done
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Checkpoint not found at: ${MODEL_PATH}"
    echo ""
    echo "Available checkpoints:"
    find "${RUN_DIR}" -type d -name "step_*" 2>/dev/null | sort
    exit 1
fi

# ── Find test parquet ──────────────────────────────────────────────────────
TEST_PARQUET="${DATA_DIR}/part-0000.parquet"
if [ ! -f "${TEST_PARQUET}" ]; then
    # Fallback: pick the first .parquet file in the test directory
    TEST_PARQUET=$(find "${DATA_DIR}" -name "*.parquet" -type f | head -1)
fi

EVAL_DIR="${RUN_DIR}/eval_step_${STEP}"

echo "=== Evaluation — ${VERSION} step ${STEP} ==="
echo "Checkpoint : ${MODEL_PATH}"
echo "Test data  : ${TEST_PARQUET}"
echo "Output     : ${EVAL_DIR}"
echo ""

python "${PROJECT_DIR}/scripts/omi/eval_checkpoint.py" \
  --checkpoint-dir "${MODEL_PATH}" \
  --test-parquet "${TEST_PARQUET}" \
  --output-dir "${EVAL_DIR}" \
  2>&1 | tee "${EVAL_DIR}.log"
