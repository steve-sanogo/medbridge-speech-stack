#!/bin/bash
###############################################################################
# Run training locally or interactively on the HPC (srun / salloc)
###############################################################################
# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_DIR="/home/data/projets-aps/projet6/MedBridge-AI2"
ENV_DIR="/home/data/projets-aps/projet6/env_aps"
OUTPUT_DIR="/home/data/projets-aps/projet6/MedBridge-AI2/outputs/checkpoints/med_asr_ewe_v1"
CONFIG_FILE="${PROJECT_DIR}/configs/omnilingual_ctc_finetune.yaml"

# ─── Environment ────────────────────────────────────────────────────────────
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"
set -uo pipefail

# ─── Asset cards (ABSOLUTE path — critical) ──────────────────────────────────
export FAIRSEQ2_USER_ASSET_CARD_DIRS="${PROJECT_DIR}/custom_cards"

echo "FAIRSEQ2_USER_ASSET_CARD_DIRS = ${FAIRSEQ2_USER_ASSET_CARD_DIRS}"
echo "Config                        = ${CONFIG_FILE}"
echo "Output                        = ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_DIR}"

python -m workflows.recipes.wav2vec2.asr "${OUTPUT_DIR}" \
  --config-file "${CONFIG_FILE}" \
  2>&1 | tee "${OUTPUT_DIR}/train.log"