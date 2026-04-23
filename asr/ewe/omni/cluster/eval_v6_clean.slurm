#!/bin/bash
###############################################################################
# SLURM — Evaluation V6_clean step 17000
###############################################################################
#SBATCH --job-name=eval-v6c
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/eval_v6c_%j.out
#SBATCH --error=logs/eval_v6c_%j.err

PROJECT_DIR="/home/data/projets-aps/projet6/MedBridge-AI2"
ENV_DIR="/home/data/projets-aps/projet6/env_aps_sanogo"

CHECKPOINT="/home/data/projets-aps/projet6/MedBridge-AI2/outputs/checkpoints/med_asr_ewe_v6_clean/ws_1.58e3de17/checkpoints/step_17000/model"
TEST_PARQUET="/home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet_omni_v6/version=0/corpus=general/split=test/language=ewe_Latn/part-0000.parquet"
OUTPUT_DIR="/home/data/projets-aps/projet6/MedBridge-AI2/outputs/checkpoints/med_asr_ewe_v6_clean/eval_step_17000_norm"

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"
set -uo pipefail

export FAIRSEQ2_USER_ASSET_CARD_DIRS="${PROJECT_DIR}/custom_cards"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TMPDIR=/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "=== Evaluation — v6_clean step 17000 ==="
echo "Checkpoint : ${CHECKPOINT}"
echo "Test data  : ${TEST_PARQUET}"
echo "Output     : ${OUTPUT_DIR}"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

cd "${PROJECT_DIR}"
python scripts/omi/eval_checkpoint.py \
    --checkpoint-dir "${CHECKPOINT}" \
    --test-parquet   "${TEST_PARQUET}" \
    --output-dir     "${OUTPUT_DIR}" \
    --batch-size     8 \
    2>&1 | tee "${OUTPUT_DIR}/eval_step_17000.log"

echo "Eval finished at $(date)"
