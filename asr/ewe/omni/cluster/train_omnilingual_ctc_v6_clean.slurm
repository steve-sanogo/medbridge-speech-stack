#!/bin/bash
###############################################################################
# SLURM — Omnilingual CTC 300M fine-tuning on Ewe
###############################################################################
#SBATCH --job-name=omni-ewe-clean
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=helios
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/omni_ewe_%j.out
#SBATCH --error=logs/omni_ewe_%j.err

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_DIR="/home/data/projets-aps/projet6/MedBridge-AI2"
ENV_DIR="/home/data/projets-aps/projet6/env_aps_sanogo"
OUTPUT_DIR="/home/data/projets-aps/projet6/MedBridge-AI2/outputs/checkpoints/med_asr_ewe_v6_clean"
CONFIG_FILE="${PROJECT_DIR}/configs/omnilingual_ctc_finetune_v6_clean.yaml"

# ─── Environment (set +u car conda utilise des variables non définies) ───────
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"
set -uo pipefail

# Tell fairseq2 where custom dataset cards live
export FAIRSEQ2_USER_ASSET_CARD_DIRS="${PROJECT_DIR}/custom_cards"

# NCCL / CUDA tuning
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TMPDIR=/home/data/projets-aps/projet6/cache_sanogo/tmp_clean
export FAIRSEQ2_CACHE_DIR=/home/data/projets-aps/projet6/cache_sanogo/fairseq2_clean
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
mkdir -p "${TMPDIR}"
mkdir -p "${FAIRSEQ2_CACHE_DIR}"
# ─── Pre-flight checks ──────────────────────────────────────────────────────
echo "========================================="
echo "Job ID       : ${SLURM_JOB_ID:-unknown}"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "CUDA version : $(nvcc --version 2>/dev/null | tail -1 || echo 'nvcc not found')"
echo "Python       : $(python --version 2>&1)"
echo "torch        : $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "Config       : ${CONFIG_FILE}"
echo "Output       : ${OUTPUT_DIR}"
echo "Asset cards  : ${FAIRSEQ2_USER_ASSET_CARD_DIRS}"
echo "========================================="

# Arrêter si pas de GPU (on est sur un nœud de calcul, il DOIT y avoir un GPU)
python -c "import torch; assert torch.cuda.is_available(), 'ERREUR: Pas de GPU détecté !'"

# ─── Create output dir ──────────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "$0")/../logs"

# ─── Launch training ─────────────────────────────────────────────────────────
cd "${PROJECT_DIR}"

python -m workflows.recipes.wav2vec2.asr "${OUTPUT_DIR}" \
  --config-file "${CONFIG_FILE}" \
  2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "Training finished at $(date)"
echo "Checkpoints in: ${OUTPUT_DIR}"
