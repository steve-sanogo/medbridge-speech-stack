#!/bin/bash
###############################################################################
# Smoke check — validate environment + dataset BEFORE submitting a real job
#
# Run this interactively:
#   bash scripts/omi/smoke_check.sh
###############################################################################
PROJECT_DIR="/home/data/projets-aps/projet6/MedBridge-AI2"
ENV_DIR="/home/data/projets-aps/projet6/env_aps"
DATA_DIR="/home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet_omni"

# Activer conda (set +u nécessaire car les scripts conda utilisent des variables non définies)
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"
set -uo pipefail

echo "============================================"
echo "  SMOKE CHECK — Omnilingual ASR pipeline"
echo "============================================"
echo ""

# ─── 1. GPU ──────────────────────────────────────────────────────────────────
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  N/A (noeud de login)"
echo ""

# ─── 2. Python imports ───────────────────────────────────────────────────────
echo "--- Python imports ---"
python -c "
import torch
import torchaudio
import fairseq2n
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
print(f'torch:      {torch.__version__}')
print(f'torchaudio: {torchaudio.__version__}')
if torch.cuda.is_available():
    print(f'CUDA:       True ({torch.cuda.get_device_name(0)})')
    print(f'bf16:       {torch.cuda.is_bf16_supported()}')
else:
    print('CUDA:       False (pas de GPU)')
print('All imports OK')
"
echo ""

# ─── 3. Dataset structure (Hive-partitioned) ─────────────────────────────────
echo "--- Dataset structure ---"
CORPUS="general"
LANG="ewe_Latn"

for split in train dev test; do
    split_dir="${DATA_DIR}/version=0/corpus=${CORPUS}/split=${split}/language=${LANG}"
    if [ -d "$split_dir" ]; then
        count=$(ls "$split_dir"/part-*.parquet 2>/dev/null | wc -l)
        size=$(du -sh "$split_dir" 2>/dev/null | cut -f1)
        echo "  [OK]  split=${split}  (${count} file(s), ${size})"
    else
        echo "  [MISSING]  split=${split}  (${split_dir})"
    fi
done

tsv="${DATA_DIR}/version=0/language_distribution_0.tsv"
if [ -f "$tsv" ]; then
    echo "  [OK]  language_distribution_0.tsv"
    echo "        Content:"
    cat "$tsv" | sed 's/^/          /'
else
    echo "  [MISSING]  language_distribution_0.tsv"
fi
echo ""

# ─── 4. Asset card ───────────────────────────────────────────────────────────
echo "--- Asset card ---"
card="${PROJECT_DIR}/custom_cards/datasets/med_custom.yaml"
if [ -f "$card" ]; then
    echo "  [OK]  med_custom.yaml"
    cat "$card" | sed 's/^/          /'
else
    echo "  [MISSING]  med_custom.yaml"
fi
echo ""

# ─── 5. Config file ──────────────────────────────────────────────────────────
echo "--- Training config ---"
config="${PROJECT_DIR}/configs/omnilingual_ctc_finetune.yaml"
if [ -f "$config" ]; then
    echo "  [OK]  omnilingual_ctc_finetune.yaml"
else
    echo "  [MISSING]  omnilingual_ctc_finetune.yaml"
fi
echo ""

# ─── 6. Workflows module ─────────────────────────────────────────────────────
echo "--- Workflows module ---"
if [ -f "${PROJECT_DIR}/workflows/recipes/wav2vec2/asr/__main__.py" ]; then
    echo "  [OK]  workflows/recipes/wav2vec2/asr/"
else
    echo "  [MISSING]  workflows/ (clone depuis GitHub requis)"
fi
echo ""

# ─── 7. Parquet content validation ───────────────────────────────────────────
echo "--- Parquet content validation ---"
export DATA_DIR_ENV="${DATA_DIR}"
python << 'PYEOF'
import pyarrow.parquet as pq
import sys, os

data_dir = os.environ["DATA_DIR_ENV"]
corpus = "general"
lang = "ewe_Latn"
errors = []

for split in ["train", "dev", "test"]:
    split_dir = f"{data_dir}/version=0/corpus={corpus}/split={split}/language={lang}"
    parquet_files = sorted([f for f in os.listdir(split_dir) if f.endswith(".parquet")]) if os.path.isdir(split_dir) else []

    if not parquet_files:
        errors.append(f"split={split}: no parquet files")
        print(f"  {split}: NO FILES")
        continue

    for pf in parquet_files:
        path = os.path.join(split_dir, pf)
        table = pq.read_table(path)
        n = len(table)
        cols = table.column_names
        print(f"  {split}/{pf}: {n} rows, columns: {cols}")

        # Check required columns (Omnilingual format)
        for col in ["text", "audio_bytes", "audio_size"]:
            if col not in cols:
                errors.append(f"{split}/{pf}: missing column '{col}'")

        # Check for empty text
        if "text" in cols:
            texts = table.column("text").to_pylist()
            empty = sum(1 for t in texts if not t or not str(t).strip())
            if empty > 0:
                errors.append(f"{split}/{pf}: {empty} empty texts")
                print(f"    WARNING: {empty} empty texts")

print()
if errors:
    print("ERRORS FOUND:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("All checks passed.")
PYEOF

echo ""
echo "============================================"
echo "  Smoke check complete"
echo "============================================"
