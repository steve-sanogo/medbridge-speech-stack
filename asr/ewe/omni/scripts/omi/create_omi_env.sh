#!/bin/bash
###############################################################################
# Create the conda environment for Omnilingual ASR on HPC
#
# Helios: CUDA 12.1, x86_64, pas de lmod
# Peut s'exécuter depuis le nœud de login (pas besoin de GPU pour installer).
###############################################################################

ENV_NAME="omnilingual"

echo "=== Creating conda environment: ${ENV_NAME} ==="

# Supprimer l'ancien env s'il existe (pour repartir propre)
conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true

# Créer l'env avec Python 3.10
conda create -n "${ENV_NAME}" python=3.10 pip -y

echo ""
echo "=== Vérification de l'env ==="
conda run -n "${ENV_NAME}" python --version
conda run -n "${ENV_NAME}" python -m pip --version

echo ""
echo "=== 1/4 Installing omnilingual-asr (+ fairseq2) ==="
conda run -n "${ENV_NAME}" python -m pip install 'omnilingual-asr[data]'

echo ""
echo "=== 2/4 Detecting torch version required by fairseq2n ==="
# fairseq2n exige une version exacte de torch — on la lit depuis le package
REQUIRED_TORCH=$(conda run -n "${ENV_NAME}" python -c "
import importlib, re
spec = importlib.util.find_spec('fairseq2n')
if spec and spec.origin:
    with open(spec.origin) as f:
        m = re.search(r'target_version\s*=\s*\"(\d+\.\d+\.\d+)\"', f.read())
        if m: print(m.group(1))
" 2>/dev/null)

if [ -z "${REQUIRED_TORCH}" ]; then
    echo "WARNING: Impossible de détecter la version torch requise par fairseq2n."
    echo "         Tentative avec 2.8.0 par défaut."
    REQUIRED_TORCH="2.8.0"
fi

echo "fairseq2n requiert torch==${REQUIRED_TORCH}"

# Helios a CUDA 12.1 (nvcc --version)
CUDA_TAG="cu121"

echo ""
echo "=== 3/4 Installing torch==${REQUIRED_TORCH} + torchaudio (${CUDA_TAG}) ==="
conda run -n "${ENV_NAME}" python -m pip install \
    "torch==${REQUIRED_TORCH}" \
    "torchaudio==${REQUIRED_TORCH}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo ""
echo "=== 4/4 Pinning numpy ==="
conda run -n "${ENV_NAME}" python -m pip install 'numpy>=2.0,<2.1'

echo ""
echo "=== Verification ==="
conda run -n "${ENV_NAME}" python -c "import torch;      print(f'torch:      {torch.__version__}')"
conda run -n "${ENV_NAME}" python -c "import torchaudio;  print(f'torchaudio: {torchaudio.__version__}')"
conda run -n "${ENV_NAME}" python -c "import numpy;      print(f'numpy:      {numpy.__version__}')"
conda run -n "${ENV_NAME}" python -c "import fairseq2n;   print('fairseq2n:  OK')"

echo ""
echo "=== Environment '${ENV_NAME}' ready ==="
echo "Pour activer : conda activate ${ENV_NAME}"
