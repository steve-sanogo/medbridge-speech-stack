#!/bin/bash
###############################################################################
# Crée l'arborescence MedBridge-AI2 sur le HPC
# et copie les fichiers nécessaires depuis le projet local.
#
# Usage (sur le HPC) :
#   bash setup_hpc.sh
###############################################################################
set -euo pipefail

BASE="/home/data/projets-aps/projet6"
PROJECT="${BASE}/MedBridge-AI2"

echo "=== Création de l'arborescence : ${PROJECT} ==="

mkdir -p "${PROJECT}/cluster"
mkdir -p "${PROJECT}/configs"
mkdir -p "${PROJECT}/custom_cards/datasets"
mkdir -p "${PROJECT}/data"
mkdir -p "${PROJECT}/logs"
mkdir -p "${PROJECT}/notebooks/exploration"
mkdir -p "${PROJECT}/outputs"
mkdir -p "${PROJECT}/scripts/omi"
mkdir -p "${PROJECT}/src/data"
mkdir -p "${PROJECT}/src/models"
mkdir -p "${PROJECT}/src/pipelines"
mkdir -p "${PROJECT}/src/utils"
mkdir -p "${PROJECT}/tests"

echo "Arborescence créée."
echo ""

# Lien symbolique vers les données parquet (pas de copie, on économise l'espace)
DATA_SRC="${BASE}/data/data_ewe"
if [ -d "${DATA_SRC}" ]; then
    ln -sfn "${DATA_SRC}" "${PROJECT}/data/data_ewe"
    echo "Lien symbolique créé : data/data_ewe → ${DATA_SRC}"
else
    echo "ATTENTION : ${DATA_SRC} introuvable. Crée le lien manuellement."
fi

echo ""
echo "=== Structure créée ==="
find "${PROJECT}" -type d | sort | sed "s|${PROJECT}|MedBridge-AI2|"
echo ""
echo "Copie maintenant tes fichiers de config dans ${PROJECT}/"
echo "Ou utilise scp depuis ta machine locale :"
echo "  scp -r configs/ custom_cards/ cluster/ scripts/ user@hpc:${PROJECT}/"
