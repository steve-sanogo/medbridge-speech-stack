"""
backtranslation_benchmark.py
─────────────────────────────
Pipeline de backtranslation Ewe → Français → Ewe avec NLLB-200-1.3B.
Évaluation : BLEU, chrF + export pour analyse manuelle.

Usage:
  python backtranslation_benchmark.py --batch-size 8 --max-samples 100
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import sacrebleu

# ── Chemins ────────────────────────────────────────────────────────────────────
BASE         = Path("/home/data/projets-aps/projet6")
DATA_DIR     = BASE / "data" / "data_ewe" / "ewe_tts"
RESULTS_DIR  = BASE / "experiments" / "backtranslation"
HF_CACHE     = str(BASE / "hf_cache")

os.environ["HF_HOME"]           = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = f"{HF_CACHE}/datasets"

# ── Modèle ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "facebook/nllb-200-1.3B"
SRC_LANG   = "ewe_Latn"    # Ewe
PIVOT_LANG = "fra_Latn"    # Français

# ── Utilitaires ────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("  ⚠ Pas de GPU détecté, utilisation du CPU")
    return dev

def load_model(device: torch.device):
    print(f"\n[1/4] Chargement du modèle {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
        
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        cache_dir=HF_CACHE,
        use_safetensors=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    print(f"  ✓ Modèle chargé")
    return tokenizer, model

def translate_batch(
    texts: list[str],
    tokenizer,
    model,
    src_lang: str,
    tgt_lang: str,
    device: torch.device,
    batch_size: int = 8,
    max_length: int = 256,
) -> list[str]:
    tokenizer.src_lang = src_lang
    all_translations = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=max_length,
                num_beams=4,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_translations.extend(decoded)
        print(f"    {min(i + batch_size, len(texts))}/{len(texts)} traduits...", end="\r")

    print()
    return all_translations

# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-length",  type=int, default=256)
    p.add_argument("--split",       type=str, default="train", choices=["train", "validation", "test"])
    return p.parse_args()

def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Backtranslation Benchmark — (COMET Removed)")
    print(f"  {SRC_LANG} → {PIVOT_LANG} → {SRC_LANG}")
    print("=" * 60)

    device = get_device()

    # ── 1. Chargement des données ──────────────────────────────────────────────
    print(f"\n[2/4] Chargement des données ({args.split})...")
    
    parquet_path = DATA_DIR / f"{args.split}.parquet"
    if not parquet_path.exists():
        # Fallback if specific file naming exists
        shards = list(DATA_DIR.glob(f"ewe-{args.split}-*.parquet"))
        if shards:
            parquet_path = shards[0]
        else:
            raise FileNotFoundError(f"Fichier {args.split}.parquet introuvable dans {DATA_DIR}")

    ds = load_dataset("parquet", data_files={"train": str(parquet_path)}, split="train")
    
    # On évite de charger l'audio en mémoire
    keep_cols = [c for c in ds.column_names if c != "audio"]
    df = ds.select_columns(keep_cols).to_pandas()
    
    # Détection de la colonne texte
    col = "text" if "text" in df.columns else df.select_dtypes(include=['object']).columns[0]
    texts = df[col].dropna().tolist()
    
    if args.max_samples:
        texts = texts[: args.max_samples]
    print(f"  ✓ {len(texts)} textes chargés depuis '{col}'")

    # ── 2. Modèle & Traduction ────────────────────────────────────────────────
    tokenizer, model = load_model(device)

    print(f"\n[3/4] Forward: {SRC_LANG} → {PIVOT_LANG}")
    t0 = time.time()
    fr_texts = translate_batch(texts, tokenizer, model, SRC_LANG, PIVOT_LANG, device, args.batch_size, args.max_length)
    t_fwd = time.time() - t0

    print(f"\n[4/4] Backward: {PIVOT_LANG} → {SRC_LANG}")
    t1 = time.time()
    back_texts = translate_batch(fr_texts, tokenizer, model, PIVOT_LANG, SRC_LANG, device, args.batch_size, args.max_length)
    t_bwd = time.time() - t0

    # ── 5. Métriques ──────────────────────────────────────────────────────────
    bleu = sacrebleu.corpus_bleu(back_texts, [texts])
    chrf = sacrebleu.corpus_chrf(back_texts, [texts])

    metrics = {
        "BLEU": round(bleu.score, 4),
        "chrF": round(chrf.score, 4),
        "n_samples": len(texts),
        "time_total_s": round(t_fwd + t_bwd, 2)
    }

    print("\n  ┌─────────────────────────────────┐")
    print(f"  │  BLEU: {metrics['BLEU']:>22}  │")
    print(f"  │  chrF: {metrics['chrF']:>22}  │")
    print("  └─────────────────────────────────┘")

    # ── Sauvegarde ─────────────────────────────────────────────────────────────
    ts = time.strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame({"source": texts, "pivot": fr_texts, "backtranslation": back_texts})
    
    csv_out = RESULTS_DIR / f"backtrad_{ts}.csv"
    results_df.to_csv(csv_out, index=False)
    
    with open(RESULTS_DIR / f"metrics_{ts}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Terminé. Résultats sauvegardés dans {RESULTS_DIR}")

if __name__ == "__main__":
    main()