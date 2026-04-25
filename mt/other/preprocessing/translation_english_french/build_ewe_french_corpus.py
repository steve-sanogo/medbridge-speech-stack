# -*- coding: utf-8 -*-
"""
build_ewe_french_corpus.py
---------------------------
Traduit English -> French avec NLLB et produit un CSV:
    database_origin, id, text_ewe, text_french

Supporte la reprise automatique (checkpoint) si le job est interrompu.
"""

import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-1.3B"
SRC_LANG   = "eng_Latn"
TGT_LANG   = "fra_Latn"

parser = argparse.ArgumentParser()
parser.add_argument("--english",    required=True)
parser.add_argument("--ewe",        required=True)
parser.add_argument("--output",     required=True)
parser.add_argument("--origin",     default="AI4D_MT")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--max_length", type=int, default=128)
args = parser.parse_args()

print("Lecture des fichiers...")
with open(args.english, "r", encoding="utf-8") as f:
    english_lines = [line.strip() for line in f if line.strip()]
with open(args.ewe, "r", encoding="utf-8") as f:
    ewe_lines = [line.strip() for line in f if line.strip()]

assert len(english_lines) == len(ewe_lines)
total = len(english_lines)
print(f"   {total} lignes chargees")

# Checkpoint: reprendre si fichier existe
start_idx = 0
if os.path.exists(args.output):
    existing = pd.read_csv(args.output)
    start_idx = len(existing)
    print(f"   Reprise depuis la ligne {start_idx}/{total}")
else:
    print(f"   Nouveau fichier")

if start_idx >= total:
    print("Deja termine!")
    exit(0)

print(f"\nChargement du modele {MODEL_NAME}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir="/home/data/projets-aps/projet6/hf_cache"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    cache_dir="/home/data/projets-aps/projet6/hf_cache"
).to(device)
model.eval()
print("   Modele charge")

tokenizer.src_lang = SRC_LANG
forced_bos = tokenizer.convert_tokens_to_ids(TGT_LANG)

print(f"\nTraduction... lignes restantes: {total - start_idx}")

bs = args.batch_size
write_header = start_idx == 0
mode = "w" if write_header else "a"

with open(args.output, mode, encoding="utf-8", buffering=1) as out_f:
    if write_header:
        out_f.write("database_origin,id,text_ewe,text_french\n")

    pbar = tqdm(range(start_idx, total, bs), desc="Translating")
    for i in pbar:
        batch_en  = english_lines[i : i + bs]
        batch_ewe = ewe_lines[i : i + bs]

        inputs = tokenizer(
            batch_en,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=args.max_length,
                num_beams=4
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, (ewe, fr) in enumerate(zip(batch_ewe, decoded)):
            idx = i + j
            ewe_safe = '"' + ewe.replace('"', '""') + '"'
            fr_safe  = '"' + fr.replace('"', '""') + '"'
            out_f.write(f"{args.origin},{idx},{ewe_safe},{fr_safe}\n")

        if i % 5000 < bs:
            out_f.flush()

print(f"\nTermine! {args.output} pret.")
