import os
os.environ['HF_HOME']            = '/home/data/projets-aps/projet6/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/home/data/projets-aps/projet6/hf_cache'
os.environ['HF_DATASETS_CACHE']  = '/home/data/projets-aps/projet6/hf_cache'
os.environ['TORCH_HOME']         = '/home/data/projets-aps/projet6/torch_cache'
os.makedirs('/home/data/projets-aps/projet6/hf_cache', exist_ok=True)
os.makedirs('/home/data/projets-aps/projet6/torch_cache', exist_ok=True)

# -*- coding: utf-8 -*-
"""
Fine-tuning Whisper Medium sur données Ewe.
- Langue de départ : Yoruba (yo) — langue africaine tonale proche de l'Ewe
- Données train : uniquement ewe-train-00*.parquet (~15 053 exemples)
- Pipeline audio : librosa.load(BytesIO(...)) — identique au premier succès (WER 27.42%)
- warmup=200, batch=8, lr=1e-5, epochs=5
"""

import io
import os
import pandas as pd
import torch
import librosa
import evaluate
from torch.nn.utils.rnn import pad_sequence
from datasets import DatasetDict, Dataset, concatenate_datasets
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import datasets as hf_datasets

# ══════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════
PATH_PARQUET = (
    "/home/data/projets-aps/projet6/data/data_ewe"
    "/speech_ug/parquet/Ewe_parquet"
)
CACHE_DIR = (
    "/home/data/projets-aps/projet6/data_preprocessing"
    "/pre_proccessing_ewe_whisper/data/Dataset_ewe_yoruba"
)
OUTPUT_DIR     = "./whisper-medium-ewe-yoruba"
SAVE_MODEL_DIR = "./whisper_medium_ewe_yoruba"
CHUNK_SIZE     = 500
MODEL_NAME     = "openai/whisper-medium"

# ══════════════════════════════════════════════════════════════
# 2. PROCESSOR
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print(f"Chargement du processor {MODEL_NAME}...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
tokenizer = processor.tokenizer

# ══════════════════════════════════════════════════════════════
# 3. PIPELINE AUDIO — librosa.load(BytesIO(...))
# Pipeline identique au premier entraînement réussi (WER 27.42%)
# ══════════════════════════════════════════════════════════════
def prepare_chunk(df_chunk, split_name, chunk_idx):
    input_features_list = []
    labels_list         = []
    skipped             = 0

    for idx, row in df_chunk.iterrows():
        try:
            # ── Audio : MP3 bytes → librosa direct ────────────
            audio_bytes = row["audio"]["bytes"]
            waveform, _ = librosa.load(
                io.BytesIO(audio_bytes), sr=16000, mono=True
            )

            # Trim silences début/fin
            waveform, _ = librosa.effects.trim(waveform, top_db=20)

            # Ignore les audios trop courts (< 0.1s)
            if len(waveform) < 1600:
                skipped += 1
                continue

            # ── Mel-spectrogram ────────────────────────────────
            inputs = processor.feature_extractor(
                waveform, sampling_rate=16000, return_tensors="pt"
            )
            input_features_list.append(inputs.input_features[0].numpy())

            # ── Labels ─────────────────────────────────────────
            text = str(row["transcription"]) \
                if pd.notna(row["transcription"]) else ""
            labels = processor.tokenizer(text, return_tensors="pt")
            labels_list.append(labels.input_ids[0].numpy())

        except Exception as e:
            print(f"    ⚠ Erreur ligne {idx} : {e}")
            skipped += 1
            continue

    print(f"    [{split_name}] chunk {chunk_idx} : "
          f"{len(input_features_list)} OK | {skipped} ignorés")

    return Dataset.from_dict({
        "input_features": input_features_list,
        "labels":         labels_list,
    })


def df_to_hf_dataset(df, split_name, chunk_size=CHUNK_SIZE):
    total    = len(df)
    n_chunks = (total // chunk_size) + (1 if total % chunk_size else 0)
    print(f"\n  {split_name} : {total} exemples → {n_chunks} chunks")

    chunks = []
    for i in range(n_chunks):
        start    = i * chunk_size
        end      = min(start + chunk_size, total)
        df_chunk = df.iloc[start:end].reset_index(drop=True)
        chunks.append(prepare_chunk(df_chunk, split_name, i + 1))

    full_ds = concatenate_datasets(chunks)
    print(f"  → {len(full_ds)} exemples au total")
    return full_ds


if os.path.exists(CACHE_DIR):
    print(f"\nDataset trouvé dans le cache : {CACHE_DIR}")
    dataset = hf_datasets.load_from_disk(CACHE_DIR)
    print(dataset)

else:
    print("\nPréparation depuis les parquet...")
    print("=" * 60)

    # ── Train : UNIQUEMENT ewe-train-00*.parquet (~15 053 ex) ─
    train_files = sorted([
        os.path.join(PATH_PARQUET, f)
        for f in os.listdir(PATH_PARQUET)
        if f.startswith("ewe-train-00") and f.endswith(".parquet")
    ])
    print(f"Fichiers train : {len(train_files)}")
    for f in train_files:
        print(f"  - {os.path.basename(f)}")

    train_df = pd.concat(
        [pd.read_parquet(f) for f in train_files], ignore_index=True
    )

    # ── Val et Test nettoyés (0 chevauchement) ────────────────
    val_df  = pd.read_parquet(
        os.path.join(PATH_PARQUET, "ewe_validation_clean.parquet")
    )
    test_df = pd.read_parquet(
        os.path.join(PATH_PARQUET, "ewe_test_clean.parquet")
    )

    print(f"\nTrain : {len(train_df)} | Val : {len(val_df)} | Test : {len(test_df)}")

    print("\nPréparation train...")
    train_dataset = df_to_hf_dataset(train_df, "train")

    print("\nPréparation validation (clean)...")
    val_dataset   = df_to_hf_dataset(val_df, "validation")

    print("\nPréparation test (clean)...")
    test_dataset  = df_to_hf_dataset(test_df, "test")

    dataset = DatasetDict({
        "train":      train_dataset,
        "validation": val_dataset,
        "test":       test_dataset,
    })

    print(f"\nSauvegarde dans {CACHE_DIR}...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    dataset.save_to_disk(CACHE_DIR)
    print("Dataset sauvegardé.")
    print(dataset)

# Format PyTorch
dataset.set_format(type="torch", columns=["input_features", "labels"])

# ══════════════════════════════════════════════════════════════
# 4. MODÈLE
# ══════════════════════════════════════════════════════════════
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice : {device}")
print(f"Chargement du modèle {MODEL_NAME}...")

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# ── Langue de départ : Yoruba (yo) ────────────────────────────
# Langue africaine tonale proche de l'Ewe
# Force le décodeur à générer des tokens africains dès le début
# → accélère la convergence vs partir de l'anglais
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="yo", task="transcribe"
)
model.config.forced_decoder_ids = forced_decoder_ids
model.config.suppress_tokens    = []
model.to(device)

# ══════════════════════════════════════════════════════════════
# 5. MÉTRIQUES WER
# ══════════════════════════════════════════════════════════════
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    pred_str  = [p.strip().lower() for p in pred_str]
    label_str = [l.strip().lower() for l in label_str]
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ══════════════════════════════════════════════════════════════
# 6. DATA COLLATOR — VERSION ORIGINALE (pad_sequence)
# ══════════════════════════════════════════════════════════════
class DataCollatorWhisper:
    def __call__(self, features):
        features = [f for f in features if f["input_features"] is not None
                    and f["labels"] is not None]

        input_features = [
            torch.tensor(f["input_features"]).squeeze(0)
            for f in features
        ]
        input_features_padded = pad_sequence(input_features, batch_first=True)

        labels = [
            torch.tensor(f["labels"]).squeeze(0)
            for f in features
        ]
        max_len       = max(l.size(0) for l in labels)
        labels_padded = torch.full(
            (len(labels), max_len), -100, dtype=torch.long
        )
        for i, l in enumerate(labels):
            labels_padded[i, : l.size(0)] = l

        return {
            "input_features": input_features_padded,
            "labels":         labels_padded,
        }

# ══════════════════════════════════════════════════════════════
# 7. ARGUMENTS D'ENTRAÎNEMENT
# Identiques au premier entraînement réussi (WER 27.42%)
# ══════════════════════════════════════════════════════════════
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,   # batch effectif = 16

    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=10,

    fp16=True,
    gradient_checkpointing=True,

    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,

    predict_with_generate=True,
    generation_max_length=250,

    logging_steps=50,
    report_to=["none"],

    push_to_hub=False,
)

# ══════════════════════════════════════════════════════════════
# 8. TRAINER
# ══════════════════════════════════════════════════════════════
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=DataCollatorWhisper(),
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# ══════════════════════════════════════════════════════════════
# 9. ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Début de l'entraînement — Whisper Medium Ewe (départ Yoruba)...")
print("=" * 60)
trainer.train()

# ══════════════════════════════════════════════════════════════
# 10. SAUVEGARDE
# ══════════════════════════════════════════════════════════════
print(f"\nSauvegarde dans {SAVE_MODEL_DIR}...")
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
model.save_pretrained(SAVE_MODEL_DIR)
processor.save_pretrained(SAVE_MODEL_DIR)
print("Modèle sauvegardé.")

# ══════════════════════════════════════════════════════════════
# 11. ÉVALUATION FINALE SUR LE TEST CLEAN
# ══════════════════════════════════════════════════════════════
print("\nÉvaluation finale sur le jeu de test nettoyé...")
test_results = trainer.evaluate(eval_dataset=dataset["test"])
wer = test_results.get("eval_wer", None)
if wer is not None:
    print(f"WER test final (clean) : {wer:.2f}%")
else:
    print(f"Résultats test : {test_results}")

print("\n" + "=" * 60)
print("Fin du script.")
print("=" * 60)
