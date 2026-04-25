import os
import torch
import pandas as pd
import numpy as np
import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import sacrebleu
from datasets import Dataset

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/data/projets-aps/projet6")
INPUT_CSV = BASE_DIR / "data_preprocessing/data_arab/unified_arabic_master_corpus.csv"
OUTPUT_DIR = BASE_DIR / "experiments/translation_arab_french_v2"

MODEL_NAME = "facebook/nllb-200-1.3B"
SRC_LANG = "arz_Arab"  # Egyptian Arabic
TGT_LANG = "fra_Latn"  # French
MAX_LENGTH = 128
SEED = 42

# ── DATA & METRICS ──────────────────────────────────────────────────────────
from sklearn.model_selection import GroupShuffleSplit

def create_datasets():
    # 1. Raw Load
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    # 2. Assign Pair IDs IMMEDIATELY with Name Filter
    df = df.reset_index(drop=True)
    
    # Initialize with unique IDs for everyone (default: pair-by-one)
    df['pair_id'] = df.index 

    # Filter for Kaggle: group these specifically by two
    # We use a mask to find rows belonging to Kaggle
    kaggle_mask = df['database_origin'].str.contains('Kaggle', na=False)
    
    if kaggle_mask.any():
        # Get the subset of indices that are Kaggle
        kaggle_indices = df[kaggle_mask].index
        # Create a sub-sequence (0, 1, 2, 3...) and divide by 2 -> (0, 0, 1, 1...)
        # We add a large offset to ensure these IDs never clash with non-Kaggle IDs
        df.loc[kaggle_mask, 'pair_id'] = (np.arange(len(kaggle_indices)) // 2) + 2000000

    # 3. Clean up (Now safe to drop rows without shifting IDs)
    total_before = len(df)
    
    # --- Drop NaN ---
    df = df.dropna(subset=['text_arab_egypt', 'text_french'])
    nan_dropped = total_before - len(df)
    if nan_dropped > 0:
        print(f"🗑️  Dropped {nan_dropped:,} rows with missing (NaN) Arabic or French text.")

    # --- Drop Duplicates ---
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['text_arab_egypt', 'text_french'])
    dupes_dropped = before_dedup - len(df)
    if dupes_dropped > 0:
        print(f"👯  Removed {dupes_dropped:,} exact bilingual duplicates (same Arabic AND French).")

    print(f"📊 Dataset size after cleaning: {len(df):,} rows (Total filtered: {total_before - len(df):,})")

    # 4. Grouped Split
    # Even though non-Kaggle rows have unique IDs, GroupShuffleSplit 
    # handles them perfectly as "groups of one."
    gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    train_idx, temp_idx = next(gss.split(df, groups=df['pair_id']))
    
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    # 5. Split temp into Val and Test
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df['pair_id']))
    
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    print(f"✅ Final Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return (train_df.drop(columns=['pair_id']), 
            val_df.drop(columns=['pair_id']), 
            test_df.drop(columns=['pair_id']))

def compute_metrics(eval_preds):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = [[l for l in label if l != -100] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean up whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    bleu = sacrebleu.corpus_bleu(decoded_preds, decoded_labels).score
    return {"bleu": bleu}

# ── TRAINING ────────────────────────────────────────────────────────────────
def train_model(train_df, val_df):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.src_lang = SRC_LANG
    tokenizer.tgt_lang = TGT_LANG
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=True)

    def preprocess_function(examples):
        return tokenizer(
            [str(ex) for ex in examples["text_arab_egypt"]],
            text_target=[str(ex) for ex in examples["text_french"]],
            max_length=MAX_LENGTH,
            truncation=True
        )

    train_ds = Dataset.from_pandas(train_df).map(preprocess_function, batched=True)
    val_ds = Dataset.from_pandas(val_df).map(preprocess_function, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=1,            # Only keep one best checkpoint
        num_train_epochs=10,
        predict_with_generate=True,
        fp16=True,                     # Faster training on GPU
        load_best_model_at_end=True,   # Required for EarlyStopping
        metric_for_best_model="bleu",
        greater_is_better=True,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=50,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # --- Step 1: Baseline Evaluation (Zero-Shot) ---
    print("\n" + "="*50)
    print("🔍 RUNNING INITIAL BASELINE EVALUATION (ZERO-SHOT)")
    print("="*50)

    # This evaluates the base model on your val_dataset
    baseline_results = trainer.evaluate()

    print(f"\n📈 BASELINE METRICS:")
    for key, value in baseline_results.items():
        print(f"  - {key}: {value}")
    print("="*50 + "\n")

    # --- Step 2: Start Fine-Tuning ---
    print("🚀 STARTING FINE-TUNING...")
    train_result = trainer.train()

    # 💾 SAVE LOG HISTORY (The "Evolution" part)
    # This captures the metrics recorded at every eval/logging step
    history_path = OUTPUT_DIR / "benchmarking_evolution.json"
    with open(history_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=4)
    print(f"📈 Evolution history saved to {history_path}")

    # Final Model Save
    model.save_pretrained(OUTPUT_DIR / "final_model")
    tokenizer.save_pretrained(OUTPUT_DIR / "final_model")

    # 🧹 Cleanup
    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    return model, tokenizer

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = create_datasets()
    final_model, final_tokenizer = train_model(train_df, val_df)
    print("✅ Training Complete and History Logged.")