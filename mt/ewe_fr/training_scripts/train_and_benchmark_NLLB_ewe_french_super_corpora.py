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
# CHANGE THIS TO "4" FOR THE SECOND RUN
CORPUS_TYPE = "2" 

BASE_DIR = Path("/home/data/projets-aps/projet6")
INPUT_CSV = BASE_DIR / f"data_preprocessing/ewe/ewe_super_corpus_ratio_1_to_{CORPUS_TYPE}.csv"
# Pointing to experiments2 for space
OUTPUT_DIR = BASE_DIR / f"experiments2/translation_ewe_french_{CORPUS_TYPE}"

MODEL_NAME = "facebook/nllb-200-1.3B"
SRC_LANG = "ewe_Latn"
TGT_LANG = "fra_Latn"
MAX_LENGTH = 128
SEED = 42

# ── DATASET CREATION (Logic for Pure Testing) ───────────────────────────────
def create_datasets():
    print(f"📊 Loading Super Corpus: {INPUT_CSV.name}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    # 1. Cleaning
    df = df.dropna(subset=['text_ewe', 'text_french'])
    
    # 2. Separate Synthetic from Organic
    # The origin was standardized to 'english_ewe_translated' in previous steps
    is_synthetic = df['database_origin'].str.contains('english_ewe_translated', na=False)
    df_synthetic = df[is_synthetic].copy()
    df_organic = df[~is_synthetic].copy()
    
    print(f"💎 Organic Rows: {len(df_organic):,}")
    print(f"🤖 Synthetic Rows: {len(df_synthetic):,}")

    # 3. Split ONLY Organic into Train/Val/Test
    # This ensures Val/Test contain zero synthetic data
    train_org, temp_org = train_test_split(df_organic, test_size=0.10, random_state=SEED)
    val_df, test_df = train_test_split(temp_org, test_size=0.50, random_state=SEED)
    
    # 4. Add ALL Synthetic data back to Training
    train_df = pd.concat([train_org, df_synthetic], ignore_index=True).sample(frac=1, random_state=SEED)

    print(f"✅ Final Setup: Train={len(train_df)} (inc. synthetic), Val={len(val_df)} (organic), Test={len(test_df)} (organic)")
    return train_df, val_df, test_df

# ── METRICS ─────────────────────────────────────────────────────────────────
def compute_metrics(eval_preds):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = [[l for l in label if l != -100] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    bleu = sacrebleu.corpus_bleu(decoded_preds, decoded_labels).score
    chrf = sacrebleu.corpus_chrf(decoded_preds, decoded_labels).score
    
    return {"bleu": bleu, "chrf": chrf}

# ── TRAINING ────────────────────────────────────────────────────────────────
def train_model(train_df, val_df, test_df):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.src_lang = SRC_LANG
    tokenizer.tgt_lang = TGT_LANG
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def preprocess_function(examples):
        return tokenizer(
            [str(ex) for ex in examples["text_ewe"]],
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
        save_total_limit=1,
        num_train_epochs=10, # Up to 10 with Early Stopping
        predict_with_generate=True,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=100,
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

    # --- Step 1: Zero-Shot Baseline ---
    print("\n🔍 RUNNING INITIAL BASELINE EVALUATION")
    baseline_results = trainer.evaluate()
    
    # --- Step 2: Fine-Tuning ---
    print("\n🚀 STARTING FINE-TUNING...")
    trainer.train()

    # --- Step 3: Final Test Evaluation ---
    print("\n🧪 RUNNING FINAL TEST EVALUATION")
    test_ds = Dataset.from_pandas(test_df).map(preprocess_function, batched=True)
    final_test_results = trainer.evaluate(test_ds)

    # 💾 Save History & Results
    history_path = OUTPUT_DIR / "benchmarking_evolution.json"
    with open(history_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=4)

    report = {
        "baseline_val": baseline_results,
        "final_test": final_test_results
    }
    with open(OUTPUT_DIR / "improvement_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # Save Model
    model.save_pretrained(OUTPUT_DIR / "final_model")
    tokenizer.save_pretrained(OUTPUT_DIR / "final_model")

    # Cleanup
    shutil.rmtree(OUTPUT_DIR / "checkpoints", ignore_errors=True)
    return model

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = create_datasets()
    train_model(train_df, val_df, test_df)
    print(f"✅ Training Complete. Results in {OUTPUT_DIR}")