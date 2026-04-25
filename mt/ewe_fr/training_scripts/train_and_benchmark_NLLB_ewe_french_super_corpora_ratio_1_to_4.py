import os
import torch
import pandas as pd
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

# ── CONFIG & HYPERPARAMETERS ────────────────────────────────────────────────
BASE_DIR = Path("/home/data/projets-aps/projet6")
# Dossiers sources
BASELINE_DIR = BASE_DIR / "data_preprocessing/ewe/splitting_baseline"
RATIO_DIR = BASE_DIR / "data_preprocessing/ewe/baseline_plus_ratios"

# Chemins des fichiers (Le train est déjà pré-chaîné et filtré dans RATIO_DIR)
INPUT_VAL_CSV = BASELINE_DIR / "val_baseline.csv"
INPUT_TEST_CSV = BASELINE_DIR / "test_baseline.csv"

# Sélectionne ici le fichier chaîné (Ratio 1:2 ou 1:4)
# Note: Ce fichier contient déjà [Train_Baseline + Synth_Data]
INPUT_TRAIN_CHAINED = RATIO_DIR / "test_baseline_plus_english_ewe_ratio_1_to_4.csv"

# SAVE_DIR_SPLIT = BASE_DIR / "data_preprocessing/ewe/splitting_baseline"
OUTPUT_DIR = BASE_DIR / "experiments/translation_ewe_french_super_corpus_ratio_1_to_4"

MODEL_NAME = "facebook/nllb-200-1.3B"
SRC_LANG = "ewe_Latn"
TGT_LANG = "fra_Latn"
MAX_LENGTH = 128
SEED = 42

# Training Constants
BATCH_SIZE_PER_DEVICE = 16    # Physical batch size on GPU
ACCUMULATION_STEPS = 1       
LEARNING_RATE = 2e-5         # Adjusted for larger effective batch
EPOCHS = 10
PATIENCE = 2                 # Early stopping rounds
EVAL_METRIC = "eval_bleu"

# ── 1. DATASET CREATION & SAVING ────────────────────────────────────────────
def create_datasets():
    """
    Charge les datasets Éwé (Train augmenté, Val, Test).
    Supprime sim_score pour éviter les erreurs PyArrow.
    """
    print(f"📂 Loading Augmented Train Dataset: {INPUT_TRAIN_CHAINED.name}")
    train_df = pd.read_csv(INPUT_TRAIN_CHAINED)
    
    print(f"📂 Loading Validation & Test from Baseline...")
    val_df = pd.read_csv(INPUT_VAL_CSV)
    test_df = pd.read_csv(INPUT_TEST_CSV)
    
    # --- NETTOYAGE DES COLONNES ---
    # On retire sim_score s'il existe pour éviter le crash "ArrowTypeError"
    for df in [train_df, val_df, test_df]:
        if 'sim_score' in df.columns:
            df.drop(columns=['sim_score'], inplace=True)
            
    # Gestion des valeurs manquantes et mélange
    train_df = train_df.dropna(subset=['text_ewe', 'text_french'])
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"✅ Data Loading Complete. 'sim_score' column removed.")
    print(f"📊 Final Train size: {len(train_df)}")
    
    return train_df, val_df, test_df

# ── 2. METRICS ──────────────────────────────────────────────────────────────
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

# ── 3. TRAINING WRAPPER ─────────────────────────────────────────────────────
def train_model(train_df, val_df, test_df):
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    print("🚀 Initializing Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.src_lang = SRC_LANG
    tokenizer.tgt_lang = TGT_LANG
    
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=True)

    def preprocess_function(examples):
        return tokenizer(
            [str(ex) for ex in examples["text_ewe"]],
            text_target=[str(ex) for ex in examples["text_french"]],
            max_length=MAX_LENGTH,
            truncation=True
        )

    print("🛠 Tokenizing datasets...")
    train_ds = Dataset.from_pandas(train_df).map(preprocess_function, batched=True)
    val_ds = Dataset.from_pandas(val_df).map(preprocess_function, batched=True)
    test_ds = Dataset.from_pandas(test_df).map(preprocess_function, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
        per_device_eval_batch_size=BATCH_SIZE_PER_DEVICE,
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model=EVAL_METRIC,
        greater_is_better=True,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=10,
        report_to="none"
    )

    # Log Parameters for SLURM audit
    print("\n📝 --- TRAINING PARAMETERS ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Effective Batch Size: {BATCH_SIZE_PER_DEVICE * ACCUMULATION_STEPS}")
    print(f"Physical Batch Size: {BATCH_SIZE_PER_DEVICE}")
    print(f"Gradient Accumulation: {ACCUMULATION_STEPS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Evaluation metric: {EVAL_METRIC}")
    print(f"Epochs: {EPOCHS} (Patience: {PATIENCE})")
    print("----------------------------\n")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
    )

    def run_full_benchmark(tag="Baseline"):
        print(f"\n🔎 --- {tag} Full Evaluation ---")
        val_res = trainer.evaluate(val_ds, metric_key_prefix="val")
        test_res = trainer.evaluate(test_ds, metric_key_prefix="test")
        print(f"📊 {tag} - Val BLEU: {val_res['val_bleu']:.2f} | Test BLEU: {test_res['test_bleu']:.2f}")
        return {"val": val_res, "test": test_res}

    # Step 1: Initial Benchmark
    initial_metrics = run_full_benchmark("INITIAL")

    # Step 2: Fine-Tuning
    print("🚀 Starting Fine-Tuning...")
    trainer.train()

    # Step 3: Final Benchmark
    final_metrics = run_full_benchmark("FINAL")

    # Save History & Comparison
    with open(OUTPUT_DIR / "benchmarking_evolution.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=4)

    # Save results
    report = {"initial": initial_metrics, "final": final_metrics}
    with open(OUTPUT_DIR / "improvement_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # Save Model & Tokenizer
    model.save_pretrained(OUTPUT_DIR / "final_model")
    tokenizer.save_pretrained(OUTPUT_DIR / "final_model")
    
    print(f"✅ Training Complete. Model saved to {OUTPUT_DIR / 'final_model'}")

    if (OUTPUT_DIR / "checkpoints").exists():
        shutil.rmtree(OUTPUT_DIR / "checkpoints")

if __name__ == "__main__":

    print("*"*50)
    print(f"Script : train_and_benchmark_NLLB_ewe_french_super_corpora_ratio_1_to_4.py")
    print("*"*50)
    print(f"✅ Starting Ewe-French Training with Super Corpus Ratio 1:4")

    train_df, val_df, test_df = create_datasets()
    train_model(train_df, val_df, test_df)














