import torch
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sacrebleu

# ── CONFIG ──────────────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16  # Significant speedup
BEAMS = 2        # Faster than 4, still better than greedy
BASE_DIR = Path("/home/data/projets-aps/projet6")
ORGANIC_CSV = BASE_DIR / "data_preprocessing/ewe/ewe_corpus.csv"

MODELS_TO_TEST = {
    "NLLB_Base": "facebook/nllb-200-1.3B",
    "FineTuned_Organic_Only": BASE_DIR / "experiments/translation_ewe_french/final_model",
    "FineTuned_Ratio_1_2": BASE_DIR / "experiments/translation_ewe_french_super_corpus_ratio_1_to_2/final_model",
    "FineTuned_Ratio_1_4": BASE_DIR / "experiments/translation_ewe_french_super_corpus_ratio_1_to_4/final_model"
}

# ── DATA PREPARATION ────────────────────────────────────────────────────────
def get_locked_test_data():
    df = pd.read_csv(ORGANIC_CSV, low_memory=False).dropna(subset=['text_ewe', 'text_french'])
    is_parallel = (df['text_french'].str.len() > 2)
    df_sl = df[is_parallel].copy()
    
    # Strictly replicating your original split
    _, temp_df = train_test_split(df_sl, test_size=0.10, random_state=SEED)
    _, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED)
    return test_df

# ── BATCHED EVALUATION FUNCTION ─────────────────────────────────────────────
def evaluate_model_batched(model_path, data_df, name):
    print(f"\n🚀 Evaluating: {name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.src_lang = "ewe_Latn"
    tokenizer.tgt_lang = "fra_Latn"
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, 
        use_safetensors=True
    ).to(DEVICE)
    model.eval()

    inputs = data_df['text_ewe'].astype(str).tolist()
    references = data_df['text_french'].astype(str).tolist()
    predictions = []

    forced_token_id = tokenizer.convert_tokens_to_ids("fra_Latn")

    # Process in batches
    for i in tqdm(range(0, len(inputs), BATCH_SIZE), desc=f"Batching {name}"):
        batch_texts = inputs[i : i + BATCH_SIZE]
        
        # Padding=True and truncation=True are vital for batching
        inputs_tokenized = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(DEVICE)
        
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs_tokenized,
                forced_bos_token_id=forced_token_id,
                max_length=128,
                num_beams=BEAMS
            )
        
        decoded_batch = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        predictions.extend([d.strip() for d in decoded_batch])

    # Calculate Metrics
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    chrf = sacrebleu.corpus_chrf(predictions, [references]).score
    
    return {"bleu": bleu, "chrf": chrf}

# ── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_df = get_locked_test_data()
    print(f"✅ Data Locked: Test={len(test_df)} rows")

    final_comparison = {}

    for name, path in MODELS_TO_TEST.items():
        try:
            results = evaluate_model_batched(str(path), test_df, name)
            final_comparison[name] = results
            print(f"✅ {name}: BLEU {results['bleu']:.2f} | chrF {results['chrf']:.2f}")
        except Exception as e:
            print(f"❌ Failed to evaluate {name}: {e}")

    # Output to JSON
    with open("final_batched_comparison_report.json", "w") as f:
        json.dump(final_comparison, f, indent=4)
    
    print("\n🏁 Batched Benchmark complete.")