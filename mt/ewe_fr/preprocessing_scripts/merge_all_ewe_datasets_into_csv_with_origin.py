import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_BASE = Path("/home/data/projets-aps/projet6/data/data_ewe")
LOCAL_CSV = Path("/home/data/projets-aps/projet6/data_preprocessing/ewe/raw_extracted_ewe_text_with_origin.csv")

# Parallel Data Sources
AI4D_PATH   = DATA_BASE / "AI4D_MT" / "ai4d_full_aligned_fr_ee.csv"
MAFAND_PATH = DATA_BASE / "MAFAND" / "mafand_train_fr_ee.csv" # Using Train split
SENT_PATH   = DATA_BASE / "Ewe_sentiment_corpus" / "ewe_sentiment_corpus.txt"

FINAL_CORPUS_FILE = Path("/home/data/projets-aps/projet6/data_preprocessing/ewe/ewe_master_corpus.csv")

def main():
    all_dfs = []

    # 1. Load Local Processed Data (Script 1 output)
    if LOCAL_CSV.exists():
        print(f"Adding Local ASR/TTS data...")
        df_local = pd.read_csv(LOCAL_CSV)
        df_local = df_local.rename(columns={"raw_text": "ewe"})
        df_local["french"] = "" # No French match
        all_dfs.append(df_local)

    # 2. Load AI4D (Parallel)
    if AI4D_PATH.exists():
        print("Adding AI4D (Parallel)...")
        df_ai4d = pd.read_csv(AI4D_PATH)
        # Ensure column names match [french, ewe]
        if 'french' not in df_ai4d.columns: # Sometimes AI4D uses 'fr' / 'ee'
            df_ai4d = df_ai4d.rename(columns={'fr': 'french', 'ee': 'ewe'})
        df_ai4d["dataset_origin"] = "AI4D_MT"
        all_dfs.append(df_ai4d[["ewe", "french", "dataset_origin"]])

    # 3. Load MAFAND (Parallel)
    if MAFAND_PATH.exists():
        print("Adding MAFAND (Parallel)...")
        df_mafand = pd.read_csv(MAFAND_PATH)
        df_mafand["dataset_origin"] = "MAFAND_MT"
        all_dfs.append(df_mafand[["ewe", "french", "dataset_origin"]])

    # 4. Load Sentiment (Monolingual)
    if SENT_PATH.exists():
        print("Adding Sentiment Corpus (Monolingual)...")
        with open(SENT_PATH, "r", encoding="utf-8") as f:
            sent_lines = [line.strip() for line in f if line.strip()]
        df_sent = pd.DataFrame({
            "ewe": sent_lines,
            "french": "",
            "dataset_origin": "Ewe_Sentiment"
        })
        all_dfs.append(df_sent)

    # ── Final Merge ────────────────────────────────────────────────────────────
    print("\nMerging everything into Master Corpus...")
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # Basic cleanup: remove rows where 'ewe' is empty
    master_df = master_df[master_df["ewe"].notna() & (master_df["ewe"] != "")]
    
    # Save as CSV
    master_df.to_csv(FINAL_CORPUS_FILE, index=False, encoding="utf-8")
    
    print("-" * 30)
    print(f"📊 MASTER CORPUS SUMMARY:")
    print(master_df["dataset_origin"].value_counts())
    print("-" * 30)
    print(f"✅ Master file saved to: {FINAL_CORPUS_FILE}")

if __name__ == "__main__":
    main()
