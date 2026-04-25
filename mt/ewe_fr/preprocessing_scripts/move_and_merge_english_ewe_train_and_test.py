import pandas as pd
import os
from pathlib import Path

# --- CONFIGURATION ---
SOURCE_DIR = Path("/home/data/projets-aps/projet6/data/data_ewe/ewe_anglais/English_ewe")
OUTPUT_DIR = Path("/home/data/projets-aps/projet6/data_preprocessing/ewe/standardized/ewe_english")
GOLD_COUNT = 28664  # Based on your inspection of high-quality organic pairs

def process_and_balance():
    # 1. Define file paths
    train_path = SOURCE_DIR / "train_ewe_french.csv"
    test_path = SOURCE_DIR / "test_ewe_french.csv"
    
    if not train_path.exists() or not test_path.exists():
        print(f"❌ Error: Files missing in {SOURCE_DIR}")
        return

    print("📖 Loading datasets...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    merged_df = pd.concat([df_train, df_test], ignore_index=True)

    # 2. STRICT CLEANING
    initial_len = len(merged_df)
    
    # Drop rows with any NaN/Empty values
    merged_df = merged_df.dropna(subset=['text_ewe', 'text_french'])
    
    # Drop exact bilingual duplicates
    merged_df = merged_df.drop_duplicates(subset=['text_ewe', 'text_french'])
    
    # Optional: Drop rows where Ewe is identical to French (usually noise)
    merged_df = merged_df[merged_df['text_ewe'].str.strip() != merged_df['text_french'].str.strip()]
    
    print(f"🧹 Cleaned: {initial_len} -> {len(merged_df)} rows (Removed {initial_len - len(merged_df)} bad/duplicate rows)")

    # 3. SCORE BY GOODNESS (Word Count)
    # We assume longer sentences provide richer contextual information for NLLB
    merged_df['word_count'] = merged_df['text_ewe'].astype(str).apply(lambda x: len(x.split()))
    merged_df = merged_df.sort_values(by='word_count', ascending=False)

    # 4. PREPARE STANDARDIZED COLUMNS
    merged_df['database_origin'] = "english_ewe_translated"
    # Re-index to ensure IDs are fresh after sorting
    merged_df['id'] = ["trans_" + str(i) for i in range(len(merged_df))]

    # 5. GENERATE RATIO-BASED OUTPUTS
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Output A: Ratio 1:2 (Gold * 2)
    count_2x = GOLD_COUNT * 2
    df_2x = merged_df.head(count_2x).drop(columns=['word_count'])
    path_2x = OUTPUT_DIR / "english_ewe_standardized_ratio_1_2.csv"
    df_2x.to_csv(path_2x, index=False)

    # Output B: Ratio 1:4 (Gold * 4)
    count_4x = GOLD_COUNT * 4
    df_4x = merged_df.head(count_4x).drop(columns=['word_count'])
    path_4x = OUTPUT_DIR / "english_ewe_standardized_ratio_1_4.csv"
    df_4x.to_csv(path_4x, index=False)

    print(f"\n🚀 BALANCING COMPLETE")
    print(f"📍 1:2 Ratio File (Gold x 2): {len(df_2x):,} rows -> {path_2x}")
    print(f"📍 1:4 Ratio File (Gold x 4): {len(df_4x):,} rows -> {path_4x}")
    print(f"💡 Strategy: Selected top {count_4x} longest sentences for maximum quality.")

if __name__ == "__main__":
    process_and_balance()