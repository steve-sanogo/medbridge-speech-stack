import pandas as pd
import re
from pathlib import Path

FILE_PATH = "/home/data/projets-aps/projet6/data_preprocessing/data_arab/unified_arabic_master_corpus.csv"

def get_stats(file_path):
    print(f"📊 Analyzing Arabic Corpus: {file_path}\n" + "-"*50)

    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"❌ Read failed: {e}")
        return

    # --- 1. Deduplication Analysis ---
    total_rows = len(df)
    # Check for exact duplicate pairs (AR + FR are identical)
    exact_duplicates = df.duplicated(subset=['text_arab_egypt', 'text_french']).sum()
    
    # Check for Arabic duplicates (Same source, potentially different translations)
    arab_duplicates = df.duplicated(subset=['text_arab_egypt']).sum()
    
    # Check for French duplicates (Different sources, same target)
    french_duplicates = df.duplicated(subset=['text_french']).sum()

    # --- 2. Length Analysis ---
    df['word_count'] = df['text_arab_egypt'].fillna("").apply(lambda x: len(str(x).split()))
    avg_len = df['word_count'].mean()
    robust_lines = len(df[df['word_count'] >= 10])

    # --- 3. Script Diversity Check ---
    def has_latin(text):
        return bool(re.search(r'[a-zA-Z]', str(text)))
    mixed_script_count = df['text_arab_egypt'].apply(has_latin).sum()

    # --- 4. Trilingual Check ---
    complete_rows = df.dropna(subset=['text_arab_egypt','text_french']).shape[0]

    # OUTPUT
    print(f"Total Bilingual Rows:     {total_rows:,}")
    print(f"Complete (A_eg+FR):       {complete_rows:,}")
    print("-" * 50)
    print(f"🔍 DUPLICATION CHECK:")
    print(f"Exact Bilingual Dups:    {exact_duplicates:,}")
    print(f"Arabic Text Dups:         {arab_duplicates:,}")
    print(f"French Text Dups:         {french_duplicates:,}")
    if exact_duplicates > 0:
        print(f"⚠️  WARNING: {exact_duplicates} rows are redundant.")
    print("-" * 50)
    print(f"Average Words/Sentence:   {avg_len:.1f}")
    print(f"Robust Lines (10+ wds):   {robust_lines:,} <--- High Semantic Value")
    print(f"Mixed Script (AR+Latin):  {mixed_script_count:,} (e.g., technical terms)")
    print("-" * 50)
    print("Dataset Composition:")
    print(df['database_origin'].value_counts())
    print("-" * 50)
    
    print("Common Egyptian Particles Found:")
    particles = ['عشان', 'اللي', 'زي', 'ده', 'دي', 'مش']
    for p in particles:
        # Note: Using .fillna('') to avoid errors on NaN rows
        count = df['text_arab_egypt'].fillna('').str.contains(rf'\b{p}\b', regex=True).sum()
        print(f" - {p}: found in {count:,} rows")

if __name__ == "__main__":
    if Path(FILE_PATH).exists():
        get_stats(FILE_PATH)
    else:
        print(f"❌ File not found: {FILE_PATH}")