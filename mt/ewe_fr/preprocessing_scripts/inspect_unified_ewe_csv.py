import pandas as pd
import re
from pathlib import Path

# List of all corpora to analyze
CORPORA_PATHS = [
    "/home/data/projets-aps/projet6/data_preprocessing/ewe/ewe_corpus.csv",
    "/home/data/projets-aps/projet6/data_preprocessing/ewe/ewe_super_corpus_ratio_1_to_2.csv",
    "/home/data/projets-aps/projet6/data_preprocessing/ewe/ewe_super_corpus_ratio_1_to_4.csv"
]

def analyze_file(file_path):
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        return None

    print(f"\n📊 Analyzing: {path.name}")
    print("-" * 50)
    
    try:
        df = pd.read_csv(file_path, sep=',', encoding='utf-8', low_memory=False)
    except Exception as e:
        print(f"❌ Read failed: {e}")
        return None

    # 1. Basic Cleaning
    df['clean_ewe'] = df['text_ewe'].fillna("").astype(str).str.strip()
    df['clean_fr'] = df['text_french'].fillna("").astype(str).str.strip()
    
    total_raw = len(df)
    empty_count = sum(df['clean_ewe'] == "")
    unique_count = df['clean_ewe'].nunique()

    # 2. Parallel Check (Both Ewe and French must have content)
    parallel_df = df[(df['clean_ewe'] != "") & (df['clean_fr'].str.len() > 2)]
    parallel_rows = len(parallel_df)

    # 3. Length Analysis
    df['word_count'] = df['clean_ewe'].apply(lambda x: len(x.split()) if x else 0)
    robust_lines = len(df[df['word_count'] >= 10])
    avg_len = df[df['word_count'] > 0]['word_count'].mean()

    # 4. Characters
    sample_text = " ".join(df['clean_ewe'].sample(min(10000, len(df))).tolist())
    special_chars = sorted(set(re.findall(r'[ɖɛƒɣɔʋŋ]', sample_text)))

    # PRINT INDIVIDUAL STATS
    print(f"Rows: {total_raw:,} | Parallel: {parallel_rows:,} | Unique: {unique_count:,}")
    print(f"Robust (10+ words): {robust_lines:,} <--- THE GOLD")
    print(f"Avg Word Count: {avg_len:.1f}")
    print(f"Special Chars: {', '.join(special_chars)}")
    
    print("\nDataset Composition:")
    print(df['database_origin'].value_counts().head(5)) # Top 5 origins

    # Return summary for the final comparison table
    return {
        "File": path.name,
        "Total": total_raw,
        "Parallel": parallel_rows,
        "Robust": robust_lines,
        "Unique": unique_count
    }

def main():
    results = []
    for path in CORPORA_PATHS:
        res = analyze_file(path)
        if res:
            results.append(res)
    
    if results:
        print("\n" + "="*60)
        print("Final Comparison Table".center(60))
        print("="*60)
        comparison_df = pd.DataFrame(results)
        print(comparison_df.to_string(index=False))
        print("="*60)

if __name__ == "__main__":
    main()