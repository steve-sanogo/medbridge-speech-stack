import pandas as pd
import re
from pathlib import Path

# Update to your master Fongbe corpus path
FILE_PATH = "/home/data/projets-aps/projet6/data_preprocessing/fongbe/fongbe_corpus.csv"

def get_stats(file_path):
    print(f"📊 Analyzing Fongbe Corpus: {file_path}\n" + "-"*40)

    try:
        df = pd.read_csv(file_path, sep=',', encoding='utf-8', low_memory=False)
    except Exception as e:
        print(f"❌ Initial read failed: {e}")
        return

    # Basic Counts
    total_raw = len(df)

    # 1. Cleaning for analysis
    # Ensure strings are valid for uniqueness checks
    df['text_fongbe'] = df['text_fongbe'].fillna("").astype(str).str.strip()
    df['text_french'] = df['text_french'].fillna("").astype(str).str.strip()

    # 2. Uniqueness Analysis (The new requested info)
    unique_fongbe = df['text_fongbe'].nunique()
    unique_french = df['text_french'].nunique()
    
    # Unique Pairs (Fongbe + French combined)
    # This counts how many distinct translation examples exist
    unique_pairs = len(df.drop_duplicates(subset=['text_fongbe', 'text_french']))

    # 3. Length Analysis (Word count based on Fongbe)
    def count_words(text):
        if not text: return 0
        return len(text.split())

    df['word_count_fon'] = df['text_fongbe'].apply(count_words)
    avg_len = df[df['word_count_fon'] > 0]['word_count_fon'].mean()
    max_len = df['word_count_fon'].max()

    # 4. Content Categorization
    short_fragments = len(df[(df['word_count_fon'] > 0) & (df['word_count_fon'] <= 3)])
    robust_sentences = len(df[df['word_count_fon'] >= 10])

    # 5. Vocabulary & Character Diversity
    sample_size = min(20000, len(df))
    all_text = " ".join(df['text_fongbe'].sample(sample_size).tolist())
    special_chars = set(re.findall(r'[ɖɔɛɩʋŋ]', all_text.lower()))

    # 6. Parallel Data Check
    parallel_rows = len(df[(df['text_french'] != "") & (df['text_french'].str.len() > 2)])

    # OUTPUT
    print(f"Total Rows in File:      {total_raw:,}")
    print(f"Parallel (FR matched):   {parallel_rows:,}")
    print("-" * 40)
    print(f"Unique Fongbe only:      {unique_fongbe:,}")
    print(f"Unique French only:      {unique_french:,}")
    print(f"Unique Pairs (FON+FR):   {unique_pairs:,}")
    print("-" * 40)
    print(f"Average Words (Fon):     {avg_len:.1f}")
    print(f"Max Words (Fon):         {max_len}")
    print(f"Short Frags (1-3 wds):   {short_fragments:,}")
    print(f"Robust Lines (10+ wds):  {robust_sentences:,} <--- THE GOLD")
    print("-" * 40)
    print(f"Fongbe Characters Found: {', '.join(sorted(special_chars))}")
    print("-" * 40)

    print("Dataset Composition:")
    print(df['database_origin'].value_counts())

    print("-" * 40)
    print("Top 3 Potential Noise (Frequent Strings):")
    top_duplicates = df[df['text_fongbe'] != ""]['text_fongbe'].value_counts().head(3)
    for txt, count in top_duplicates.items():
        print(f" [{count:,}x] \"{txt[:70]}...\"")

if __name__ == "__main__":
    if Path(FILE_PATH).exists():
        get_stats(FILE_PATH)
    else:
        print(f"❌ File not found: {FILE_PATH}")