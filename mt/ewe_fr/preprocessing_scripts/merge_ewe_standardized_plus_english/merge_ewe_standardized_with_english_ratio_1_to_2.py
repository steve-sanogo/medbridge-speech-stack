import pandas as pd
from pathlib import Path
import json
import string

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = Path("/home/data/projets-aps/projet6")
STD_DIR  = BASE / "data_preprocessing" / "ewe" / "standardized"
ENG_DIR  = STD_DIR / "ewe_english"
OUT_DIR  = BASE / "data_preprocessing" / "ewe"

# Updated Output Name
OUT_FILE = OUT_DIR / "ewe_super_corpus_ratio_1_to_2.csv"
MAP_FILE = OUT_DIR / "dataset_mapping_1_to_2.json"

def main():
    if not STD_DIR.exists():
        print(f"❌ Directory not found: {STD_DIR}")
        return

    # 1. Gather CSVs from the main standardized folder
    base_csvs = sorted(list(STD_DIR.glob("*_standardized.csv")))
    
    # 2. Specifically target the 1:2 Ratio file in the subfolder
    ratio_file = ENG_DIR / "english_ewe_standardized_ratio_1_2.csv"
    
    all_files = base_csvs
    if ratio_file.exists():
        all_files.append(ratio_file)
    else:
        print(f"⚠️ Warning: Ratio file not found at {ratio_file}")

    if not all_files:
        print("⚠️ No files found to merge.")
        return

    # 3. Create mapping and merge
    mapping = {}
    alphabet = string.ascii_lowercase
    all_dfs = []
    
    print(f"🔗 Starting Super-Merge (Ratio 1:2) of {len(all_files)} datasets...")

    for i, file in enumerate(all_files):
        prefix = alphabet[i] if i < len(alphabet) else f"z{i}"
        
        # Read the CSV
        df = pd.read_csv(file)
        
        # Data Cleaning: Ensure no empty text rows enter the super corpus
        df = df.dropna(subset=['text_ewe', 'text_french'])
        
        origin_name = df["database_origin"].iloc[0] if not df.empty else file.stem
        mapping[prefix] = origin_name
        
        print(f"  [{prefix}] Processing {origin_name} ({len(df):,} rows)")

        # Update the ID with the unique prefix
        df["id"] = prefix + "_" + df["id"].astype(str)
        all_dfs.append(df)

    # 4. Final Concatenation
    final_corpus = pd.concat(all_dfs, ignore_index=True)

    # 5. Save the Super Corpus
    final_corpus.to_csv(OUT_FILE, index=False, encoding="utf-8")
    
    # 6. Save the Mapping JSON
    with open(MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4)

    print("\n✨ Super-Merge Complete!")
    print(f"📍 Super Corpus: {OUT_FILE} ({len(final_corpus):,} total rows)")
    print(f"📍 Mapping Key: {MAP_FILE}")
    print("-" * 30)
    for char, name in mapping.items():
        print(f"  {char} -> {name}")

if __name__ == "__main__":
    main()