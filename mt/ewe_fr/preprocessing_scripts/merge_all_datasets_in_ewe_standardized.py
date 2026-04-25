import pandas as pd
from pathlib import Path
import json
import string

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = Path("/home/data/projets-aps/projet6")
STD_DIR  = BASE / "data_preprocessing" / "ewe" / "standardized"
OUT_DIR  = BASE / "data_preprocessing" / "ewe"
OUT_FILE = OUT_DIR / "ewe_corpus.csv"
MAP_FILE = OUT_DIR / "dataset_mapping.json"

def main():
    if not STD_DIR.exists():
        print(f"❌ Directory not found: {STD_DIR}")
        return

    # 1. Identify all CSVs in the standardized folder
    csv_files = sorted(list(STD_DIR.glob("*_standardized.csv")))
    if not csv_files:
        print("⚠️ No standardized CSVs found to merge.")
        return

    # 2. Create the mapping (a, b, c...)
    # We use string.ascii_lowercase to get 'abcdef...'
    mapping = {}
    alphabet = string.ascii_lowercase
    
    all_dfs = []
    
    print(f"🔗 Starting merge of {len(csv_files)} datasets...")

    for i, file in enumerate(csv_files):
        # Assign a prefix character
        prefix = alphabet[i] if i < len(alphabet) else f"z{i}" # Fallback if > 26 files
        
        # Read the CSV
        df = pd.read_csv(file)
        
        # Get the origin name from the first row or filename
        origin_name = df["database_origin"].iloc[0] if not df.empty else file.stem
        mapping[prefix] = origin_name
        
        print(f"  [{prefix}] Processing {origin_name} ({len(df)} rows)")

        # 3. Update the ID with the prefix
        # We ensure ID is string then prepend prefix + underscore
        df["id"] = prefix + "_" + df["id"].astype(str)
        
        all_dfs.append(df)

    # 4. Concatenate all data
    final_corpus = pd.concat(all_dfs, ignore_index=True)

    # 5. Save the Master Corpus
    final_corpus.to_csv(OUT_FILE, index=False, encoding="utf-8")
    
    # 6. Save the Mapping JSON
    with open(MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4)

    print("\n✨ Merge Complete!")
    print(f"📍 Master Corpus: {OUT_FILE} ({len(final_corpus):,} total rows)")
    print(f"📍 Mapping Key: {MAP_FILE}")
    print("-" * 30)
    for char, name in mapping.items():
        print(f"  {char} -> {name}")

if __name__ == "__main__":
    main()
