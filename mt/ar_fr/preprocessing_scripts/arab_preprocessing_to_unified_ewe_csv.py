import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/data/projets-aps/projet6/data_preprocessing/data_arab")
INPUT_DIR = BASE_DIR / "standardized_with_french"
OUT_FILE  = BASE_DIR / "unified_arabic_master_corpus.csv"

def main():
    all_data = []
    
    # 1. Grab all CSVs we just translated
    files = list(INPUT_DIR.glob("*_with_french.csv"))
    print(f"📂 Found {len(files)} translated files.")

    for f in files:
        print(f"📖 Reading: {f.name}...")
        try:
            df = pd.read_csv(f)
            # DROPPED: 'text_english'
            cols = ['database_origin', 'text_arab_egypt', 'text_french']
            
            # Check if columns exist before copying to avoid errors
            existing_cols = [c for c in cols if c in df.columns]
            df = df[existing_cols].copy()
            
            all_data.append(df)
        except Exception as e:
            print(f"  ❌ Error processing {f.name}: {e}")

    # 2. Combine
    if not all_data:
        print("❌ No data found to merge!")
        return

    master_df = pd.concat(all_data, ignore_index=True)

    # 3. Clean Whitespace
    # Only processing the two language columns now
    for col in ['text_arab_egypt', 'text_french']:
        if col in master_df.columns:
            master_df[col] = master_df[col].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

    # 4. Save
    print(f"\n💾 Saving {len(master_df):,} rows to {OUT_FILE}...")
    master_df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print("✅ Arabic-French Master Corpus created.")

if __name__ == "__main__":
    main()