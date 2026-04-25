import pandas as pd
from pathlib import Path

BASE = Path("/home/data/projets-aps/projet6")
OUT_DIR = BASE / "data_preprocessing" / "ewe" / "standardized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def process_parquet_folder(input_dir, origin_label):
    input_path = Path(input_dir)
    all_dfs = []
    
    files = list(input_path.glob("*.parquet"))
    if not files:
        print(f"⚠️ No files found in {input_dir}")
        return

    print(f"Processing {origin_label}...")
    for f in files:
        df = pd.read_parquet(f)
        # Identify text column
        target_col = next((c for c in ["transcription", "text"] if c in df.columns), None)
        
        if target_col:
            temp_df = pd.DataFrame({
                "database_origin": origin_label,
                "id": df.index.map(lambda x: f"{f.stem}_{x}"), # Unique ID: Filename + Index
                "text_ewe": df[target_col].astype(str),
                "text_french": "" # Placeholder
            })
            all_dfs.append(temp_df)

    final_df = pd.concat(all_dfs, ignore_index=True)
    out_file = OUT_DIR / f"{origin_label}_standardized.csv"
    final_df.to_csv(out_file, index=False, encoding="utf-8")
    print(f"✅ Saved {len(final_df)} rows to {out_file}")

# Execution
process_parquet_folder(BASE / "data" / "data_ewe" / "ewe_asr", "ewe_asr")
process_parquet_folder(BASE / "data" / "data_ewe" / "ewe_tts", "ewe_tts")
