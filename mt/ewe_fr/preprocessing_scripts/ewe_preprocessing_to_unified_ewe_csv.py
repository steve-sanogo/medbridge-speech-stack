import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = Path("/home/data/projets-aps/projet6")
ASR_DIR  = BASE / "data" / "data_ewe" / "ewe_asr"
TTS_DIR  = BASE / "data" / "data_ewe" / "ewe_tts"
XLS_FILE = BASE / "data" / "data_ewe" / "speech_ug" / "Ewe" / "selected transcribed audios" / "selected transcribed audios.xlsx"

OUT_DIR  = BASE / "data_preprocessing" / "ewe"
OUT_FILE = OUT_DIR / "raw_extracted_ewe_text_with_origin.csv"

def extract_from_parquet(directory, label):
    data_list = []
    files = list(directory.glob("*.parquet"))
    if not files:
        return []
        
    print(f"Reading {len(files)} files in {directory.name}...")
    
    for f in files:
        try:
            df = pd.read_parquet(f)
            target_col = next((c for c in ["transcription", "text"] if c in df.columns), None)
            
            if target_col:
                # Create a temporary dataframe to hold text and origin
                temp_df = pd.DataFrame({
                    "raw_text": df[target_col].astype(str),
                    "dataset_origin": label
                })
                data_list.append(temp_df)
                
        except Exception as e:
            print(f"  ❌ Error in {f.name}: {e}")
            
    return pd.concat(data_list, ignore_index=True) if data_list else pd.DataFrame()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Collect from Parquet folders
    df_asr = extract_from_parquet(ASR_DIR, "ewe_asr")
    df_tts = extract_from_parquet(TTS_DIR, "ewe_tts")

    # 2. Add Excel data
    df_xls = pd.DataFrame()
    if XLS_FILE.exists():
        print(f"Adding Excel: {XLS_FILE.name}")
        xls_raw = pd.read_excel(XLS_FILE)
        possible = ["transcription", "text", "sentence"]
        col = next((c for c in possible if c in xls_raw.columns), xls_raw.columns[0])
        
        df_xls = pd.DataFrame({
            "raw_text": xls_raw[col].astype(str),
            "dataset_origin": "speech_ug_excel"
        })

    # 3. Combine and Save
    df_final = pd.concat([df_asr, df_tts, df_xls], ignore_index=True)
    
    # Basic cleaning of extra whitespace/newlines
    df_final["raw_text"] = df_final["raw_text"].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    print(f"\nSaving {len(df_final)} rows to {OUT_FILE}...")
    df_final.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print("✅ Local extraction done.")

if __name__ == "__main__":
    main()