import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
INPUT_FILE = Path("/home/data/projets-aps/projet6/data/data_fongbe/FFR_v1/ffr_v1_with_diacritics.csv")
DEST_DIR = Path("/home/data/projets-aps/projet6/data_preprocessing/fongbe/standardized")
OUTPUT_FILE = DEST_DIR / "ffr_v1_with_diacritics_standardized.csv"

def preprocess():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"⏳ Processing FFR_v1: {INPUT_FILE.name}")
    
    try:
        # FFR uses lowercase 'fongbe' and 'french'
        df = pd.read_csv(INPUT_FILE)
        
        std_df = pd.DataFrame()
        std_df['database_origin'] = ["FFR_v1"] * len(df)
        std_df['id'] = [f"FFR_{i}" for i in range(len(df))]
        std_df['text_fongbe'] = df['fongbe'].astype(str).str.strip()
        std_df['text_french'] = df['french'].astype(str).str.strip()
        
        std_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Saved {len(std_df)} rows to {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    preprocess()
