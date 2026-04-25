import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
INPUT_FILE = Path("/home/data/projets-aps/projet6/data/data_fongbe/AI4D_Fongbe_Ewe/AI4D_MT_Fongbe.csv")
DEST_DIR = Path("/home/data/projets-aps/projet6/data_preprocessing/fongbe/standardized")
OUTPUT_FILE = DEST_DIR / "AI4D_MT_Fongbe_standardized.csv"

def preprocess():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"⏳ Processing AI4D: {INPUT_FILE.name}")
    
    try:
        df = pd.read_csv(INPUT_FILE)
        
        std_df = pd.DataFrame()
        std_df['database_origin'] = ["AI4D"] * len(df)
        std_df['id'] = [f"AI4D_{i}" for i in range(len(df))]
        std_df['text_fongbe'] = df['Fon'].astype(str).str.strip()
        std_df['text_french'] = df['French'].astype(str).str.strip()
        
        std_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Saved {len(std_df)} rows to {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    preprocess()
