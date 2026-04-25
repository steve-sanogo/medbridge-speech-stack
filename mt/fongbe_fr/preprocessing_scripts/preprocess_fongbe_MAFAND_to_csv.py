import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
INPUT_DIR = Path("/home/data/projets-aps/projet6/data/data_fongbe/MAFAND")
DEST_DIR = Path("/home/data/projets-aps/projet6/data_preprocessing/fongbe/standardized")

def preprocess():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each split (train, dev, test)
    for csv_file in INPUT_DIR.glob("*.csv"):
        print(f"⏳ Processing MAFAND: {csv_file.name}")
        try:
            # MAFAND uses 'fon' and 'fr'
            df = pd.read_csv(csv_file)
            
            split_name = csv_file.stem # 'train', 'dev', or 'test'
            std_df = pd.DataFrame()
            std_df['database_origin'] = [f"MAFAND_{split_name}"] * len(df)
            std_df['id'] = [f"MAFAND_{split_name}_{i}" for i in range(len(df))]
            std_df['text_fongbe'] = df['fon'].astype(str).str.strip()
            std_df['text_french'] = df['fr'].astype(str).str.strip()
            
            output_file = DEST_DIR / f"{split_name}_standardized.csv"
            std_df.to_csv(output_file, index=False)
            print(f"✅ Saved {len(std_df)} rows to {output_file}")
        except Exception as e:
            print(f"❌ Error processing {csv_file.name}: {e}")

if __name__ == "__main__":
    preprocess()
