import pandas as pd
from pathlib import Path

BASE = Path("/home/data/projets-aps/projet6")
DATA_DIR = BASE / "data" / "data_ewe"
OUT_DIR = BASE / "data_preprocessing" / "ewe" / "standardized"

def standardize_parallel(file_path, origin_label, ewe_col, fr_col):
    if not Path(file_path).exists():
        print(f"Skipping {origin_label}, file not found.")
        return

    df = pd.read_csv(file_path)
    
    df_std = pd.DataFrame({
        "database_origin": origin_label,
        "id": [f"{origin_label}_{i}" for i in range(len(df))],
        "text_ewe": df[ewe_col].astype(str),
        "text_french": df[fr_col].astype(str)
    })
    
    out_path = OUT_DIR / f"{origin_label}_standardized.csv"
    df_std.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Saved {len(df_std)} parallel rows for {origin_label}")

# Run for AI4D and MAFAND
standardize_parallel(DATA_DIR / "AI4D_MT" / "ai4d_full_aligned_fr_ee.csv", "AI4D_MT", "ewe", "french")
# Adjust MAFAND filename if needed based on your previous harvest
standardize_parallel(DATA_DIR / "MAFAND" / "mafand_test_fr_ee.csv", "MAFAND_MT", "ewe", "french")
standardize_parallel(DATA_DIR / "MAFAND" / "mafand_train_fr_ee.csv", "MAFAND_MT", "ewe", "french")
standardize_parallel(DATA_DIR / "MAFAND" / "mafand_validation_fr_ee.csv", "MAFAND_MT", "ewe", "french")
