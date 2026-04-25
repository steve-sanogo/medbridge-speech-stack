import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = Path("/home/data/projets-aps/projet6")
DATA_DIR = BASE / "data" / "data_ewe"
OUT_DIR  = BASE / "data_preprocessing" / "ewe" / "standardized"

# 🛠️ Ensure the output directory exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

def standardize_parallel(file_path, origin_label, ewe_col, fr_col, split_name=""):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Skipping {origin_label} {split_name}, file not found at {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
        
        # We append the split_name to the origin label if it exists
        # e.g., 'MAFAND_MT_train'
        full_origin = f"{origin_label}_{split_name}" if split_name else origin_label
        
        df_std = pd.DataFrame({
            "database_origin": full_origin,
            "id": range(len(df)), # Simple integer ID
            "text_ewe": df[ewe_col].astype(str).str.strip(),
            "text_french": df[fr_col].astype(str).str.strip()
        })
        
        # File naming logic
        file_suffix = f"_{split_name}" if split_name else ""
        out_path = OUT_DIR / f"{origin_label}{file_suffix}_standardized.csv"
        
        df_std.to_csv(out_path, index=False, encoding="utf-8")
        print(f"✅ Saved {len(df_std)} rows for {full_origin}")
        
    except Exception as e:
        print(f"❌ Error processing {file_path.name}: {e}")

# ── Execution ──────────────────────────────────────────────────────────────────

# 1. AI4D
standardize_parallel(
    DATA_DIR / "AI4D_MT" / "ai4d_full_aligned_fr_ee.csv", 
    "AI4D_MT", 
    "ewe", 
    "french"
)

# 2. MAFAND (Origin will be 'MAFAND_MT_test', 'MAFAND_MT_train', etc.)
mafand_base = DATA_DIR / "MAFAND"
standardize_parallel(mafand_base / "mafand_test_fr_ee.csv", "MAFAND_MT", "ewe", "french", "test")
standardize_parallel(mafand_base / "mafand_train_fr_ee.csv", "MAFAND_MT", "ewe", "french", "train")
standardize_parallel(mafand_base / "mafand_validation_fr_ee.csv", "MAFAND_MT", "ewe", "french", "validation")