import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- CONFIG ---
SEED = 42
BASE_DIR = Path("/home/data/projets-aps/projet6")

# Input Paths
PATH_ORGANIC = BASE_DIR / "data_preprocessing/ewe/ewe_corpus.csv"
PATH_SYNTHETIC = BASE_DIR / "data_preprocessing/ewe/standardized/ewe_english/english_ewe_standardized_ratio_1_4.csv"

# ── DATASET CREATION ────────────────────────────────────────────────────────
def create_datasets():
    print(f"📊 Loading Organic: {PATH_ORGANIC.name}")
    
    # --- YOUR EXACT LINES ---
    df = pd.read_csv(PATH_ORGANIC, low_memory=False).dropna(subset=['text_ewe'])
    is_parallel = df['text_french'].notna() & (df['text_french'].str.len() > 2)
    # ------------------------
    
    df_organic = df[is_parallel].copy()

    # Split Organic into Train/Val/Test (90/5/5)
    # This ensures Val and Test are 100% Organic "Gold" data
    train_org_base, temp_org = train_test_split(df_organic, test_size=0.10, random_state=SEED)
    val_df, test_df = train_test_split(temp_org, test_size=0.50, random_state=SEED)

    print(f"🤖 Loading Synthetic: {PATH_SYNTHETIC.name}")
    df_synthetic = pd.read_csv(PATH_SYNTHETIC, low_memory=False).dropna(subset=['text_ewe', 'text_french'])
    
    # ── THE CHAINING STEP ──
    # We append the synthetic data ONLY to the organic training part
    train_df = pd.concat([train_org_base, df_synthetic], ignore_index=True)
    
    # Shuffle the combined training set
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # ── CONSISTENCY ASSERTIONS & CLEANUP ──
    print("\n🔍 --- SPLIT CONSISTENCY CHECK ---")
    val_hashes = set(val_df['text_ewe'].unique())
    train_hashes = set(train_df['text_ewe'].unique())
    overlap = val_hashes.intersection(train_hashes)
    
    if len(overlap) > 0:
        print(f"⚠️  Fixing overlap: Removing {len(overlap)} rows from Training found in Validation.")
        train_df = train_df[~train_df['text_ewe'].isin(overlap)]

    # Verification of final counts
    assert len(val_df) == 1433, f"Validation mismatch: {len(val_df)}"
    assert len(test_df) == 1434, f"Test mismatch: {len(test_df)}"
    
    print(f"✅ Validation (1433) and Test (1434) are locked.")
    print(f"📈 Total Train Size: {len(train_df):,} (Organic + Synthetic)")

    return train_df, val_df, test_df

# --- EXECUTION ---

# 1. Generate Logic 1 (Baseline)
train_v1, val_v1, test_v1 = create_datasets()

# 2. Generate Logic 2 (Super Corpus)
# We start with the SAME split as V1
train_v2_base, val_v2, test_v2 = create_datasets()

# Now we append the synthetic data ONLY to the training set
df_synthetic = pd.read_csv(PATH_SYNTHETIC, low_memory=False)
# Ensure columns match (adjust names if necessary)
# Assuming PATH_SYNTHETIC has 'text_ewe' and 'text_french'
train_v2 = pd.concat([train_v2_base, df_synthetic], ignore_index=True).sample(frac=1, random_state=SEED)

# --- VERIFICATION ---
print("🔍 --- FINAL CONSISTENCY CHECK ---")

def check(df_a, df_b, name):
    match = df_a.sort_values(by='text_ewe').reset_index(drop=True).equals(
            df_b.sort_values(by='text_ewe').reset_index(drop=True))
    status = "✅ PERFECT MATCH" if match else "❌ DISCREPANCY"
    print(f"{name}: {status} ({len(df_a)} rows)")

check(val_v1, val_v2, "Validation Set")
check(test_v1, test_v2, "Test Set")

print(f"\n📈 Training Set Comparison:")
print(f"   V1 Train: {len(train_v1):,} rows")
print(f"   V2 Train: {len(train_v2):,} rows (+{len(df_synthetic):,} synthetic)")

# Check for leakage: Is any sentence in Val/Test also in the NEW Train?
val_texts = set(val_v2['text_ewe'].tolist())
train_texts = set(train_v2['text_ewe'].tolist())
leakage = val_texts.intersection(train_texts)

if not leakage:
    print("\n🏆 SUCCESS: No data leakage detected. Val/Test are strictly unseen.")
else:
    print(f"\n⚠️ WARNING: {len(leakage)} rows from Validation are in the Training set!")