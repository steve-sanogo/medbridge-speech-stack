import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── CONFIGURATION ──────────────────────────────────────────────────────────
BASE_DIR = Path("/home/data/projets-aps/projet6")
INPUT_CSV = BASE_DIR / "data_preprocessing/ewe/ewe_corpus.csv"
SAVE_DIR_SPLIT = BASE_DIR / "data_preprocessing/ewe/splitting_baseline"
SEED = 42

files = {
    "train": SAVE_DIR_SPLIT / "train_baseline.csv",
    "val": SAVE_DIR_SPLIT / "val_baseline.csv",
    "test": SAVE_DIR_SPLIT / "test_baseline.csv"
}

# ── 1. CREATION DU DATASET ─────────────────────────────────────────────────
def create_datasets(debug=False):
    print(f"📊 Creating Datasets (Debug Mode: {debug})...")
    
    # Chargement initial
    df = pd.read_csv(INPUT_CSV, low_memory=False).dropna(subset=['text_ewe'])
    
    # Filtrage organique (exactement ta logique)
    is_parallel = df['text_french'].notna() & (df['text_french'].str.len() > 2)
    df_sl = df[is_parallel].copy()

    # --- ÉTAPE CRUCIALE : Suppression des doublons de paires (Ewe + FR) ---
    # On fait ça AVANT le split pour éviter que le modèle apprenne par coeur
    initial_count = len(df_sl)
    df_sl = df_sl.drop_duplicates(subset=['text_ewe', 'text_french'])
    print(f"🧹 Doublons supprimés du corpus source : {initial_count - len(df_sl)}")
  
    # Split SL into Train, Val, Test (90/5/5)
    train_df, temp_df = train_test_split(df_sl, test_size=0.10, random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED)
    
    # --- LEAK FIX : Sécurité sur les textes sources ---
    val_sources = set(val_df['text_ewe'].unique())
    test_sources = set(test_df['text_ewe'].unique())
    forbidden_in_train = val_sources.union(test_sources)

    train_hashes = set(train_df['text_ewe'].unique())
    overlap = forbidden_in_train.intersection(train_hashes)
    
    if len(overlap) > 0:
        print(f"⚠️  Fixing overlap: Removing {len(overlap)} source rows from Training found in Val/Test.")
        train_df = train_df[~train_df['text_ewe'].isin(overlap)]
  
    # Sauvegarde physique
    if SAVE_DIR_SPLIT.exists():
        shutil.rmtree(SAVE_DIR_SPLIT)
    SAVE_DIR_SPLIT.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(files["train"], index=False)
    val_df.to_csv(files["val"], index=False)
    test_df.to_csv(files["test"], index=False)
    
    print(f"✅ Data Locked & Saved: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

# ── 2. INSPECTION DU DATASET ───────────────────────────────────────────────
def inspect_datasets():
    print(f"\n🔍 Début de l'inspection (Analyse des Paires Bilingues) dans : {SAVE_DIR_SPLIT}\n")
    
    dfs = {}
    for name, path in files.items():
        if not path.exists():
            print(f"❌ Erreur : Le fichier {name} est introuvable !")
            return
        dfs[name] = pd.read_csv(path)
        print(f"✅ {name.capitalize()} chargé : {len(dfs[name])} lignes.")

    print("\n--- Vérification des doublons de paires (Ewe + Français) ---")
    for name, df in dfs.items():
        duplicates = df.duplicated(subset=['text_ewe', 'text_french']).sum()
        print(f"🔸 Paires identiques INTERNES à {name} : {duplicates}")

    print("\n--- Vérification du Data Leakage (Intersections de paires) ---")
    def get_pair_set(df):
        return set(zip(df['text_ewe'], df['text_french']))

    train_pairs = get_pair_set(dfs['train'])
    val_pairs = get_pair_set(dfs['val'])
    test_pairs = get_pair_set(dfs['test'])

    print(f"❌ Fuite réelle (Paires) TRAIN ∩ VAL  : {len(train_pairs.intersection(val_pairs))}")
    print(f"❌ Fuite réelle (Paires) TRAIN ∩ TEST : {len(train_pairs.intersection(test_pairs))}")
    print(f"⚠️  Chevauchement VAL ∩ TEST          : {len(val_pairs.intersection(test_pairs))}")

    print("\n--- Analyse de la diversité (Polysémie) ---")
    for name, df in dfs.items():
        ewe_counts = df.groupby('text_ewe')['text_french'].nunique()
        polysemy = (ewe_counts > 1).sum()
        print(f"🔹 {name} : {polysemy} phrases Éwé ont plus d'une traduction française.")

# ── 3. EXECUTION ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Change 'debug=True' pour un test rapide, 'debug=False' pour le run final d'une heure
    create_datasets() 
    inspect_datasets()