import os
import torch
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics.pairwise import cosine_similarity

# ── CONFIGURATION ──────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.85
SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path("/home/data/projets-aps/projet6")
BASELINE_DIR = BASE_DIR / "data_preprocessing/ewe/splitting_baseline"
ENG_EWE_DIR = BASE_DIR / "data_preprocessing/ewe/standardized/ewe_english"
OUTPUT_DIR = BASE_DIR / "data_preprocessing/ewe/baseline_plus_ratios"

# Fichiers sources Anglais-Ewe
RATIO_FILES = {
    "1_2": ENG_EWE_DIR / "english_ewe_standardized_ratio_1_2.csv",
    "1_4": ENG_EWE_DIR / "english_ewe_standardized_ratio_1_4.csv"
}

# ── 1. CHARGEMENT DU MODÈLE DE FILTRAGE ──────────────────────────────────────
print(f"🚀 Loading SBERT filter on {DEVICE}...")
word_embedding_model = models.Transformer(SBERT_MODEL)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
embed_model = SentenceTransformer(modules=[word_embedding_model, pooling_model]).to(DEVICE)

def filter_dataframe_semantically(df, text_col_1, text_col_2):
    """Applique le filtrage sémantique pour éliminer le bruit."""
    if df.empty: return df
    
    t1 = df[text_col_1].astype(str).tolist()
    t2 = df[text_col_2].astype(str).tolist()
    
    emb1 = embed_model.encode(t1, convert_to_tensor=False, show_progress_bar=True)
    emb2 = embed_model.encode(t2, convert_to_tensor=False, show_progress_bar=True)
    
    sims = np.array([cosine_similarity(e1.reshape(1,-1), e2.reshape(1,-1))[0][0] 
                     for e1, e2 in zip(emb1, emb2)])
    
    df_clean = df[sims < SIMILARITY_THRESHOLD].copy()
    print(f"🧹 Semantic Filter: {len(df) - len(df_clean)} pairs removed (Sim > {SIMILARITY_THRESHOLD}).")
    return df_clean

def create_augmented_test_sets():
    # A. Charger les bases pour identifier les phrases interdites
    print("📂 Loading baseline sets for cross-leakage check...")
    train_base = pd.read_csv(BASELINE_DIR / "train_baseline.csv")
    val_base = pd.read_csv(BASELINE_DIR / "val_baseline.csv")
    test_base = pd.read_csv(BASELINE_DIR / "test_baseline.csv")

    # Création du set de phrases Ewe INTERDITES (présentes dans Train ou Val)
    # On normalise en minuscules/strip pour être sûr de tout attraper
    forbidden_ewe = set(train_base['text_ewe'].astype(str).str.strip().unique())
    forbidden_ewe.update(val_base['text_ewe'].astype(str).str.strip().unique())
    forbidden_ewe.update(test_base['text_ewe'].astype(str).str.strip().unique())
    

    print(f"🚫 Total unique Ewe sentences forbidden (from Train/Val): {len(forbidden_ewe)}")

    for ratio_key, ratio_path in RATIO_FILES.items():
        if not ratio_path.exists():
            print(f"⚠️ Warning: {ratio_path} not found.")
            continue
            
        print(f"\n--- Processing Ratio {ratio_key} ---")
        df_eng = pd.read_csv(ratio_path).dropna(subset=['text_ewe', 'text_french'])
        
        # 1. Filtrage Sémantique (Qualité de la paire)
        df_eng = filter_dataframe_semantically(df_eng, 'text_french', 'text_ewe')
        
        # --- AJOUTE CECI ICI ---
        if 'sim_score' in df_eng.columns:
            df_eng = df_eng.drop(columns=['sim_score'])

        # 2. ANTI-LEAKAGE STRICT (On retire tout ce que le modèle a pu voir au training)
        initial_count = len(df_eng)
        df_eng['ewe_clean'] = df_eng['text_ewe'].astype(str).str.strip()
        df_eng = df_eng[~df_eng['ewe_clean'].isin(forbidden_ewe)]
        
        print(f"🛡️ Cross-Leakage: {initial_count - len(df_eng)} rows removed because Ewe text exists in Train/Val baseline.")

        # 3. Concaténation avec le test set original
        # Note: Les colonnes seront text_ewe, text_french (NaN pour l'anglais) et text_french (NaN pour le français)
        final_test = pd.concat([test_base, df_eng], axis=0, ignore_index=True)
        
        # Nettoyage colonne temporaire
        if 'ewe_clean' in final_test.columns:
            final_test = final_test.drop(columns=['ewe_clean'])

        # 4. Sauvegarde
        file_label = ratio_key.replace("_", "_to_")
        output_path = OUTPUT_DIR / f"test_baseline_plus_english_ewe_ratio_{file_label}.csv"
        final_test.to_csv(output_path, index=False)
        
        print(f"✅ Created {output_path.name}")
        print(f"📊 Final composition: {len(test_base)} (FR-EWE) + {len(df_eng)} (ENG-EWE) = {len(final_test)} total rows.")

if __name__ == "__main__":

    print("*"*50)
    print(" SCRIPT NAME : create_ewe_baseline_plus_ratio_datasets.py")
    print("*"*50)

    create_augmented_test_sets()