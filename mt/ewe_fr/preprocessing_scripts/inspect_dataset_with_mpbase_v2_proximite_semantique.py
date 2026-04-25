import os
import torch
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics.pairwise import cosine_similarity

# Force Matplotlib en mode non-interactif
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ── CONFIGURATION & HYPERPARAMÈTRES ──────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.85  # On dégage tout ce qui ressemble trop à du copier-coller
SEED = 42

BASE_DIR = Path("/home/data/projets-aps/projet6")
INPUT_CSV = BASE_DIR / "data_preprocessing/ewe/ewe_corpus.csv"
SAVE_DIR_SPLIT = BASE_DIR / "data_preprocessing/ewe/splitting_baseline"
OUTPUT_DIR = BASE_DIR / "experiments/translation_ewe_french"

# Modèle pour le filtrage sémantique
SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# ── 1. CHARGEMENT DU MODÈLE DE FILTRAGE ──────────────────────────────────────
print(f"🚀 Loading SBERT filter on {DEVICE}...")
word_embedding_model = models.Transformer(SBERT_MODEL)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
embed_model = SentenceTransformer(modules=[word_embedding_model, pooling_model]).to(DEVICE)

# ── 2. FONCTION DE CRÉATION ET FILTRAGE ───────────────────────────────────────
def create_cleaned_datasets():
    print(f"📊 Loading source data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, low_memory=False).dropna(subset=['text_ewe', 'text_french'])
    
    # Filtrage organique de base
    df = df[df['text_french'].str.len() > 2].copy()
    
    # --- ÉTAPE A : CALCUL DE SIMILARITÉ ---
    print("🧠 Computing semantic similarities for filtering...")
    texts_fr = df['text_french'].astype(str).tolist()
    texts_ewe = df['text_ewe'].astype(str).tolist()
    
    fr_emb = embed_model.encode(texts_fr, convert_to_tensor=False, show_progress_bar=True)
    ewe_emb = embed_model.encode(texts_ewe, convert_to_tensor=False, show_progress_bar=True)
    
    # Calcul ligne par ligne (Cosine Similarity)
    similarities = np.array([cosine_similarity(f.reshape(1,-1), e.reshape(1,-1))[0][0] 
                             for f, e in zip(fr_emb, ewe_emb)])
    df['sim_score'] = similarities

    # --- ÉTAPE B : FILTRAGE RADICAL ---
    initial_count = len(df)
    # 1. On retire les scores trop hauts (> 0.85) : Adieu URLs et copies FR==FR
    df = df[df['sim_score'] < SIMILARITY_THRESHOLD]
    # 2. On retire les doublons exacts de paires
    df = df.drop_duplicates(subset=['text_ewe', 'text_french'])
    
    clean_count = len(df)
    print(f"🧹 Nettoyage terminé : {initial_count - clean_count} lignes supprimées (Similarité > {SIMILARITY_THRESHOLD} ou doublons).")

    # --- ÉTAPE C : SPLIT (90/5/5) ---
    train_df, temp_df = train_test_split(df, test_size=0.10, random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED)

    # --- ÉTAPE D : ANTI-LEAKAGE (Sécurité sur les sources) ---
    val_sources = set(val_df['text_ewe'].unique())
    test_sources = set(test_df['text_ewe'].unique())
    forbidden_in_train = val_sources.union(test_sources)
    
    overlap = forbidden_in_train.intersection(set(train_df['text_ewe'].unique()))
    if len(overlap) > 0:
        print(f"⚠️  Fixing overlap: Removing {len(overlap)} rows from Train found in Val/Test.")
        train_df = train_df[~train_df['text_ewe'].isin(overlap)]

    # --- ÉTAPE E : SAUVEGARDE ET VISUALISATION ---
    if SAVE_DIR_SPLIT.exists(): shutil.rmtree(SAVE_DIR_SPLIT)
    SAVE_DIR_SPLIT.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(SAVE_DIR_SPLIT / "train_baseline.csv", index=False)
    val_df.to_csv(SAVE_DIR_SPLIT / "val_baseline.csv", index=False)
    test_df.to_csv(SAVE_DIR_SPLIT / "test_baseline.csv", index=False)
    
    print(f"✅ Data Locked: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Génération du Boxplot pour vérification
    generate_report(train_df, val_df, test_df)

def generate_report(train, val, test):
    plt.figure(figsize=(10, 6))
    plt.boxplot([train['sim_score'], val['sim_score'], test['sim_score']], 
                labels=['TRAIN', 'VAL', 'TEST'])
    plt.axhline(y=SIMILARITY_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({SIMILARITY_THRESHOLD})')
    plt.title("Distribution de l'alignement après filtrage sémantique")
    plt.ylabel("Cosinus Similarity")
    plt.legend()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "final_semantic_distribution.png")
    print(f"📊 Graphique de contrôle généré dans {OUTPUT_DIR}")

if __name__ == "__main__":
    create_cleaned_datasets()