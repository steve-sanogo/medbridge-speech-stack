#!/usr/bin/env python3
"""
audit_dataset.py — Audit post-merge du dataset Éwé ASR

Vérifie la distribution, les overlaps de speakers/textes, et les anomalies
du dataset final après fusion de nouvelles sources dans le split train.

Supporte les deux formats coexistants :
  - Omnilingual : colonnes audio_bytes, audio_size, text
  - HuggingFace : colonnes audio (dict bytes/array), transcription

Usage :
    python scripts/audit/audit_dataset.py \\
        --data-dir /path/to/parquet/dir \\
        --output-dir outputs/audit \\
        [--json] [--sample-audio 500] [--no-duration]
"""

import argparse
import io
import json
import re
import sys
import unicodedata
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# soundfile est utilisé pour lire les durées depuis les bytes audio
try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False
    warnings.warn("soundfile non disponible : durées depuis bytes audio non calculables.")


# ── Constantes ────────────────────────────────────────────────────────────────

AUDIO_SAMPLE_RATE = 16000  # taux utilisé dans le pipeline Omnilingual

# Patterns pour inférer le split depuis le chemin complet
SPLIT_PATTERNS = {
    "train":      re.compile(r"train",             re.IGNORECASE),
    "validation": re.compile(r"val(?:id)?|dev",    re.IGNORECASE),
    "test":       re.compile(r"test",              re.IGNORECASE),
}

# Colonnes légères à charger pour les métadonnées (pas l'audio)
META_COLS = [
    "transcription", "text",
    "speaker_id",
    "locale", "gender", "age",
    "language", "corpus", "source",
    "audio_size",   # Omnilingual : nombre de samples à 16kHz
    "split",        # présent dans certains parquets, utilisé pour validation
]


# ── Découverte des fichiers ───────────────────────────────────────────────────

def discover_parquets(data_dir: Path) -> dict[str, list[Path]]:
    """
    Parcourt récursivement data_dir et groupe les .parquet par split inféré.
    Priorité : nom de fichier d'abord, puis chemin complet en fallback.
    Les fichiers non reconnus sont regroupés sous 'unknown'.
    """
    all_parquets = sorted(data_dir.rglob("*.parquet"))

    if not all_parquets:
        return {}

    splits: dict[str, list[Path]] = defaultdict(list)

    for path in all_parquets:
        matched = False
        # 1. Inférer depuis le nom de fichier uniquement (plus précis)
        for split_name, pattern in SPLIT_PATTERNS.items():
            if pattern.search(path.name):
                splits[split_name].append(path)
                matched = True
                break
        # 2. Fallback : chercher dans le chemin complet (répertoires parents)
        if not matched:
            for split_name, pattern in SPLIT_PATTERNS.items():
                if pattern.search(str(path)):
                    splits[split_name].append(path)
                    matched = True
                    break
        if not matched:
            splits["unknown"].append(path)

    return dict(splits)


# ── Chargement des métadonnées ────────────────────────────────────────────────

def load_metadata(paths: list[Path]) -> pd.DataFrame:
    """
    Charge uniquement les colonnes de métadonnées (sans audio) depuis
    une liste de fichiers parquet. Gère les colonnes absentes.
    Retourne un DataFrame concaténé avec une colonne _source_file.

    Utilise pyarrow.parquet.ParquetFile pour lire le schéma sans charger
    les données, puis effectue une seule lecture pandas avec les colonnes filtrées.
    """
    frames = []
    for path in paths:
        try:
            # Lecture du schéma uniquement (pas de données chargées)
            all_cols = pq.ParquetFile(path).schema_arrow.names
            cols_to_load = [c for c in META_COLS if c in all_cols]

            if not cols_to_load:
                warnings.warn(f"{path.name} : aucune colonne de métadonnées reconnue.")
                continue

            df = pd.read_parquet(path, columns=cols_to_load)
            df["_source_file"] = path.name
            frames.append(df)

        except Exception as e:
            warnings.warn(f"Impossible de charger {path}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ── Extraction des durées audio ───────────────────────────────────────────────

def extract_durations(paths: list[Path], sample_n: int = 0) -> pd.Series:
    """
    Extrait les durées audio (en secondes) pour une liste de fichiers parquet.

    Stratégie (par ordre de préférence) :
      1. Colonne audio_size (Omnilingual) : durée = audio_size / 16000
      2. Colonne audio.array + sampling_rate (HuggingFace array)
      3. soundfile.info() sur audio.bytes (HuggingFace bytes)
      4. NaN si aucun format reconnu

    sample_n : si > 0, échantillonne N lignes par fichier (estimation rapide).
    """
    all_durations = []

    for path in paths:
        try:
            all_cols = pd.read_parquet(path, columns=None).columns.tolist()

            # Chemin rapide : audio_size disponible
            if "audio_size" in all_cols:
                ser = pd.read_parquet(path, columns=["audio_size"])["audio_size"]
                all_durations.append(ser / AUDIO_SAMPLE_RATE)
                continue

            if "audio" not in all_cols:
                warnings.warn(f"{path.name} : pas de colonne audio ni audio_size.")
                continue

            # Chargement de la colonne audio (potentiellement lourd)
            df_audio = pd.read_parquet(path, columns=["audio"])
            if sample_n > 0 and len(df_audio) > sample_n:
                df_audio = df_audio.sample(sample_n, random_state=42)

            durations = df_audio["audio"].apply(_extract_single_duration)
            all_durations.append(durations)

        except Exception as e:
            warnings.warn(f"Extraction durée échouée pour {path}: {e}")

    if not all_durations:
        return pd.Series(dtype=float)

    return pd.concat(all_durations, ignore_index=True)


def _extract_single_duration(audio_val) -> float:
    """Extrait la durée en secondes depuis une valeur audio individuelle."""
    if audio_val is None:
        return float("nan")

    if isinstance(audio_val, dict):
        # Format HuggingFace array
        if "array" in audio_val and audio_val["array"] is not None:
            arr = np.asarray(audio_val["array"])
            sr = int(audio_val.get("sampling_rate", AUDIO_SAMPLE_RATE))
            return len(arr) / sr if sr > 0 else float("nan")

        # Format HuggingFace bytes
        if "bytes" in audio_val and _HAS_SF:
            try:
                raw = audio_val["bytes"]
                if isinstance(raw, str):
                    raw = raw.encode("latin-1")
                info = sf.info(io.BytesIO(raw))
                return info.duration
            except Exception:
                pass

    return float("nan")


# ── Statistiques par split ────────────────────────────────────────────────────

def compute_split_stats(df: pd.DataFrame, durations: pd.Series, split: str) -> dict:
    """
    Calcule les statistiques descriptives pour un split.
    Gère les colonnes absentes de façon défensive.
    """
    stats: dict = {"split": split, "n_rows": len(df)}

    # Colonne texte (Omnilingual: 'text', HuggingFace: 'transcription')
    text_col = (
        "transcription" if "transcription" in df.columns else
        "text"          if "text"          in df.columns else
        None
    )
    if text_col:
        empty_mask = df[text_col].isna() | (df[text_col].astype(str).str.strip() == "")
        stats["n_empty_transcriptions"] = int(empty_mask.sum())

    # Speakers
    if "speaker_id" in df.columns:
        stats["n_speakers"]           = int(df["speaker_id"].nunique())
        stats["n_missing_speaker_id"] = int(df["speaker_id"].isna().sum())
    else:
        stats["n_speakers"]           = None
        stats["n_missing_speaker_id"] = None

    # Colonnes catégorielles optionnelles
    for col in ("locale", "gender", "language", "corpus", "source"):
        if col in df.columns:
            stats[f"{col}_values"] = df[col].value_counts().to_dict()

    # Durées audio
    valid_dur = durations.dropna()
    stats["n_duration_available"] = int(len(valid_dur))

    if len(valid_dur) > 0:
        stats["total_hours"]  = round(float(valid_dur.sum() / 3600), 4)
        stats["mean_dur_s"]   = round(float(valid_dur.mean()),   2)
        stats["median_dur_s"] = round(float(valid_dur.median()), 2)
        stats["min_dur_s"]    = round(float(valid_dur.min()),    2)
        stats["max_dur_s"]    = round(float(valid_dur.max()),    2)
    else:
        stats["total_hours"] = None

    return stats


# ── Vérification des overlaps ─────────────────────────────────────────────────

def check_speaker_overlap(splits_meta: dict[str, pd.DataFrame]) -> dict[str, list]:
    """
    Détecte les speakers présents dans plusieurs splits (risque de contamination).
    Retourne un dict { 'train↔test': [speaker_id, ...], ... }.
    """
    speaker_sets: dict[str, set] = {}
    for split, df in splits_meta.items():
        if "speaker_id" in df.columns:
            speaker_sets[split] = set(df["speaker_id"].dropna().unique())

    overlaps: dict[str, list] = {}
    keys = list(speaker_sets.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            common = speaker_sets[a] & speaker_sets[b]
            if common:
                overlaps[f"{a}↔{b}"] = sorted(str(x) for x in common)

    return overlaps


def _normalize_text(s: str) -> str:
    """
    Normalise un texte pour la comparaison d'overlap :
    - NFKC (compatibilité Unicode : ligatures, demi-chasse, etc.)
    - lowercase + strip
    - collapse des espaces multiples
    - suppression de toute ponctuation/symbole Unicode (catégories P* et S*)
      sans toucher aux lettres accentuées ni aux caractères africains
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.lower().strip()
    s = "".join(
        ch for ch in s
        if not unicodedata.category(ch).startswith(("P", "S"))
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


def check_text_overlap(splits_meta: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """
    Détecte les transcriptions identiques entre splits (indicateur de fuite).
    Les textes sont normalisés avant comparaison (casse, espaces, ponctuation).
    Retourne un dict { 'train↔test': { 'count': N, 'examples': [...] } }.
    """
    text_sets: dict[str, set] = {}
    for split, df in splits_meta.items():
        col = (
            "transcription" if "transcription" in df.columns else
            "text"          if "text"          in df.columns else
            None
        )
        if col:
            normalized = df[col].dropna().apply(lambda x: _normalize_text(str(x)))
            text_sets[split] = set(normalized.unique()) - {""}

    overlaps: dict[str, dict] = {}
    keys = list(text_sets.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            common = text_sets[a] & text_sets[b]
            if common:
                overlaps[f"{a}↔{b}"] = {
                    "count":    len(common),
                    "examples": sorted(common)[:5],
                }

    return overlaps


# ── Validation de la colonne 'split' ─────────────────────────────────────────

_CANONICAL_SPLIT: dict[str, str] = {
    "train":      "train",
    "training":   "train",
    "validation": "validation",
    "valid":      "validation",
    "val":        "validation",
    "dev":        "validation",
    "test":       "test",
    "testing":    "test",
    "eval":       "test",
}


def _canonical_split(label: str) -> str | None:
    """Normalise un label de split vers 'train', 'validation', 'test', ou None."""
    return _CANONICAL_SPLIT.get(label.strip().lower())


def _validate_split_column(splits_meta: dict[str, pd.DataFrame]) -> list[str]:
    """
    Si un parquet contient une colonne 'split', normalise ses valeurs déclarées
    et les compare au split détecté par nom de fichier/chemin.
    Retourne une liste d'avertissements en cas d'incohérence.
    """
    warnings_list: list[str] = []
    for detected_split, df in splits_meta.items():
        if "split" not in df.columns:
            continue
        raw_declared = set(df["split"].dropna().str.strip().str.lower().unique())
        if not raw_declared:
            continue

        # Normaliser chaque valeur déclarée vers une forme canonique
        canonical_declared = {_canonical_split(v) for v in raw_declared} - {None}

        if not canonical_declared:
            warnings_list.append(
                f"[{detected_split}] Colonne 'split' contient des valeurs inconnues : "
                f"{raw_declared}."
            )
            continue

        if detected_split not in canonical_declared:
            warnings_list.append(
                f"[{detected_split}] Incohérence : colonne 'split' déclare "
                f"{canonical_declared} (brut : {raw_declared}), "
                f"mais le fichier/chemin suggère '{detected_split}'."
            )
    return warnings_list


# ── Détection d'anomalies ─────────────────────────────────────────────────────
def detect_anomalies(
    stats_list:       list[dict],
    speaker_overlaps: dict[str, list],
    text_overlaps:    dict[str, dict],
) -> list[str]:
    """
    Génère une liste d'avertissements lisibles à partir des statistiques
    et des résultats de vérification des overlaps.
    """
    anomalies: list[str] = []

    for s in stats_list:
        sp = s["split"]

        if s["n_rows"] < 50:
            anomalies.append(f"[{sp}] Très peu d'échantillons : {s['n_rows']}")

        n_speakers = s.get("n_speakers")
        if n_speakers == 0:
            anomalies.append(f"[{sp}] Aucun speaker_id valide.")

        n_missing = s.get("n_missing_speaker_id")
        if n_missing is not None and n_missing > 0:
            anomalies.append(f"[{sp}] {n_missing} speaker_id manquants.")

        n_empty = s.get("n_empty_transcriptions")
        if n_empty is not None and n_empty > 0:
            anomalies.append(f"[{sp}] {n_empty} transcriptions vides ou nulles.")

        total_hours = s.get("total_hours")
        if total_hours is not None:
            max_dur = s.get("max_dur_s")
            min_dur = s.get("min_dur_s")

            if max_dur is not None and max_dur > 60:
                anomalies.append(
                    f"[{sp}] Durée maximale suspecte : {max_dur}s (> 60s)."
                )
            if min_dur is not None and min_dur < 0.5:
                anomalies.append(
                    f"[{sp}] Durée minimale suspecte : {min_dur}s (< 0.5s)."
                )

    for pair, speakers in speaker_overlaps.items():
        anomalies.append(
            f"[OVERLAP SPEAKERS {pair}] {len(speakers)} speaker(s) en commun."
        )

    for pair, info in text_overlaps.items():
        anomalies.append(
            f"[OVERLAP TEXTE {pair}] {info['count']} texte(s) identique(s) — fuite potentielle."
        )

    return anomalies


# ── Rapport console ───────────────────────────────────────────────────────────

def print_report(
    stats_list:       list[dict],
    speaker_overlaps: dict[str, list],
    text_overlaps:    dict[str, dict],
    anomalies:        list[str],
) -> None:
    """Affiche le rapport complet dans la console."""
    sep = "=" * 68

    print(f"\n{sep}")
    print("  AUDIT POST-MERGE — Dataset Éwé ASR")
    print(sep)

    for s in stats_list:
        print(f"\n── Split : {s['split'].upper()} ──")
        print(f"  Lignes               : {s['n_rows']:,}")
        print(f"  Speakers uniques     : {s.get('n_speakers', 'N/A')}")
        print(f"  Speaker ID manquants : {s.get('n_missing_speaker_id', 'N/A')}")
        print(f"  Transcriptions vides : {s.get('n_empty_transcriptions', 'N/A')}")
        print(f"  Durées disponibles   : {s.get('n_duration_available', 0)}")

        if s.get("total_hours") is not None:
            print(f"  Heures audio totales : {s['total_hours']:.4f} h")
            print(f"  Durée moyenne        : {s['mean_dur_s']} s")
            print(f"  Durée médiane        : {s['median_dur_s']} s")
            print(f"  Durée [min, max]     : [{s['min_dur_s']} s, {s['max_dur_s']} s]")
        else:
            print("  Durées audio         : non disponibles")

        for col in ("language", "corpus", "source", "locale", "gender"):
            key = f"{col}_values"
            if key in s:
                print(f"  {col.capitalize():<20} : {s[key]}")

    print(f"\n{sep}")
    print("  OVERLAPS")
    print(sep)

    if speaker_overlaps:
        for pair, speakers in speaker_overlaps.items():
            print(f"  ⚠  Speakers {pair} : {len(speakers)} en commun")
            if len(speakers) <= 10:
                print(f"     → {speakers}")
    else:
        print("  ✓  Aucun overlap de speakers détecté.")

    if text_overlaps:
        for pair, info in text_overlaps.items():
            print(f"  ⚠  Textes {pair} : {info['count']} doublon(s)")
            print(f"     Exemples : {info['examples'][:3]}")
    else:
        print("  ✓  Aucun overlap textuel détecté.")

    print(f"\n{sep}")
    print("  ANOMALIES")
    print(sep)

    if anomalies:
        for a in anomalies:
            print(f"  ⚠  {a}")
    else:
        print("  ✓  Aucune anomalie détectée.")

    print(f"\n{sep}\n")


# ── Sauvegarde des sorties ────────────────────────────────────────────────────

def save_outputs(
    stats_list:       list[dict],
    speaker_overlaps: dict[str, list],
    text_overlaps:    dict[str, dict],
    anomalies:        list[str],
    output_dir:       Path,
    save_json:        bool = False,
) -> None:
    """Sauvegarde le résumé CSV et optionnellement le rapport JSON détaillé."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV : une ligne par split, colonnes scalaires uniquement
    csv_rows = []
    for s in stats_list:
        row = {k: v for k, v in s.items() if not isinstance(v, dict)}
        csv_rows.append(row)

    csv_path = output_dir / "audit_summary.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  → CSV  : {csv_path}")

    if save_json:
        report = {
            "stats":            stats_list,
            "speaker_overlaps": {k: list(v) for k, v in speaker_overlaps.items()},
            "text_overlaps":    text_overlaps,
            "anomalies":        anomalies,
        }
        json_path = output_dir / "audit_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"  → JSON : {json_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit post-merge du dataset Éwé ASR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python audit_dataset.py --data-dir data/parquet --output-dir outputs/audit\n"
            "  python audit_dataset.py --data-dir data/parquet --json --sample-audio 500\n"
            "  python audit_dataset.py --data-dir data/parquet --no-duration\n"
        ),
    )
    p.add_argument(
        "--data-dir", type=Path, required=True,
        help="Répertoire racine contenant les fichiers .parquet (recherche récursive).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("outputs/audit"),
        help="Dossier de sortie pour CSV et JSON (défaut : outputs/audit).",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Générer aussi un rapport JSON détaillé.",
    )
    p.add_argument(
        "--sample-audio", type=int, default=0, metavar="N",
        help="Échantillonner N lignes par fichier pour les durées (0 = tout lire).",
    )
    p.add_argument(
        "--no-duration", action="store_true",
        help="Sauter le calcul des durées audio (lecture plus rapide).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        print(f"ERROR : {args.data_dir} n'existe pas.", file=sys.stderr)
        sys.exit(1)

    # ── Découverte
    print(f"\nDécouverte des parquets dans : {args.data_dir}")
    split_paths = discover_parquets(args.data_dir)

    if not split_paths:
        print("Aucun fichier .parquet trouvé.", file=sys.stderr)
        sys.exit(1)

    for split, paths in split_paths.items():
        print(f"  {split:12s} : {len(paths)} fichier(s)")
        for p in paths:
            print(f"               {p.name}")

    # ── Chargement et statistiques
    splits_meta:  dict[str, pd.DataFrame] = {}
    splits_stats: list[dict] = []

    for split, paths in split_paths.items():
        print(f"\nChargement des métadonnées — {split} ...")
        df = load_metadata(paths)
        splits_meta[split] = df

        if args.no_duration:
            durations = pd.Series(dtype=float)
        else:
            print(f"  Extraction des durées ...")
            durations = extract_durations(paths, sample_n=args.sample_audio)

        stats = compute_split_stats(df, durations, split)
        splits_stats.append(stats)

    # ── Validation de la colonne 'split' (si présente dans les données)
    split_col_warnings = _validate_split_column(splits_meta)
    if split_col_warnings:
        print("\nAvertissements colonne 'split' :")
        for w in split_col_warnings:
            print(f"  ⚠  {w}")

    # ── Overlaps
    print("\nVérification des overlaps ...")
    speaker_overlaps = check_speaker_overlap(splits_meta)
    text_overlaps    = check_text_overlap(splits_meta)

    # ── Anomalies (les warnings colonne 'split' sont intégrés)
    anomalies = detect_anomalies(splits_stats, speaker_overlaps, text_overlaps)
    anomalies = split_col_warnings + anomalies

    # ── Rapport console
    print_report(splits_stats, speaker_overlaps, text_overlaps, anomalies)

    # ── Sauvegarde
    print("Sauvegarde des sorties ...")
    save_outputs(
        splits_stats, speaker_overlaps, text_overlaps, anomalies,
        args.output_dir, save_json=args.json,
    )

    print("Audit terminé.\n")


if __name__ == "__main__":
    main()
