"""
Fix Parquet files for Omnilingual mixture_parquet_asr_dataset compatibility.

This script:
1. Adds missing 'language' and 'corpus' columns
2. Renames ewe_train.parquet → train.parquet, ewe_test.parquet → test.parquet
3. Creates a dev.parquet split from the test set (50/50) if no dev exists
4. Validates that no transcription is empty/null
5. Regenerates language_distribution_0.tsv

Usage (run on the HPC or in Colab with Drive mounted):
    python scripts/omi/fix_parquets.py --data-dir /path/to/parquet/dir

Example:
    python scripts/omi/fix_parquets.py \
        --data-dir /home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def add_columns(df: pd.DataFrame, language: str, corpus: str) -> pd.DataFrame:
    """Add language and corpus columns if missing."""
    if "language" not in df.columns:
        df["language"] = language
        print(f"    Added column 'language' = '{language}'")
    if "corpus" not in df.columns:
        df["corpus"] = corpus
        print(f"    Added column 'corpus' = '{corpus}'")
    return df


def validate_transcriptions(df: pd.DataFrame, name: str) -> int:
    """Check for empty or null transcriptions. Returns count of bad rows."""
    if "transcription" not in df.columns:
        print(f"    WARNING: no 'transcription' column in {name}")
        return 0

    null_count = df["transcription"].isna().sum()
    empty_count = (df["transcription"].astype(str).str.strip() == "").sum()
    total_bad = null_count + empty_count

    if total_bad > 0:
        print(f"    WARNING: {name} has {null_count} null + {empty_count} empty transcriptions")
        print(f"    Dropping {total_bad} bad rows...")
        mask = df["transcription"].notna() & (df["transcription"].astype(str).str.strip() != "")
        return (~mask).sum()
    return 0


def estimate_hours(df: pd.DataFrame, audio_col: str = "audio", sample_limit: int = 200) -> float:
    """Estimate total audio duration by sampling rows."""
    total_seconds = 0.0
    n = min(len(df), sample_limit)
    indices = np.random.choice(len(df), n, replace=False) if len(df) > sample_limit else range(len(df))

    for idx in indices:
        audio_data = df.iloc[idx][audio_col]
        try:
            if isinstance(audio_data, dict) and "bytes" in audio_data:
                raw = audio_data["bytes"]
                if isinstance(raw, str):
                    raw = raw.encode("latin-1")
                arr, sr = sf.read(io.BytesIO(raw))
            elif isinstance(audio_data, dict) and "array" in audio_data:
                arr = np.asarray(audio_data["array"])
                sr = int(audio_data.get("sampling_rate", 16000))
            elif isinstance(audio_data, bytes):
                arr, sr = sf.read(io.BytesIO(audio_data))
            else:
                continue
            total_seconds += len(arr) / sr
        except Exception:
            pass

    # Extrapolate if sampled
    if n < len(df) and n > 0:
        total_seconds = total_seconds * (len(df) / n)

    return total_seconds / 3600


def main():
    parser = argparse.ArgumentParser(description="Fix Parquets for Omnilingual")
    parser.add_argument("--data-dir", required=True, help="Directory containing the parquet files")
    parser.add_argument("--language", default="ewe_Latn", help="Language ID (default: ewe_Latn)")
    parser.add_argument("--corpus", default="general", help="Corpus name (default: general)")
    parser.add_argument("--dev-ratio", type=float, default=0.5,
                        help="If no dev.parquet exists, split test into dev/test with this ratio for dev (default: 0.5)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Data directory: {data_dir}")
    print(f"Language: {args.language}, Corpus: {args.corpus}")
    print(f"Dry run: {args.dry_run}")
    print()

    # ── Step 1: Rename ewe_*.parquet → *.parquet ─────────────────────────────
    print("=== Step 1: Rename files ===")
    rename_map = {
        "ewe_train.parquet": "train.parquet",
        "ewe_test.parquet": "test.parquet",
        "ewe_dev.parquet": "dev.parquet",
        "ewe_val.parquet": "dev.parquet",
    }
    for old_name, new_name in rename_map.items():
        old_path = data_dir / old_name
        new_path = data_dir / new_name
        if old_path.exists() and not new_path.exists():
            print(f"  Rename: {old_name} → {new_name}")
            if not args.dry_run:
                old_path.rename(new_path)
        elif old_path.exists() and new_path.exists():
            print(f"  SKIP: {old_name} (target {new_name} already exists)")

    print()

    # ── Step 2: Load and fix Parquets ────────────────────────────────────────
    print("=== Step 2: Add columns + validate ===")
    splits = {}
    for split_name in ["train", "dev", "test"]:
        path = data_dir / f"{split_name}.parquet"
        if path.exists():
            print(f"  Loading {split_name}.parquet...")
            df = pd.read_parquet(path)
            print(f"    {len(df)} rows, columns: {list(df.columns)}")

            df = add_columns(df, args.language, args.corpus)

            bad = validate_transcriptions(df, split_name)
            if bad > 0:
                mask = df["transcription"].notna() & (df["transcription"].astype(str).str.strip() != "")
                df = df[mask].reset_index(drop=True)
                print(f"    After cleanup: {len(df)} rows")

            splits[split_name] = (path, df)
        else:
            print(f"  {split_name}.parquet: NOT FOUND")

    print()

    # ── Step 3: Create dev split if missing ──────────────────────────────────
    if "dev" not in splits and "test" in splits:
        print("=== Step 3: Creating dev split from test ===")
        test_path, test_df = splits["test"]

        if "speaker_id" in test_df.columns:
            # Speaker-aware split to avoid leakage
            speakers = test_df["speaker_id"].unique()
            np.random.seed(42)
            np.random.shuffle(speakers)
            n_dev = max(1, int(len(speakers) * args.dev_ratio))
            dev_speakers = set(speakers[:n_dev])

            dev_df = test_df[test_df["speaker_id"].isin(dev_speakers)].reset_index(drop=True)
            new_test_df = test_df[~test_df["speaker_id"].isin(dev_speakers)].reset_index(drop=True)
            print(f"  Speaker-aware split: {len(dev_speakers)} dev speakers, {len(speakers) - len(dev_speakers)} test speakers")
        else:
            # Random split
            np.random.seed(42)
            indices = np.random.permutation(len(test_df))
            n_dev = int(len(test_df) * args.dev_ratio)
            dev_df = test_df.iloc[indices[:n_dev]].reset_index(drop=True)
            new_test_df = test_df.iloc[indices[n_dev:]].reset_index(drop=True)
            print(f"  Random split (no speaker_id column)")

        print(f"  dev:  {len(dev_df)} rows")
        print(f"  test: {len(new_test_df)} rows")

        splits["dev"] = (data_dir / "dev.parquet", dev_df)
        splits["test"] = (test_path, new_test_df)
    elif "dev" in splits:
        print("=== Step 3: dev.parquet already exists, skipping ===")
    else:
        print("=== Step 3: WARNING — no test.parquet to split from ===")

    print()

    # ── Step 4: Save ─────────────────────────────────────────────────────────
    print("=== Step 4: Save fixed Parquets ===")
    total_hours = 0.0

    for split_name in ["train", "dev", "test"]:
        if split_name in splits:
            path, df = splits[split_name]
            print(f"  {split_name}: {len(df)} rows → {path}")
            if not args.dry_run:
                df.to_parquet(path, index=False)

            # Estimate hours (sample for speed)
            audio_col = "audio" if "audio" in df.columns else None
            if audio_col:
                hours = estimate_hours(df, audio_col)
                total_hours += hours
                print(f"    Estimated duration: {hours:.2f} hours")

    print()

    # ── Step 5: Regenerate language_distribution_0.tsv ────────────────────────
    print("=== Step 5: Generate language_distribution_0.tsv ===")
    tsv_path = data_dir / "language_distribution_0.tsv"

    # Use only train split for hours (that's what the sampler uses)
    if "train" in splits:
        _, train_df = splits["train"]
        audio_col = "audio" if "audio" in train_df.columns else None
        if audio_col:
            train_hours = estimate_hours(train_df, audio_col, sample_limit=500)
        else:
            train_hours = total_hours
    else:
        train_hours = total_hours

    tsv_df = pd.DataFrame([{
        "language": args.language,
        "corpus": args.corpus,
        "hours": round(train_hours, 4),
    }])

    print(f"  {tsv_path}")
    print(f"  Content: language={args.language}, corpus={args.corpus}, hours={train_hours:.4f}")

    if not args.dry_run:
        tsv_df.to_csv(tsv_path, sep="\t", index=False)

    print()
    print("=" * 60)
    if args.dry_run:
        print("  DRY RUN — no files were modified.")
        print("  Remove --dry-run to apply changes.")
    else:
        print("  All fixes applied successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
