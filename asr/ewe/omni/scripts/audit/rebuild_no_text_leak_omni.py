#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = "".join(
        ch for ch in s
        if not unicodedata.category(ch).startswith(("P", "S"))
    )
    return s.strip()


def detect_text_column(df: pd.DataFrame) -> str:
    if "text" in df.columns:
        return "text"
    if "transcription" in df.columns:
        return "transcription"
    raise ValueError("No text column found. Expected 'text' or 'transcription'.")


def load_normalized_texts(parquet_paths: list[Path]) -> set[str]:
    texts = set()
    for p in parquet_paths:
        df = pd.read_parquet(p)
        text_col = detect_text_column(df)
        normalized = (
            df[text_col]
            .dropna()
            .astype(str)
            .map(normalize_text)
        )
        texts.update(t for t in normalized if t)
    return texts


def list_split_files(root: Path, split_name: str) -> list[Path]:
    aliases = {
        "train": ["train", "training"],
        "validation": ["validation", "valid", "val", "dev"],
        "test": ["test", "testing", "eval"],
    }

    candidates = aliases.get(split_name, [split_name])
    found = []

    for p in root.rglob("*.parquet"):
        full = str(p).lower()
        if any(f"split={alias.lower()}" in full for alias in candidates):
            found.append(p)

    return sorted(found)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_split_verbatim(src_files: list[Path], src_root: Path, dst_root: Path) -> int:
    total_rows = 0
    for src in src_files:
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        ensure_parent(dst)
        shutil.copy2(src, dst)

        meta = pq.ParquetFile(src).metadata
        total_rows += meta.num_rows
    return total_rows


def filter_train_files(
    train_files: list[Path],
    src_root: Path,
    dst_root: Path,
    forbidden_texts: set[str],
) -> tuple[int, int]:

    kept_rows = 0
    removed_rows = 0

    for src in train_files:
        print(f"[train] Processing {src.name}...")

        rel = src.relative_to(src_root)
        dst = dst_root / rel
        ensure_parent(dst)

        pf = pq.ParquetFile(src)
        writer = None

        for batch in pf.iter_batches(batch_size=500):
            df = batch.to_pandas()

            text_col = detect_text_column(df)
            normalized = df[text_col].astype(str).map(normalize_text)

            keep_mask = ~normalized.isin(forbidden_texts)

            kept = df[keep_mask]
            removed = int((~keep_mask).sum())

            if not kept.empty:
                table = pa.Table.from_pandas(kept, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(str(dst), table.schema)

                writer.write_table(table)

            kept_rows += len(kept)
            removed_rows += removed

        if writer is not None:
            writer.close()

        print(f"[train] {src.name}: kept={kept_rows} removed={removed_rows}")

    return kept_rows, removed_rows


def recompute_language_distribution(dst_root: Path, language: str, corpus: str) -> Path:
    train_files = sorted(dst_root.rglob("split=train/**/*.parquet"))
    total_audio_size = 0

    for p in train_files:
        schema_names = pq.ParquetFile(p).schema_arrow.names
        if "audio_size" not in schema_names:
            raise ValueError(
                f"'audio_size' column missing in {p}. "
                "This script expects Omnilingual-formatted parquet."
            )
        df = pd.read_parquet(p, columns=["audio_size"])
        total_audio_size += int(df["audio_size"].sum())

    total_hours = total_audio_size / 16000 / 3600

    out_tsv = dst_root / "language_distribution_0.tsv"
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("language\tcorpus\thours\n")
        f.write(f"{language}\t{corpus}\t{total_hours:.4f}\n")

    return out_tsv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild Omnilingual dataset by removing train text overlaps with validation/test."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        required=True,
        help="Root directory of existing parquet_omni/version=0",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        required=True,
        help="Root directory of cleaned parquet_omni_clean/version=0",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ewe_Latn",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="general",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional JSON report path",
    )
    args = parser.parse_args()

    src_root = args.src_root
    dst_root = args.dst_root

    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    val_files = list_split_files(src_root, "validation")
    test_files = list_split_files(src_root, "test")
    train_files = list_split_files(src_root, "train")

    if not val_files:
        raise RuntimeError(
        f"No validation parquet files found under {src_root}. "
        "Expected paths containing split=validation or split=dev."
    )

    if not test_files:
        raise RuntimeError(
            f"No test parquet files found under {src_root}. "
            "Expected paths containing split=test."
        )

    print("Loading validation texts...")
    val_texts = load_normalized_texts(val_files)

    print("Loading test texts...")
    test_texts = load_normalized_texts(test_files)

    forbidden_texts = val_texts | test_texts

    print(f"Validation unique texts: {len(val_texts)}")
    print(f"Test unique texts: {len(test_texts)}")
    print(f"Forbidden train texts: {len(forbidden_texts)}")

    print("Copying validation split unchanged...")
    val_rows = copy_split_verbatim(val_files, src_root, dst_root)

    print("Copying test split unchanged...")
    test_rows = copy_split_verbatim(test_files, src_root, dst_root)

    print("Filtering train split...")
    kept_train_rows, removed_train_rows = filter_train_files(
        train_files=train_files,
        src_root=src_root,
        dst_root=dst_root,
        forbidden_texts=forbidden_texts,
    )

    print("Recomputing language_distribution_0.tsv ...")
    tsv_path = recompute_language_distribution(
        dst_root=dst_root,
        language=args.language,
        corpus=args.corpus,
    )

    report = {
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "language": args.language,
        "corpus": args.corpus,
        "validation_rows_copied": val_rows,
        "test_rows_copied": test_rows,
        "train_rows_kept": kept_train_rows,
        "train_rows_removed_due_to_text_overlap": removed_train_rows,
        "validation_unique_texts": len(val_texts),
        "test_unique_texts": len(test_texts),
        "forbidden_unique_texts": len(forbidden_texts),
        "language_distribution_tsv": str(tsv_path),
    }

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()