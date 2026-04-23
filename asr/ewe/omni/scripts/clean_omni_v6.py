#!/usr/bin/env python3
"""
clean_omni_v6.py — Remove text leakage from a parquet_omni_v6 dataset.

Strategy:
  1. Collect all normalised texts from dev + test (gold sets, kept intact).
  2. Collect all normalised texts from test only (to clean val↔test overlap).
  3. Rewrite train parquets, dropping rows whose text appears in dev OR test.
  4. Rewrite val/dev parquets, dropping rows whose text appears in test.
  5. Copy test parquets unchanged.

Text normalisation (identical to audit_dataset.py):
  NFKC + remove Unicode P* and S* category characters + lowercase + strip.

Output structure mirrors input:
  {output_dir}/version=0/corpus=.../split=.../language=.../part-XXXX.parquet

Usage:
  python clean_omni_v6.py \\
      --input-dir   /path/to/parquet_omni_v6 \\
      --output-dir  /path/to/parquet_omni_v6_clean
"""

import argparse
import sys
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# ── Text normalisation ────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith(("P", "S"))
    )
    return text.lower().strip()


# ── Parquet discovery ─────────────────────────────────────────────────────────

def _find_parquets(root: Path) -> dict[str, list[Path]]:
    """
    Walk root and return {split_label: [parquet_paths]}.
    Matches split= partitions at any depth.
    """
    splits: dict[str, list[Path]] = {}
    for p in sorted(root.rglob("*.parquet")):
        for part in p.parts:
            if part.startswith("split="):
                label = part[len("split="):]
                splits.setdefault(label, []).append(p)
                break
    return splits


# ── Contamination set builder ─────────────────────────────────────────────────

def _collect_texts(files: list[Path], text_col: str = "text") -> set[str]:
    """Return the set of normalised texts found in a list of parquet files."""
    texts: set[str] = set()
    for f in files:
        tbl = pq.read_table(f, columns=[text_col])
        for val in tbl.column(text_col).to_pylist():
            if val:
                texts.add(_normalize(str(val)))
    return texts


# ── Parquet filtering ─────────────────────────────────────────────────────────

def _filter_and_write(
    src_file:   Path,
    output_dir: Path,
    banned:     set[str],
    text_col:   str = "text",
) -> tuple[int, int]:
    """
    Read src_file, drop rows whose normalised text is in `banned`,
    write to the mirrored path under output_dir.

    Returns (kept, dropped).
    """
    # Mirror the input path under output_dir
    rel   = src_file.relative_to(src_file.parents[
        next(i for i, p in enumerate(src_file.parents)
             if p.name == "version=0")
    ])
    # Find the version=0 anchor in the path
    parts = src_file.parts
    try:
        v0_idx = next(i for i, p in enumerate(parts) if p == "version=0")
    except StopIteration:
        print(f"  WARN: cannot locate version=0 in {src_file}, skipping.", file=sys.stderr)
        return 0, 0

    out_path = output_dir / Path(*parts[v0_idx:])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tbl  = pq.read_table(src_file)
    schema = tbl.schema

    # Build boolean mask
    texts = [
        _normalize(str(v)) if v is not None else ""
        for v in tbl.column(text_col).to_pylist()
    ]
    mask  = [t not in banned for t in texts]
    kept  = sum(mask)
    dropped = len(mask) - kept

    if kept == 0:
        print(f"    WARN: all rows dropped from {src_file.name} — file not written.")
        return kept, dropped

    # Filter using PyArrow (avoids pandas dependency)
    mask_arr = pa.array(mask, type=pa.bool_())
    filtered = tbl.filter(mask_arr)

    pq.write_table(filtered, out_path)
    return kept, dropped


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remove train/val/test text leakage from parquet_omni_v6"
    )
    p.add_argument("--input-dir",  type=Path, required=True,
                   help="parquet_omni_v6 root directory")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Clean output directory (parquet_omni_v6_clean)")
    p.add_argument("--text-col",   type=str,  default="text",
                   help="Name of the text column (default: text)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        print(f"ERROR: {args.input_dir} not found.", file=sys.stderr)
        sys.exit(1)

    splits = _find_parquets(args.input_dir)
    if not splits:
        print("ERROR: no parquet files found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nInput  : {args.input_dir}")
    print(f"Output : {args.output_dir}")
    print(f"Splits found: {sorted(splits.keys())}\n")

    # Normalise split labels (dev / validation → both treated as "dev")
    DEV_LABELS  = {"dev", "validation", "valid", "val"}
    TEST_LABELS = {"test", "testing"}

    dev_files   = [f for lbl, files in splits.items() if lbl in DEV_LABELS  for f in files]
    test_files  = [f for lbl, files in splits.items() if lbl in TEST_LABELS for f in files]
    train_files = [
        f for lbl, files in splits.items()
        if lbl not in DEV_LABELS and lbl not in TEST_LABELS
        for f in files
    ]

    print(f"  dev files   : {len(dev_files)}")
    print(f"  test files  : {len(test_files)}")
    print(f"  train files : {len(train_files)}")

    # ── Build contamination sets
    print("\nCollecting dev texts  ...", end=" ", flush=True)
    dev_texts  = _collect_texts(dev_files,  args.text_col)
    print(f"{len(dev_texts):,} unique texts")

    print("Collecting test texts ...", end=" ", flush=True)
    test_texts = _collect_texts(test_files, args.text_col)
    print(f"{len(test_texts):,} unique texts")

    train_banned = dev_texts | test_texts          # train must not see either
    val_banned   = test_texts                       # val must not see test

    # ── Process each split
    total_kept = total_dropped = 0

    # TRAIN — remove anything seen in dev or test
    print(f"\n── Cleaning TRAIN ({len(train_files)} files, banned={len(train_banned):,}) ──")
    for f in sorted(train_files):
        k, d = _filter_and_write(f, args.output_dir, train_banned, args.text_col)
        print(f"  {f.name:40s}  kept={k:6d}  dropped={d:4d}")
        total_kept    += k
        total_dropped += d

    # DEV / VALIDATION — remove anything seen in test
    print(f"\n── Cleaning DEV/VAL ({len(dev_files)} files, banned={len(val_banned):,}) ──")
    for f in sorted(dev_files):
        k, d = _filter_and_write(f, args.output_dir, val_banned, args.text_col)
        print(f"  {f.name:40s}  kept={k:6d}  dropped={d:4d}")
        total_kept    += k
        total_dropped += d

    # TEST — copied as-is (no filtering)
    print(f"\n── Copying TEST unchanged ({len(test_files)} files) ──")
    for f in sorted(test_files):
        k, d = _filter_and_write(f, args.output_dir, set(), args.text_col)
        print(f"  {f.name:40s}  kept={k:6d}  dropped={d:4d}")
        total_kept += k

    # ── Also copy the TSV
    tsv_src = args.input_dir / "version=0" / "language_distribution_0.tsv"
    if tsv_src.exists():
        tsv_dst = args.output_dir / "version=0" / "language_distribution_0.tsv"
        tsv_dst.parent.mkdir(parents=True, exist_ok=True)
        tsv_dst.write_bytes(tsv_src.read_bytes())
        print(f"\nTSV copied → {tsv_dst}")
        print("NOTE: recompute TSV hours if exact figures are needed after cleaning.")

    print(f"\n{'='*58}")
    print(f"  CLEANING SUMMARY")
    print(f"{'='*58}")
    print(f"  Total kept    : {total_kept:,}")
    print(f"  Total dropped : {total_dropped:,}")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
