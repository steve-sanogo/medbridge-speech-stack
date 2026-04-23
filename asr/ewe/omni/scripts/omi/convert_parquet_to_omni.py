"""
Convert existing Parquet files (HuggingFace format) to Omnilingual ASR format.
Processes in chunks to avoid OOM on memory-limited nodes.

Usage:
    python convert_parquet_to_omni.py \
        --input-dir /path/to/parquet/ewe \
        --output-dir /path/to/output/dataset \
        --language ewe_Latn \
        --corpus general
"""

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def audio_to_flac_list_int8(audio_data, target_sr=16000):
    """
    Convert audio (dict/bytes) to FLAC-compressed list<int8> + decoded size.
    Returns (list_of_int8, audio_size) or (None, None) on failure.
    """
    import soundfile as sf

    try:
        if isinstance(audio_data, dict):
            if "bytes" in audio_data and audio_data["bytes"] is not None:
                raw = audio_data["bytes"]
                if isinstance(raw, str):
                    raw = raw.encode("latin-1")
                arr, sr = sf.read(io.BytesIO(raw))
            elif "array" in audio_data and audio_data["array"] is not None:
                arr = np.asarray(audio_data["array"], dtype=np.float32)
                sr = int(audio_data.get("sampling_rate", 16000))
            else:
                return None, None
        elif isinstance(audio_data, bytes):
            arr, sr = sf.read(io.BytesIO(audio_data))
        else:
            return None, None

        # Mono
        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        # Resample to 16kHz
        if sr != target_sr:
            ratio = target_sr / sr
            new_len = int(len(arr) * ratio)
            indices = np.linspace(0, len(arr) - 1, new_len)
            arr = np.interp(indices, np.arange(len(arr)), arr)

        audio_size = len(arr)

        # Encode to FLAC
        buf = io.BytesIO()
        sf.write(buf, arr.astype(np.float32), target_sr, format="FLAC")
        flac_bytes = buf.getvalue()

        # Convert to list<int8>
        int8_list = []
        for b in flac_bytes:
            int8_list.append(b - 256 if b > 127 else b)

        return int8_list, audio_size

    except Exception as e:
        return None, None


def detect_splits(input_dir):
    """Find parquet files and map them to splits."""
    input_dir = Path(input_dir)
    split_map = {}
    for pattern, split_name in [
        ("*train*.parquet", "train"),
        ("*dev*.parquet", "dev"),
        ("*val*.parquet", "dev"),
        ("*test*.parquet", "test"),
    ]:
        for f in input_dir.glob(pattern):
            if split_name not in split_map:
                split_map[split_name] = f
    return split_map


SCHEMA = pa.schema([
    ("text", pa.string()),
    ("audio_bytes", pa.list_(pa.int8())),
    ("audio_size", pa.int64()),
])


def convert_split_chunked(parquet_path, split_name, output_dir, language, corpus,
                          chunk_size=200, row_group_size=100):
    """Convert one parquet file to Omnilingual format, processing in chunks."""
    print(f"\n--- Converting {split_name}: {parquet_path} ---")

    # Read only metadata first
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    print(f"  Total rows: {total_rows}")

    # Detect column names from first row group
    first_batch = pf.read_row_group(0, columns=None)
    all_cols = first_batch.column_names

    audio_col = None
    for candidate in ["audio", "speech", "wav", "input_values"]:
        if candidate in all_cols:
            audio_col = candidate
            break

    text_col = None
    for candidate in ["transcription", "text", "sentence", "transcript"]:
        if candidate in all_cols:
            text_col = candidate
            break

    if audio_col is None or text_col is None:
        print(f"  ERROR: audio_col={audio_col}, text_col={text_col}", file=sys.stderr)
        return

    print(f"  Audio: '{audio_col}', Text: '{text_col}'")
    del first_batch

    # Output path
    out_dir = Path(output_dir) / f"version=0/corpus={corpus}/split={split_name}/language={language}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "part-0000.parquet"

    # Process in chunks using ParquetWriter (streaming, low memory)
    writer = pq.ParquetWriter(out_file, SCHEMA)
    converted = 0
    skipped = 0

    # Read row groups one at a time
    for rg_idx in range(pf.metadata.num_row_groups):
        batch = pf.read_row_group(rg_idx, columns=[audio_col, text_col]).to_pandas()

        texts = []
        audio_bytes_list = []
        audio_sizes = []

        for _, row in batch.iterrows():
            text = str(row[text_col]).strip() if row[text_col] is not None else ""
            if not text:
                skipped += 1
                continue

            ab, az = audio_to_flac_list_int8(row[audio_col])
            if ab is None:
                skipped += 1
                continue

            texts.append(text)
            audio_bytes_list.append(ab)
            audio_sizes.append(az)
            converted += 1

        # Write chunk
        if texts:
            chunk_table = pa.table({
                "text": texts,
                "audio_bytes": audio_bytes_list,
                "audio_size": audio_sizes,
            }, schema=SCHEMA)
            writer.write_table(chunk_table, row_group_size=row_group_size)
            del chunk_table

        del batch, texts, audio_bytes_list, audio_sizes

        done = converted + skipped
        if done % 1000 < 200 or rg_idx == pf.metadata.num_row_groups - 1:
            print(f"  Progress: {done}/{total_rows} (converted: {converted}, skipped: {skipped})")

    writer.close()
    size_mb = out_file.stat().st_size / 1e6
    print(f"  Done: {converted} rows written to {out_file} ({size_mb:.1f} MB)")
    print(f"  Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Convert Parquet to Omnilingual ASR format")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--language", default="ewe_Latn")
    parser.add_argument("--corpus", default="general")
    parser.add_argument("--row-group-size", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("  Omnilingual ASR Parquet Converter (chunked)")
    print("=" * 60)
    print(f"Input:    {args.input_dir}")
    print(f"Output:   {args.output_dir}")
    print(f"Language: {args.language}")
    print(f"Corpus:   {args.corpus}")

    split_map = detect_splits(args.input_dir)
    if not split_map:
        print(f"\nERROR: No parquet files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nSplits found:")
    for split_name, path in sorted(split_map.items()):
        size_mb = path.stat().st_size / 1e6
        print(f"  {split_name}: {path.name} ({size_mb:.1f} MB)")

    # Convert each split
    for split_name, parquet_path in sorted(split_map.items()):
        convert_split_chunked(
            parquet_path, split_name, args.output_dir,
            args.language, args.corpus, row_group_size=args.row_group_size,
        )

    # Create dev from test if no dev
    if "dev" not in split_map and "test" in split_map:
        print("\n--- Creating dev split from test ---")
        test_file = (Path(args.output_dir) /
                     f"version=0/corpus={args.corpus}/split=test/language={args.language}/part-0000.parquet")
        if test_file.exists():
            table = pq.read_table(test_file)
            n = len(table)
            indices = np.random.RandomState(42).permutation(n)
            mid = n // 2

            dev_dir = Path(args.output_dir) / f"version=0/corpus={args.corpus}/split=dev/language={args.language}"
            dev_dir.mkdir(parents=True, exist_ok=True)

            pq.write_table(table.take(indices[:mid]), dev_dir / "part-0000.parquet",
                           row_group_size=args.row_group_size)
            pq.write_table(table.take(indices[mid:]), test_file,
                           row_group_size=args.row_group_size)
            print(f"  dev: {mid} rows, test: {n - mid} rows")
            del table

    # Generate TSV
    print("\n--- Generating language_distribution_0.tsv ---")
    train_file = (Path(args.output_dir) /
                  f"version=0/corpus={args.corpus}/split=train/language={args.language}/part-0000.parquet")
    if train_file.exists():
        sizes = pq.read_table(train_file, columns=["audio_size"]).column("audio_size").to_pylist()
        total_seconds = sum(sizes) / 16000
        hours = round(total_seconds / 3600, 4)
    else:
        hours = 0

    import pandas as pd
    tsv_path = Path(args.output_dir) / "version=0" / "language_distribution_0.tsv"
    pd.DataFrame([{"language": args.language, "corpus": args.corpus, "hours": hours}]).to_csv(
        tsv_path, sep="\t", index=False)
    print(f"  {tsv_path}: hours={hours}")

    print("\n" + "=" * 60)
    print("  Conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
