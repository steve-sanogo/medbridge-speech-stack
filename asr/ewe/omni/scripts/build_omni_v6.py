#!/usr/bin/env python3
"""
build_omni_v6.py — Convert Ewe source parquets to Omnilingual format (V6)
                   Distributed, scalable, production-grade HPC/SLURM pipeline.

Architecture:
  Each worker process writes its own Parquet files directly to disk.
  No audio data is transferred back to the main process — only lightweight
  stats (counts, seconds per split) are returned via IPC.

Audio pipeline per sample:
  audio["bytes"] (MP3, stereo, 44.1kHz)
    → torchaudio.load              → float32 tensor [channels, samples]
    → mean(dim=0)                  → mono [1, samples]
    → Resample (pre-cached)        → 16 kHz
    → clipping detection           → skip if >30% of samples |amp| > 0.95
    → amplitude normalisation      → arr / max(|arr|, 1e-8)
    → np.clip(arr, -1.0, 1.0)     → guard against sinc overshoot
    → frame-based max RMS gate     → skip if max_frame_rms < --min-rms
    → duration gate                → skip if outside [--min-dur, --max-dur]
    → soundfile FLAC encode        → bytes
    → np.frombuffer(uint8)         → list<uint8>
    → MD5 digest                   → audio_md5 string

Schema:
  id          : string   — "{source_filename}_{row_index}"
  text        : string
  audio_bytes : list<uint8>
  audio_size  : int64    — FLAC byte length
  num_samples : int64    — PCM samples at 16 kHz
  duration    : double   — seconds (num_samples / 16000)
  source_file : string   — stem of the source parquet
  audio_md5   : string   — MD5 hex digest of FLAC bytes

Output structure:
  {output_dir}/version=0/corpus={corpus}/split={split}/language={language}/
    part-{worker_id:04d}.parquet      ← one file per source parquet, no conflicts
  {output_dir}/version=0/language_distribution_0.tsv

Split mapping:
  train / training           → train
  validation / valid / val   → dev
  dev                        → dev
  test / testing             → test

Usage:
  python build_omni_v6.py \\
      --source-dir  /path/to/source/parquets \\
      --output-dir  /path/to/parquet_omni_v6 \\
      [--corpus general] [--language ewe_Latn] \\
      [--batch-size 500] [--num-workers 4] \\
      [--min-duration-s 0.5] [--max-duration-s 60.0] \\
      [--min-rms 1e-4]

Notes:
  - Must run on a compute node (torchaudio requires AVX2).
  - No full dataset held in RAM — incremental batch writing per worker.
  - No race conditions — each worker owns a unique output filename.
  - No global text deduplication — all valid (text, audio) pairs kept.
"""

import argparse
import hashlib
import io
import logging
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torchaudio
import torchaudio.transforms as T_audio

# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    """
    Build (or retrieve) the module logger with a stdout handler.
    Safe to call multiple times — the guard prevents duplicate handlers.
    Called once in the main process and once at the start of each worker.
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  [pid %(process)d]  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

log = _setup_logging()


# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_SR = 16_000

# Frame parameters for frame-based RMS (more robust than global RMS)
_FRAME_LEN = 400    # 25 ms at 16 kHz
_HOP_LEN   = 160    # 10 ms hop — used to stride non-overlapping frames

# Clipping thresholds
_CLIP_AMP_THRESHOLD  = 0.95   # amplitude above which a sample is "clipped"
_CLIP_RATIO_MAX      = 0.30   # max fraction of clipped samples before rejection

SPLIT_MAP: dict[str, str] = {
    "train":      "train",
    "training":   "train",
    "validation": "dev",
    "valid":      "dev",
    "val":        "dev",
    "dev":        "dev",
    "test":       "test",
    "testing":    "test",
}

SCHEMA = pa.schema([
    pa.field("id",          pa.string()),
    pa.field("text",        pa.string()),
    pa.field("audio_bytes", pa.list_(pa.uint8())),
    pa.field("audio_size",  pa.int64()),
    pa.field("num_samples", pa.int64()),
    pa.field("duration",    pa.float64()),
    pa.field("source_file", pa.string()),
    pa.field("audio_md5",   pa.string()),    # integrity / dedup key
])


# ── Per-process Resample cache ────────────────────────────────────────────────
# Dict lives in each worker's address space — never shared across processes.
# Avoids re-building the sinc filter on every sample of the same source SR.

_resamplers: dict[int, T_audio.Resample] = {}


def _get_resampler(orig_sr: int) -> T_audio.Resample:
    """Return a cached Resample transform for orig_sr → TARGET_SR."""
    if orig_sr not in _resamplers:
        _resamplers[orig_sr] = T_audio.Resample(orig_sr, TARGET_SR)
    return _resamplers[orig_sr]


# ── Split inference ───────────────────────────────────────────────────────────

def _canonical(raw: str) -> str | None:
    return SPLIT_MAP.get(raw.strip().lower())


def infer_split_from_filename(fname: str) -> str:
    fname_lower = fname.lower()
    for key, canonical in SPLIT_MAP.items():
        if key in fname_lower:
            return canonical
    return "train"


# ── Audio quality checks ──────────────────────────────────────────────────────

def _max_frame_rms(arr: np.ndarray) -> float:
    """
    Maximum RMS over non-overlapping frames of _FRAME_LEN samples.

    Rationale over global RMS:
      A clip with a long silence plus a short loud burst has high max-frame RMS
      but low global RMS — global RMS would incorrectly reject it as silent.
      Max-frame RMS correctly identifies that speech energy is present.
    Falls back to global RMS for very short clips (< one frame).
    """
    if len(arr) < _FRAME_LEN:
        return float(np.sqrt(np.mean(arr ** 2)))
    n_frames = len(arr) // _FRAME_LEN
    frames   = arr[: n_frames * _FRAME_LEN].reshape(n_frames, _FRAME_LEN)
    return float(np.sqrt(np.mean(frames ** 2, axis=1)).max())


def _is_clipped(arr: np.ndarray) -> bool:
    """
    Return True if more than _CLIP_RATIO_MAX of samples have absolute
    amplitude above _CLIP_AMP_THRESHOLD.

    Must be called on the *pre-normalisation* waveform (torchaudio loads
    MP3 as float32 in [-1, 1]).  After our amplitude normalisation the peak
    is always 1.0, so the check would be meaningless.
    """
    return float(np.mean(np.abs(arr) > _CLIP_AMP_THRESHOLD)) > _CLIP_RATIO_MAX


# ── Audio processing ──────────────────────────────────────────────────────────

def load_and_convert(raw_bytes: bytes) -> tuple[np.ndarray, int, bool]:
    """
    Load MP3 bytes → mono float32 at TARGET_SR, normalised and hard-clipped.

    Returns:
        arr         — processed waveform [n_samples], float32, range [-1, 1]
        num_samples — len(arr)
        clipped     — True if the *original* signal was heavily clipped

    The clipping flag is set on the raw (pre-normalisation) signal so that
    we can detect ADC saturation before we mask it with normalisation.
    """
    waveform, sr = torchaudio.load(io.BytesIO(raw_bytes))   # [C, T], float32

    # Stereo → mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)        # [1, T]

    # Resample with pre-cached transform (avoids sinc filter re-init)
    if sr != TARGET_SR:
        waveform = _get_resampler(sr)(waveform)

    arr = waveform.squeeze(0).numpy().astype(np.float32)

    # Clipping detection on the raw (torchaudio-normalised) signal
    clipped = _is_clipped(arr)

    # Amplitude normalisation — maps peak to ±1.0
    peak = float(np.abs(arr).max())
    arr  = arr / max(peak, 1e-8)

    # Hard clip — guards against sinc resampling overshoot (rare but real)
    arr = np.clip(arr, -1.0, 1.0)

    return arr, len(arr), clipped


def encode_flac(arr: np.ndarray) -> tuple[np.ndarray, int, str]:
    """
    Encode float32 PCM → FLAC bytes.

    Returns:
        uint8_arr  — FLAC bytes as uint8 NumPy array
        byte_len   — len(encoded bytes), stored as audio_size
        md5_hex    — MD5 hex digest for integrity verification
    """
    buf = io.BytesIO()
    sf.write(buf, arr, TARGET_SR, format="FLAC")
    raw      = buf.getvalue()
    uint8arr = np.frombuffer(raw, dtype=np.uint8)
    md5_hex  = hashlib.md5(raw).hexdigest()
    return uint8arr, len(raw), md5_hex


# ── Worker-local Parquet helpers ──────────────────────────────────────────────

def _get_writer(
    writers:   dict[str, pq.ParquetWriter],
    split:     str,
    out_root:  Path,
    corpus:    str,
    language:  str,
    worker_id: int,
    wlog:      logging.Logger,
) -> pq.ParquetWriter:
    """
    Return (or lazily create) the ParquetWriter for this worker's split file.

    Naming convention: part-{worker_id:04d}.parquet
    Each worker writes its own file → zero contention, no locking needed.
    """
    if split not in writers:
        out_dir = (
            out_root / "version=0"
            / f"corpus={corpus}"
            / f"split={split}"
            / f"language={language}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"part-{worker_id:04d}.parquet"
        writers[split] = pq.ParquetWriter(out_path, SCHEMA)
        wlog.info("[worker %04d] Writer opened: %s", worker_id, out_path)
    return writers[split]


def _flush_to_parquet(
    batch:     dict,
    split:     str,
    writers:   dict[str, pq.ParquetWriter],
    out_root:  Path,
    corpus:    str,
    language:  str,
    worker_id: int,
    wlog:      logging.Logger,
) -> None:
    """Write one in-memory batch directly to the worker's Parquet file."""
    writer = _get_writer(writers, split, out_root, corpus, language, worker_id, wlog)
    table  = pa.table(
        {
            "id":          pa.array(batch["ids"],     type=pa.string()),
            "text":        pa.array(batch["texts"],   type=pa.string()),
            "audio_bytes": pa.array(batch["audio"],   type=pa.list_(pa.uint8())),
            "audio_size":  pa.array(batch["sizes"],   type=pa.int64()),
            "num_samples": pa.array(batch["samples"], type=pa.int64()),
            "duration":    pa.array(batch["dur"],     type=pa.float64()),
            "source_file": pa.array(batch["src"],     type=pa.string()),
            "audio_md5":   pa.array(batch["md5"],     type=pa.string()),
        },
        schema=SCHEMA,
    )
    writer.write_table(table)


# ── Group processing ──────────────────────────────────────────────────────────

def _process_group(
    df:             pd.DataFrame,
    split:          str,
    source_id:      str,
    writers:        dict[str, pq.ParquetWriter],
    stats:          dict[str, float],
    batch_size:     int,
    out_root:       Path,
    corpus:         str,
    language:       str,
    worker_id:      int,
    min_duration_s: float,
    max_duration_s: float,
    min_rms:        float,
    wlog:           logging.Logger,
) -> tuple[int, int, int, int, int]:
    """
    Convert all rows of a single-split DataFrame and write to Parquet directly.

    Returns: (converted, skipped_total, skipped_dur, skipped_sil, skipped_clip)
    """
    converted    = 0
    skipped      = 0
    skipped_dur  = 0
    skipped_sil  = 0
    skipped_clip = 0
    warn_count   = 0

    # ── Vectorised column extraction — avoids per-row Python dict overhead
    indices = df.index.tolist()

    if "transcription" in df.columns:
        texts_raw = df["transcription"].astype(str).tolist()
    elif "text" in df.columns:
        texts_raw = df["text"].astype(str).tolist()
    else:
        texts_raw = [""] * len(df)

    audio_col = df["audio"].tolist() if "audio" in df.columns else [None] * len(df)

    # ── Batch accumulators (cleared after each flush)
    batch: dict[str, list] = {
        k: [] for k in ("ids", "texts", "audio", "sizes", "samples", "dur", "src", "md5")
    }

    def _flush() -> None:
        if batch["texts"]:
            _flush_to_parquet(
                batch, split, writers, out_root, corpus, language, worker_id, wlog
            )
            for lst in batch.values():
                lst.clear()

    # ── Main row loop
    for idx, raw_text, audio_val in zip(indices, texts_raw, audio_col):

        # Text validation
        raw_text = raw_text.strip()
        if not raw_text:
            skipped += 1
            continue

        try:
            # Audio extraction
            if audio_val is None:
                skipped += 1
                continue

            if isinstance(audio_val, dict):
                raw = audio_val.get("bytes") or audio_val.get("array")
                if raw is None or not isinstance(raw, (bytes, bytearray)):
                    skipped += 1
                    continue
            elif isinstance(audio_val, (bytes, bytearray)):
                raw = audio_val
            else:
                skipped += 1
                continue

            if isinstance(raw, str):
                raw = raw.encode("latin-1")

            # Audio processing (includes clipping flag on raw signal)
            arr, num_samples, clipped = load_and_convert(raw)
            duration_s = num_samples / TARGET_SR

            # ── Filter: clipping (checked on pre-normalisation signal)
            if clipped:
                skipped      += 1
                skipped_clip += 1
                continue

            # ── Filter: duration
            if not (min_duration_s <= duration_s <= max_duration_s):
                skipped     += 1
                skipped_dur += 1
                continue

            # ── Filter: silence (max frame RMS — more robust than global RMS)
            if _max_frame_rms(arr) < min_rms:
                skipped     += 1
                skipped_sil += 1
                continue

            # FLAC encoding + MD5
            uint8_arr, byte_len, md5_hex = encode_flac(arr)

            batch["ids"].append(f"{source_id}_{idx}")
            batch["texts"].append(raw_text)
            batch["audio"].append(uint8_arr)
            batch["sizes"].append(byte_len)
            batch["samples"].append(num_samples)
            batch["dur"].append(duration_s)
            batch["src"].append(source_id)
            batch["md5"].append(md5_hex)

            stats[split] = stats.get(split, 0.0) + duration_s
            converted += 1

        except Exception as exc:
            skipped    += 1
            warn_count += 1
            if warn_count <= 10:
                wlog.warning(
                    "[worker %04d] [%s:%s] %s: %s",
                    worker_id, source_id, idx, type(exc).__name__, exc,
                )
            if warn_count == 10:
                wlog.warning(
                    "[worker %04d] [%s] Further per-sample warnings suppressed.",
                    worker_id, source_id,
                )

        # Incremental flush
        if len(batch["texts"]) >= batch_size:
            _flush()

    _flush()  # final partial batch
    return converted, skipped, skipped_dur, skipped_sil, skipped_clip


# ── File-level worker (subprocess entry point) ────────────────────────────────

def _worker(
    fpath:          Path,
    worker_id:      int,
    out_root:       Path,
    batch_size:     int,
    corpus:         str,
    language:       str,
    min_duration_s: float,
    max_duration_s: float,
    min_rms:        float,
) -> dict:
    """
    Process one source parquet file and write output Parquet directly.

    ──────────────────────────────────────────────────────────────────
    IPC payload (returned to main process):  ← lightweight stats only
      {
        "stats":        {split: total_seconds},
        "converted":    int,
        "skipped":      int,
        "skipped_dur":  int,
        "skipped_sil":  int,
        "skipped_clip": int,
        "source":       str,
      }
    Audio bytes are NEVER returned — they are written to disk directly.
    ──────────────────────────────────────────────────────────────────
    """
    # Each subprocess needs its own logging handler
    wlog = _setup_logging()
    wlog.info("[worker %04d] Loading: %s", worker_id, fpath.name)

    result: dict = {
        "stats":        {},
        "converted":    0,
        "skipped":      0,
        "skipped_dur":  0,
        "skipped_sil":  0,
        "skipped_clip": 0,
        "source":       fpath.stem,
    }

    # Worker-local writers: split → ParquetWriter
    # Unique filename (part-{worker_id:04d}.parquet) prevents all race conditions.
    writers: dict[str, pq.ParquetWriter] = {}

    try:
        df = pd.read_parquet(fpath)
        wlog.info("[worker %04d] Rows loaded: %d", worker_id, len(df))

        if "split" in df.columns:
            groups = [
                (_canonical(str(sv)) or "train", sub)
                for sv, sub in df.groupby("split", sort=False)
            ]
        else:
            groups = [(infer_split_from_filename(fpath.name), df)]

        for split, group in groups:
            wlog.info(
                "[worker %04d] Processing split='%s' (%d rows)",
                worker_id, split, len(group),
            )
            c, s, sdur, ssil, sclip = _process_group(
                group, split, fpath.stem,
                writers, result["stats"],
                batch_size, out_root, corpus, language, worker_id,
                min_duration_s, max_duration_s, min_rms,
                wlog,
            )
            result["converted"]    += c
            result["skipped"]      += s
            result["skipped_dur"]  += sdur
            result["skipped_sil"]  += ssil
            result["skipped_clip"] += sclip
            wlog.info(
                "[worker %04d] split='%s' done — "
                "converted=%d  skipped=%d (dur=%d  sil=%d  clip=%d)",
                worker_id, split, c, s, sdur, ssil, sclip,
            )

        del df

    finally:
        # Always close writers — even if an exception was raised mid-file
        for split, writer in writers.items():
            try:
                writer.close()
                wlog.info("[worker %04d] Writer closed: split=%s", worker_id, split)
            except Exception as exc:
                wlog.error(
                    "[worker %04d] Failed to close writer for split=%s: %s",
                    worker_id, split, exc,
                )

    return result


# ── TSV generation ────────────────────────────────────────────────────────────

def write_tsv(
    output_dir: Path,
    corpus:     str,
    language:   str,
    stats:      dict[str, float],
) -> None:
    tsv_path    = output_dir / "version=0" / "language_distribution_0.tsv"
    train_hours = stats.get("train", 0.0) / 3600
    total_hours = sum(stats.values()) / 3600

    with open(tsv_path, "w") as f:
        f.write("language\tcorpus\thours\n")
        f.write(f"{language}\t{corpus}\t{train_hours:.4f}\n")

    log.info("TSV written   : %s", tsv_path)
    log.info("Train hours   : %.4fh", train_hours)
    log.info("Total hours   : %.4fh", total_hours)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Omnilingual V6 — distributed, worker-side Parquet writing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python build_omni_v6.py \\\n"
            "      --source-dir /data/ewe/parquet \\\n"
            "      --output-dir /data/ewe/parquet_omni_v6 \\\n"
            "      --num-workers 4 \\\n"
            "      --min-duration-s 0.5 --max-duration-s 60.0\n"
        ),
    )
    p.add_argument("--source-dir",     type=Path,  required=True,
                   help="Source directory (HuggingFace parquets, recursive)")
    p.add_argument("--output-dir",     type=Path,  required=True,
                   help="Omnilingual output root directory")
    p.add_argument("--corpus",         type=str,   default="general")
    p.add_argument("--language",       type=str,   default="ewe_Latn")
    p.add_argument("--batch-size",     type=int,   default=500,
                   help="Rows per write batch (default: 500)")
    p.add_argument("--num-workers",    type=int,   default=1,
                   help="Parallel worker processes — use ≥2 on multi-CPU nodes"
                        " (default: 1)")
    p.add_argument("--min-duration-s", type=float, default=0.5,
                   help="Min audio duration in seconds (default: 0.5)")
    p.add_argument("--max-duration-s", type=float, default=60.0,
                   help="Max audio duration in seconds (default: 60.0)")
    p.add_argument("--min-rms",        type=float, default=1e-4,
                   help="Min max-frame RMS — below this is silent (default: 1e-4)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.source_dir.exists():
        log.error("Source directory not found: %s", args.source_dir)
        sys.exit(1)

    source_files = sorted(args.source_dir.rglob("*.parquet"))
    if not source_files:
        log.error("No .parquet files found in: %s", args.source_dir)
        sys.exit(1)

    log.info("Source files  : %d", len(source_files))
    for f in source_files:
        log.info("  %s", f.name)
    log.info(
        "Settings — batch=%d  workers=%d  dur=[%.2fs,%.2fs]  min_rms=%.1e",
        args.batch_size, args.num_workers,
        args.min_duration_s, args.max_duration_s, args.min_rms,
    )

    # Ensure TSV parent directory exists before workers start writing
    (args.output_dir / "version=0").mkdir(parents=True, exist_ok=True)

    global_stats:  dict[str, float] = {}
    total_c = total_s = total_sdur = total_ssil = total_sclip = 0

    worker_kwargs = dict(
        out_root       = args.output_dir,
        batch_size     = args.batch_size,
        corpus         = args.corpus,
        language       = args.language,
        min_duration_s = args.min_duration_s,
        max_duration_s = args.max_duration_s,
        min_rms        = args.min_rms,
    )

    # ── Dispatch — workers write Parquet independently, return stats only ─────
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(_worker, fpath, wid, **worker_kwargs): fpath
            for wid, fpath in enumerate(source_files)
        }

        for fut in as_completed(futures):
            fpath = futures[fut]
            try:
                res = fut.result()
            except Exception:
                log.error(
                    "Worker FAILED for %s:\n%s",
                    fpath.name, traceback.format_exc(),
                )
                continue

            # Accumulate lightweight stats only — no audio bytes cross the IPC
            for split, secs in res["stats"].items():
                global_stats[split] = global_stats.get(split, 0.0) + secs

            total_c     += res["converted"]
            total_s     += res["skipped"]
            total_sdur  += res["skipped_dur"]
            total_ssil  += res["skipped_sil"]
            total_sclip += res["skipped_clip"]

            log.info(
                "Done %-40s  conv=%d  skip=%d (dur=%d sil=%d clip=%d)  total=%d",
                fpath.name,
                res["converted"], res["skipped"],
                res["skipped_dur"], res["skipped_sil"], res["skipped_clip"],
                total_c,
            )

    write_tsv(args.output_dir, args.corpus, args.language, global_stats)

    log.info("=" * 62)
    log.info("  FINAL SUMMARY")
    log.info("=" * 62)
    log.info("  Converted        : %d", total_c)
    log.info("  Skipped total    : %d", total_s)
    log.info("  ├─ duration      : %d", total_sdur)
    log.info("  ├─ silence       : %d", total_ssil)
    log.info("  └─ clipping      : %d", total_sclip)
    for split, seconds in sorted(global_stats.items()):
        log.info("  %-14s : %.4fh", split, seconds / 3600)
    log.info("=" * 62)


if __name__ == "__main__":
    main()
