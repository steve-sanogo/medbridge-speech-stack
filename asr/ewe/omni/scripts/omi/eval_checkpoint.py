"""
Evaluate a fine-tuned Omnilingual ASR checkpoint on a test Parquet file.

Usage:
    python scripts/omi/eval_checkpoint.py \
        --checkpoint-dir /path/to/step_5000/model \
        --test-parquet /path/to/test.parquet \
        --output-dir /path/to/eval_output
"""

import argparse
import io
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from fairseq2.assets import AssetCard
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


def load_audio_from_parquet(parquet_path: str, max_samples: int = 0) -> list[dict]:
    """Extract audio + reference transcription from a Parquet file."""
    df = pd.read_parquet(parquet_path)

    if max_samples > 0:
        df = df.head(max_samples)

    text_col = "text" if "text" in df.columns else "transcription"
    has_omni_audio = "audio_bytes" in df.columns
    has_hf_audio = "audio" in df.columns

    samples = []
    for idx, row in df.iterrows():
        ref_text = str(row.get(text_col, ""))

        try:
            if has_omni_audio:
                flac_bytes = bytes([b & 0xFF for b in row["audio_bytes"]])
                arr, sr = sf.read(io.BytesIO(flac_bytes))
            elif has_hf_audio:
                audio_data = row["audio"]
                if isinstance(audio_data, dict) and "bytes" in audio_data:
                    audio_bytes = audio_data["bytes"]
                    if isinstance(audio_bytes, str):
                        audio_bytes = audio_bytes.encode("latin-1")
                    arr, sr = sf.read(io.BytesIO(audio_bytes))
                elif isinstance(audio_data, dict) and "array" in audio_data:
                    arr = np.asarray(audio_data["array"], dtype=np.float32)
                    sr = int(audio_data.get("sampling_rate", 16000))
                elif isinstance(audio_data, bytes):
                    arr, sr = sf.read(io.BytesIO(audio_data))
                else:
                    continue
            else:
                continue

            import tempfile

            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
                dir="/tmp",
            )
            sf.write(tmp.name, arr, sr)
            samples.append(
                {
                    "path": tmp.name,
                    "reference": ref_text,
                    "speaker_id": str(row.get("speaker_id", "")),
                    "duration_s": len(arr) / sr,
                }
            )
        except Exception as e:
            print(f"[WARN] Skipping row {idx}: {e}", file=sys.stderr)

    return samples


def _levenshtein_distance(ref_units: list[str], hyp_units: list[str]) -> int:
    """Generic Levenshtein distance."""
    d = np.zeros((len(ref_units) + 1, len(hyp_units) + 1), dtype=int)

    for i in range(len(ref_units) + 1):
        d[i][0] = i
    for j in range(len(hyp_units) + 1):
        d[0][j] = j

    for i in range(1, len(ref_units) + 1):
        for j in range(1, len(hyp_units) + 1):
            cost = 0 if ref_units[i - 1] == hyp_units[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # deletion
                d[i][j - 1] + 1,      # insertion
                d[i - 1][j - 1] + cost,  # substitution / match
            )

    return int(d[len(ref_units)][len(hyp_units)])


def compute_wer(ref: str, hyp: str) -> float:
    """Simple word error rate."""
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    dist = _levenshtein_distance(ref_words, hyp_words)
    return dist / len(ref_words)


def compute_cer(ref: str, hyp: str) -> float:
    """Simple character error rate."""
    ref_chars = list(ref.strip())
    hyp_chars = list(hyp.strip())

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    dist = _levenshtein_distance(ref_chars, hyp_chars)
    return dist / len(ref_chars)


def normalize_text_general(text: str) -> str:
    """
    General diagnostic normalization for ASR evaluation.

    This is intentionally lenient and meant for analysis:
    - lowercase
    - unicode normalization
    - removal of unknown symbol '⁇'
    - punctuation removal
    - space normalization
    """
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)

    # Replace common unknown decoding token
    text = text.replace("⁇", " ")

    # Normalize apostrophe-like variants
    text = text.replace("’", "'").replace("`", "'")

    # Remove punctuation/symbols while keeping letters, digits, underscore, spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_text_ewe(text: str) -> str:
    """
    Ewe-specific diagnostic normalization on top of general normalization.

    This is NOT a linguistic gold standard; it is used to test whether
    orthographic conventions inflate WER/CER.
    """
    text = normalize_text_general(text)

    text = (
        text.replace("ɛ", "e")
            .replace("ɔ", "o")
            .replace("ɖ", "d")
            .replace("ʋ", "v")
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="Evaluate Omnilingual ASR checkpoint")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to model checkpoint directory")
    parser.add_argument("--test-parquet", required=True, help="Path to test.parquet")
    parser.add_argument("--output-dir", required=True, help="Where to save eval results")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of samples (0 = all)")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    checkpoint_uri = f"file://{args.checkpoint_dir}"
    card = AssetCard(
        name="med_finetuned",
        metadata={
            "checkpoint": checkpoint_uri,
            "model_family": "wav2vec2_asr",
            "model_arch": "300m",
            "tokenizer_family": "char_tokenizer",
            "tokenizer_ref": "omniASR_tokenizer_v1",
        },
    )

    print(f"Loading model from: {checkpoint_uri}", flush=True)
    print(f"Device: {device}, dtype: {dtype}", flush=True)

    pipeline = ASRInferencePipeline(model_card=card, device=device, dtype=dtype)

    # ── Load test data ───────────────────────────────────────────────────────
    print(f"Loading test data from: {args.test_parquet}", flush=True)
    samples = load_audio_from_parquet(args.test_parquet, max_samples=args.max_samples)
    print(f"Loaded {len(samples)} samples", flush=True)

    if not samples:
        print("ERROR: No samples loaded.", file=sys.stderr)
        sys.exit(1)

    # ── Run inference ────────────────────────────────────────────────────────
    audio_paths = [s["path"] for s in samples]
    print(f"Running inference (batch_size={args.batch_size})...", flush=True)
    predictions = pipeline.transcribe(audio_paths, batch_size=args.batch_size)

    # ── Compute metrics ──────────────────────────────────────────────────────
    results = []

    total_wer = 0.0
    total_cer = 0.0

    total_wer_norm = 0.0
    total_cer_norm = 0.0

    total_wer_ewe = 0.0
    total_cer_ewe = 0.0

    for sample, hyp in zip(samples, predictions):
        ref = sample["reference"]

        # Raw metrics
        wer = compute_wer(ref, hyp)
        cer = compute_cer(ref, hyp)

        # General normalized metrics
        ref_norm = normalize_text_general(ref)
        hyp_norm = normalize_text_general(hyp)
        wer_norm = compute_wer(ref_norm, hyp_norm)
        cer_norm = compute_cer(ref_norm, hyp_norm)

        # Ewe-specific normalized metrics
        ref_ewe = normalize_text_ewe(ref)
        hyp_ewe = normalize_text_ewe(hyp)
        wer_ewe = compute_wer(ref_ewe, hyp_ewe)
        cer_ewe = compute_cer(ref_ewe, hyp_ewe)

        total_wer += wer
        total_cer += cer

        total_wer_norm += wer_norm
        total_cer_norm += cer_norm

        total_wer_ewe += wer_ewe
        total_cer_ewe += cer_ewe

        results.append(
            {
                "speaker_id": sample["speaker_id"],
                "duration_s": round(sample["duration_s"], 2),
                "reference": ref,
                "hypothesis": hyp,
                "reference_norm": ref_norm,
                "hypothesis_norm": hyp_norm,
                "reference_ewe_norm": ref_ewe,
                "hypothesis_ewe_norm": hyp_ewe,
                "wer": round(wer, 4),
                "cer": round(cer, 4),
                "wer_norm": round(wer_norm, 4),
                "cer_norm": round(cer_norm, 4),
                "wer_ewe_norm": round(wer_ewe, 4),
                "cer_ewe_norm": round(cer_ewe, 4),
            }
        )

    avg_wer = total_wer / len(results) if results else 0.0
    avg_cer = total_cer / len(results) if results else 0.0

    avg_wer_norm = total_wer_norm / len(results) if results else 0.0
    avg_cer_norm = total_cer_norm / len(results) if results else 0.0

    avg_wer_ewe = total_wer_ewe / len(results) if results else 0.0
    avg_cer_ewe = total_cer_ewe / len(results) if results else 0.0

    # ── Save results ─────────────────────────────────────────────────────────
    df_results = pd.DataFrame(results)
    csv_path = output_dir / "eval_results.csv"
    df_results.to_csv(csv_path, index=False)

    summary = {
        "checkpoint": args.checkpoint_dir,
        "test_file": args.test_parquet,
        "num_samples": len(results),
        "avg_wer": round(avg_wer, 4),
        "avg_cer": round(avg_cer, 4),
        "avg_wer_norm": round(avg_wer_norm, 4),
        "avg_cer_norm": round(avg_cer_norm, 4),
        "avg_wer_ewe_norm": round(avg_wer_ewe, 4),
        "avg_cer_ewe_norm": round(avg_cer_ewe, 4),
        "total_audio_hours": round(sum(r["duration_s"] for r in results) / 3600, 4),
    }

    json_path = output_dir / "eval_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Print summary ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Samples evaluated      : {len(results)}")
    print(f"  Average WER (raw)      : {avg_wer:.2%}")
    print(f"  Average CER (raw)      : {avg_cer:.2%}")
    print(f"  Average WER (norm)     : {avg_wer_norm:.2%}")
    print(f"  Average CER (norm)     : {avg_cer_norm:.2%}")
    print(f"  Average WER (ewe norm) : {avg_wer_ewe:.2%}")
    print(f"  Average CER (ewe norm) : {avg_cer_ewe:.2%}")
    print(f"  Results saved to       : {csv_path}")
    print(f"  Summary saved to       : {json_path}")
    print("=" * 60)

    # Cleanup temp WAV files
    import os

    for s in samples:
        try:
            os.unlink(s["path"])
        except OSError:
            pass


if __name__ == "__main__":
    main()