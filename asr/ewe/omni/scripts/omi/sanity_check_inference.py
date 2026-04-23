"""
Sanity-check inference using a fine-tuned checkpoint.

Usage:
    python scripts/omi/sanity_check_inference.py /path/to/checkpoint/model [audio_file.wav ...]

If no audio files are provided, extracts 3 samples from test.parquet.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from fairseq2.assets import AssetCard
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python sanity_check_inference.py <checkpoint_dir> [audio1.wav audio2.wav ...]")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    audio_files = sys.argv[2:] if len(sys.argv) > 2 else []

    # If no audio files given, extract from test parquet
    if not audio_files:
        test_parquet = "/home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet/test.parquet"
        print(f"No audio files given. Extracting 3 samples from {test_parquet}...")
        import tempfile

        df = pd.read_parquet(test_parquet)
        for i, (_, row) in enumerate(df.head(3).iterrows()):
            audio_data = row["audio"]
            if isinstance(audio_data, dict) and "bytes" in audio_data:
                raw = audio_data["bytes"]
                if isinstance(raw, str):
                    raw = raw.encode("latin-1")
                arr, sr = sf.read(io.BytesIO(raw))
            elif isinstance(audio_data, dict) and "array" in audio_data:
                arr = np.asarray(audio_data["array"], dtype=np.float32)
                sr = int(audio_data.get("sampling_rate", 16000))
            else:
                continue

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, arr, sr)
            audio_files.append(tmp.name)
            ref = row.get("transcription", "")
            print(f"  Sample {i}: {len(arr)/sr:.1f}s, ref: {ref[:80]}...")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    card = AssetCard(name="med_finetuned", metadata={"checkpoint": f"file://{checkpoint_dir}"})

    print(f"\nLoading model from: {checkpoint_dir}")
    print(f"Device: {device}, dtype: {dtype}\n")

    pipeline = ASRInferencePipeline(model_card=card, device=device, dtype=dtype)

    # Transcribe
    transcriptions = pipeline.transcribe(audio_files, batch_size=1)

    print("-" * 60)
    for path, text in zip(audio_files, transcriptions):
        print(f"File:       {path}")
        print(f"Transcript: {text}")
        print("-" * 60)


if __name__ == "__main__":
    main()
