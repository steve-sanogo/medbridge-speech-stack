# Ewe ASR — OmniASR (CTC)

## Overview

This module implements an Automatic Speech Recognition (ASR) system for the **Ewe language**, a low-resource language spoken in West Africa.

The model is based on **OmniASR (wav2vec2 + CTC)** and has been fine-tuned on a custom Ewe speech dataset.

---

## Objective

- Build a robust ASR system for a **low-resource language**
- Evaluate real performance beyond raw metrics
- Identify limitations of standard ASR evaluation (WER)

---

## Model

- **Base model**: `omniASR_CTC_300M`
- **Architecture**: wav2vec2 + CTC
- **Framework**: fairseq2
- **Tokenizer**: character-level

---

## Dataset

- Language: **Ewe (ewe_Latn)**
- Size: ~162 hours (V6)
- Clean version: ~37k samples (V6_clean)

### Preprocessing

- resampling → 16 kHz
- mono conversion
- clipping detection
- RMS filtering
- FLAC compression
- dataset decontamination (train/val/test)

---

## Training Strategy

- encoder freezing (first 1000 steps)
- full fine-tuning
- mixed precision (bfloat16)
- gradient accumulation

---

## Results

### Raw evaluation

| Metric | Score |
|------|------|
| WER | 63.20% |
| CER | 19.96% |

### After normalization

| Metric | Score |
|------|------|
| WER | **33.35%** |
| CER | **9.55%** |

---

## Key Insight

> Raw WER significantly overestimates model error in low-resource settings.

Most errors are due to:
- Unicode variations (ɛ → e, ɔ → o)
- orthographic inconsistencies
- punctuation mismatches

---

## Interpretation

- The model correctly captures **acoustic information**
- Errors are mainly **orthographic**, not phonetic
- CER confirms good character-level performance

---

## Inference

Example:

```bash
python scripts/inference.py \
    --model-path deploy_models/med_asr_ewe_v6_clean \
    --audio sample.wav
````

---

## Limitations

* limited training data (~160h)
* lack of standardized orthography
* sensitivity of WER to formatting

---

## Future Work

* orthographic normalization module
* data augmentation (SpecAugment)
* comparison with Whisper (seq2seq)
* extension to other African languages

---

## Contribution

This work shows that:

> Evaluation methodology is critical in low-resource ASR.

Raw WER alone can lead to incorrect conclusions about model performance.

---

## Status

* training: completed
* evaluation: completed
* inference: functional

````