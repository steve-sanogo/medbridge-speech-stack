# Ewe ASR : OmniASR (CTC)

## 1. Overview

This module implements an Automatic Speech Recognition (ASR) system specifically designed for **Ewe**, a low-resource language spoken in West Africa. The system leverages the **OmniASR** architecture (wav2vec2 + CTC), fine-tuned on a specialized Ewe speech corpus to address the challenges of clinical and general-purpose transcription.

---

## 2. Objectives

The primary goals of this module are:
* **Robustness**: Developing a high-performance ASR system for a low-resource linguistic environment.
* **Performance Analysis**: Evaluating real-world model utility beyond standard automated metrics.
* **Methodological Refinement**: Identifying the specific limitations of Word Error Rate (WER) in the context of non-standardized orthographies.

---

## 3. Technical Specifications

### Model Architecture
* **Base Model**: `omniASR_CTC_300M`
* **Architecture**: wav2vec2 + CTC (Connectionist Temporal Classification)
* **Framework**: fairseq2
* **Tokenizer**: Character-level tokenization

### Dataset Characteristics
* **Language**: Ewe (ewe_Latn)
* **Total Volume**: ~162 hours (V6)
* **Refined Subset**: ~37k samples (V6_clean)

### Preprocessing Pipeline
To ensure signal quality and consistency, the following steps are applied:
1. **Normalization**: Resampling to 16 kHz and mono-channel conversion.
2. **Quality Control**: Clipping detection and RMS-based silence/noise filtering.
3. **Encoding**: FLAC compression for optimized storage.
4. **Data Integrity**: Rigorous decontamination between training, validation, and testing sets.

---

## 4. Training Strategy

The fine-tuning process follows a structured approach:
* **Encoder Warm-up**: The encoder remains frozen for the first 1000 steps to stabilize the CTC head.
* **Optimization**: Full fine-tuning using mixed precision (bfloat16) to optimize memory and compute.
* **Stability**: Implementation of gradient accumulation to simulate larger batch sizes.

---

## 5. Experimental Results

### 5.1 Raw Evaluation
Standard metrics computed without post-processing:

| Metric | Score |
| :--- | :--- |
| **WER** | 63.20% |
| **CER** | 19.96% |

### 5.2 Normalized Evaluation
Metrics computed after orthographic normalization:

| Metric | Score |
| :--- | :--- |
| **WER** | **33.35%** |
| **CER** | **9.55%** |

---

## 6. Key Insights & Interpretation

> **Critical Observation**: Raw WER significantly overestimates model error in low-resource settings.

Our analysis reveals that the majority of transcription errors are **orthographic** rather than **phonetic**. Discrepancies often stem from:
* **Unicode variations**: Confusion between standard and extended Latin characters (e.g., ɛ vs e, ɔ vs o).
* **Inconsistencies**: Lack of standardized orthography in the training data.
* **Formatting**: Punctuation and casing mismatches that inflate WER.

The low **Character Error Rate (CER)** confirms that the model successfully captures the underlying acoustic information and phonetic structure of the Ewe language.

---

## 7. Inference

To run inference using the fine-tuned model:

```bash
python scripts/inference.py \
    --model-path deploy_models/med_asr_ewe_v6_clean \
    --audio sample.wav
```

---

## 8. Limitations & Future Work

### Current Limitations
* **Data Scarcity**: Performance is constrained by the ~160h training volume.
* **Orthography**: Sensitivity to non-standardized writing systems.

### Roadmap
* **Normalization**: Development of an automated orthographic normalization post-processor.
* **Augmentation**: Implementation of SpecAugment to improve generalization.
* **Benchmarking**: Comparative study with Seq2Seq architectures (e.g., OpenAI Whisper).
* **Expansion**: Extending the methodology to other West African languages.

---

## 9. Conclusion

This module demonstrates that **evaluation methodology is critical** in low-resource ASR. Relying solely on raw WER can lead to misleading conclusions; a nuanced approach focusing on phonetic accuracy (CER) and normalization provides a more accurate representation of model capabilities.

**Status**: 
- [x] Training Completed
- [x] Evaluation Completed
- [x] Inference Functional