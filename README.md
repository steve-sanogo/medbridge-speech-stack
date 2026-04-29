# MedBridge Speech Stack

**Modular repository for low-resource clinical speech technologies.**

## 1. Overview

MedBridge Speech Stack is a modular repository dedicated to research and engineering aimed at reducing language barriers in medical consultation contexts.

The project is structured by task and by language to provide reusable components for:
* **ASR** (Automatic Speech Recognition): Automatic speech recognition.
* **MT** (Machine Translation): Machine translation.
* **TTS** (Text-to-Speech): Speech synthesis.
* **LID** (Language Identification): Automatic language identification.

This repository centralizes the final and structured version of the MedBridge project's work.

---

## 2. High-Level Architecture

The system follows a sequential processing flow, enabling the conversion of voice input into an output understandable for the clinician or the patient.

```text
Audio Input
    ↓
[LID: Language Identification]
    ↓
[ASR: Ewe / Arabic]
    ↓
[MT: Ewe ↔ French / Arabic ↔ French]
    ↓
Text Output (Clinician)
    ↓
[TTS: Arabic - Optional]
    ↓
Speech Output (Patient)
```

---

## 3. Repository Structure

The repository organization is based on a strict separation of technical and linguistic domains.

```text
medbridge-speech-stack/
├── asr/                    # Speech Recognition
│   ├── ewe/                # Pipelines for Ewe (Omni, Whisper)
│   ├── arabic/             # Pipelines for Egyptian Arabic
│   └── README.md
├── mt/                     # Machine Translation
│   ├── ewe_fr/             # Ewe ↔ French
│   ├── ar_fr/              # Arabic ↔ French
│   └── README.md
├── tts/                    # Speech Synthesis
│   ├── arabic/             # Arabic Speech Generation
│   └── README.md
├── lid/                    # Language Identification
├── shared/                 # Common configurations, scripts, and assets
├── docs/                   # Detailed technical documentation
└── README.md               # Main entry point
```

---

## 4. Module Details

### 4.1 ASR (Automatic Speech Recognition)
* **Ewe**: Two complementary approaches are maintained in `asr/ewe/`:
    * `omni/`: Pipeline based on OmniASR.
    * `whisper/`: Fine-tuning and evaluation scripts based on Whisper.
* **Egyptian Arabic**: The `asr/arabic/omni/` module is handled independently to address specific linguistic features, distinct datasets, and preprocessing constraints unique to this dialect.

### 4.2 MT (Machine Translation)
The translation modules (`mt/ewe_fr/` and `mt/ar_fr/`) are bidirectional. Each subdirectory contains the training logic, tokenizer, and specific evaluation scripts.

### 4.3 TTS (Text-to-Speech)
The `tts/arabic/` module is designed for speech generation within the consultation pipeline, specifically allowing translated content to be read back to the patient.

### 4.4 LID (Language Identification)
A critical component for multilingual routing, it detects the source language to direct data to the appropriate ASR and MT branches.

---

## 5. Design Principles

* **Modularity**: Each task-language pair is isolated to facilitate maintenance.
* **Reproducibility**: Configuration files (`configs/`) and environments (`environment.yml`) are included in each module.
* **Scalability**: The architecture allows for the addition of new languages without major structural modifications.
* **Large File Management**: In accordance with project policy, heavy files (model checkpoints `.pt`, raw datasets) are not versioned on GitHub. Links to external assets are provided in the README of each module.

---

## 6. Current Status

The repository is in the final structuring phase.
* **Integrated components**: Ewe ASR (OmniASR & Whisper).
* **Components being integrated**: Egyptian Arabic ASR, MT (Ewe/FR & Arabic/FR), Arabic TTS, and LID.

## 7. Hugging Face Organization

We use the Hugging Face platform to host all our models and datasets for reproducibility and collaboration.

🔗 Organization:
[https://huggingface.co/medbridge-ai](https://huggingface.co/medbridge-ai)

---

### Models

- **OmniASR (Ewe)** [https://huggingface.co/medbridge-ai/omni-ewe-asr](https://huggingface.co/medbridge-ai/omni-ewe-asr)  

- **Whisper ASR (Ewe)** [https://huggingface.co/medbridge-ai/whisper-ewe-asr](https://huggingface.co/medbridge-ai/whisper-ewe-asr)  

- **NLLB Translation (Ewe → French)** [https://huggingface.co/medbridge-ai/nllb-ewe-fr](https://huggingface.co/medbridge-ai/nllb-ewe-fr)  

- **NLLB Translation (Ewe → Arabic)** [https://huggingface.co/medbridge-ai/nllb-ewe-ar](https://huggingface.co/medbridge-ai/nllb-ewe-ar)  

- **OmniASR (Arabic)** [https://huggingface.co/medbridge-ai/omni-ar-asr](https://huggingface.co/medbridge-ai/omni-ar-asr)  

---

### Datasets

- **Ewe ASR (Omni format)** [https://huggingface.co/datasets/medbridge-ai/asr-ewe-omni](https://huggingface.co/datasets/medbridge-ai/asr-ewe-omni)  

- **Ewe ASR (Whisper format)** [https://huggingface.co/datasets/medbridge-ai/asr-ewe-whisper](https://huggingface.co/datasets/medbridge-ai/asr-ewe-whisper)  

- **Arabic ASR (Omni format)** [https://huggingface.co/datasets/medbridge-ai/asr-ar-omni](https://huggingface.co/datasets/medbridge-ai/asr-ar-omni)  

- **Ewe → French Translation** [https://huggingface.co/datasets/medbridge-ai/translation-ewe-fr](https://huggingface.co/datasets/medbridge-ai/translation-ewe-fr)  

- **Ewe → Arabic Translation** [https://huggingface.co/datasets/medbridge-ai/translation-ewe-ar](https://huggingface.co/datasets/medbridge-ai/translation-ewe-ar)  

---

### Usage

All models and datasets can be loaded directly using Hugging Face:

```python
from huggingface_hub import snapshot_download

model_dir = snapshot_download(
    repo_id="medbridge-ai/omni-ewe-asr",
    repo_type="model"
)
```

---

## 8. Authors & Contributors

The MedBridge project team consists of:

* **Keli Kekeli**
* **Ahmad Aldenawi**
* **Patrice Sebastiano**
* **Steve Sanogo**

---

## 9. License
Academic and research use only, unless otherwise specified.
