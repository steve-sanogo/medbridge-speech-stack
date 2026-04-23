Voici un **README global premium**, en Markdown, pour ton repo principal `medbridge-speech-stack`.

````markdown
# MedBridge Speech Stack

Modular repository for low-resource clinical speech technologies.

## Overview

MedBridge Speech Stack is a modular research and engineering repository dedicated to reducing language barriers in medical consultations.

The project is organized by **task** and **language**, with the objective of building reusable components for:

- automatic speech recognition (**ASR**)
- machine translation (**MT**)
- text-to-speech (**TTS**)
- language identification (**LID**)

The current repository is designed to host the final, structured version of the MedBridge project.

---

## Repository Scope

This repository is not limited to a single model.  
It is intended to host multiple language-task modules.

### Current scope

#### ASR
- **Ewe**
  - `omni/` — OmniASR-based pipeline
  - `whisper/` — Whisper-based pipeline
- **Egyptian Arabic**
  - `omni/` — OmniASR-based pipeline

#### MT
- `ewe_fr/` — Ewe ↔ French translation
- `ar_fr/` — Arabic ↔ French translation

#### TTS
- **Arabic**
  - text-to-speech module

#### LID
- language identification component for routing audio/text to the correct downstream module

---

## High-Level Architecture

```text
Audio input
   ↓
[LID]
   ↓
[ASR: Ewe / Arabic]
   ↓
[MT: Ewe↔FR / Arabic↔FR]
   ↓
Text output for clinician
   ↓
[TTS: Arabic, optional]
   ↓
Speech output
````

---

## Repository Structure

```text
medbridge-speech-stack/
├── asr/
│   ├── ewe/
│   │   ├── omni/
│   │   ├── whisper/
│   │   └── README.md
│   ├── arabic/
│   │   ├── omni/
│   │   └── README.md
│   └── README.md
│
├── mt/
│   ├── ewe_fr/
│   ├── ar_fr/
│   └── README.md
│
├── tts/
│   ├── arabic/
│   │   └── model/
│   └── README.md
│
├── lid/
│   └── README.md
│
├── docs/
│   └── README.md
│
├── shared/
│   ├── configs/
│   ├── scripts/
│   ├── assets/
│   └── README.md
│
├── .gitignore
└── README.md
```

---

## Module Details

## 1. ASR

### Ewe ASR

Two ASR approaches are maintained for Ewe:

#### `asr/ewe/omni/`

OmniASR-based pipeline for Ewe speech recognition.

Typical contents:

* training and evaluation scripts
* configuration files
* exported inference-ready model
* experiment notes and results

#### `asr/ewe/whisper/`

Whisper-based pipeline for Ewe speech recognition.

Typical contents:

* Whisper fine-tuning scripts
* decoding and evaluation scripts
* comparative experiments against OmniASR

### Egyptian Arabic ASR

#### `asr/arabic/omni/`

OmniASR-based pipeline for Egyptian Arabic speech recognition.

This module is intentionally kept separate from Ewe because:

* the linguistic setting is different,
* the datasets differ,
* the preprocessing and evaluation constraints may diverge.

---

## 2. MT

### `mt/ewe_fr/`

Machine translation module for:

* Ewe → French
* French → Ewe

### `mt/ar_fr/`

Machine translation module for:

* Arabic → French
* French → Arabic

Each MT module is expected to contain:

* training code
* evaluation code
* tokenizer / preprocessing logic
* notes on datasets and model choices

---

## 3. TTS

### `tts/arabic/`

Arabic text-to-speech module.

This component is intended for speech generation in the consultation pipeline, for example:

* spoken output to the patient,
* synthetic rendering of translated content,
* future voice-enabled interaction scenarios.

---

## 4. LID

### `lid/`

Language identification module.

Purpose:

* detect the source language,
* dispatch audio or text to the correct ASR / MT / TTS branch,
* support multilingual routing in the full MedBridge pipeline.

---

## Design Principles

This repository follows a few core principles:

### Modularization

Each task-language pair is isolated in its own subdirectory.

### Reproducibility

Code, configs, and documentation should be sufficient to understand and reproduce each module independently.

### Separation of concerns

* model code stays inside the relevant module
* shared utilities go to `shared/`
* documentation goes to `docs/`

### Scalability

The structure is designed so new languages and tasks can be added without refactoring the whole repository.

---

## Large Files Policy

Large artifacts are **not meant to be pushed directly to GitHub**.

This includes:

* checkpoints
* `.pt` model weights
* raw datasets
* `.parquet` files
* heavy outputs

Recommended practice:

* keep code and configs in Git
* store heavy artifacts externally
* reference downloadable assets in module-level README files

---

## Suggested Per-Module Contents

Each concrete module such as `asr/ewe/omni/` should ideally contain:

```text
module/
├── configs/
├── scripts/
├── deploy_models/         # optional, not necessarily versioned
├── outputs/               # optional, usually ignored by git
├── README.md
└── environment.yml        # optional, if module-specific
```

---

## Current Status

This repository is intended to host the **final organized version** of the MedBridge project.

The first fully structured component to integrate is:

* **Ewe ASR**

  * OmniASR
  * Whisper

Other components are expected to be added progressively:

* Egyptian Arabic ASR
* Ewe/French MT
* Arabic/French MT
* Arabic TTS
* LID

---

## Recommended Workflow

### For development

* work inside the appropriate module
* keep reusable helpers in `shared/`
* document important choices in module-level README files

### For experiments

* store lightweight experiment metadata in Git
* keep heavy outputs outside GitHub
* retain only final or essential artifacts in versioned form

### For release

* publish clean code
* keep deployment artifacts external if too large
* maintain a clear README in each submodule

---

## Authors

MedBridge project team.

* Keli Ke keli
* Ahmad Aldenawi
* Patrice Sebastiano
* Steve Sanogo
---

## License

Academic and research use unless stated otherwise