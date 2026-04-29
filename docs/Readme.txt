==============================
MedBridge AI — Data & Models Access Guide
==============================
Version: v1.0
Last update: April 2026

GitHub Repository:
https://github.com/steve-sanogo/medbridge-speech-stack

Hugging Face Organization:
https://huggingface.co/medbridge-ai

--------------------------------
Overview
--------------------------------
This folder contains tutorial notebooks to help you:

- Load datasets from Hugging Face
- Load trained models (ASR / Translation)
- Test inference pipelines

All team resources (models & datasets) are hosted on Hugging Face.

--------------------------------
Project Structure
--------------------------------
Each member is responsible for specific components:

- Steve → Omni ASR (Ewe) + pipeline
- Keli → Whisper ASR (Ewe)
- Patrice → Machine Translation (NLLB)
- Hamad → Omni ASR (Arabic)

--------------------------------
How to Use
--------------------------------

1. Open the tutorial notebook corresponding to your task
2. Run all cells step by step
3. Verify that:
   - dataset loads correctly
   - model loads correctly
   - inference works

--------------------------------
Requirements
--------------------------------

Make sure you have installed:

- Python 3.10+
- transformers
- datasets
- huggingface_hub

Install with:

pip install transformers datasets huggingface_hub

--------------------------------
Quick Test Example

from huggingface_hub import snapshot_download

model_dir = snapshot_download(
    repo_id="medbridge-ai/omni-ewe-asr",
    repo_type="model"
)

--------------------------------
Notes
--------------------------------

- Do NOT modify shared datasets directly
- Always test locally before uploading
- Follow naming conventions used in the organization

--------------------------------
Support
--------------------------------

If you encounter issues:
→ Patrice Sebastiano patrice.sebastiano@alumni.univ-avignon.fr
→ Hamad Aldenawi hamad.aldenawi@alumni.univ-avignon.fr
→ kekeli keli kekeli.keli@alumni.univ-avignon.fr
→ steve sanogo moussa-steve-belvin.sanogo@alumni.univ-avignon.fr