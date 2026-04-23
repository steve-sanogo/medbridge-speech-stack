# MedBridge Speech Stack

**Modular repository for low-resource clinical speech technologies.**

## 1. Overview

MedBridge Speech Stack est un dépôt modulaire dédié à la recherche et à l'ingénierie pour la réduction des barrières linguistiques dans les contextes de consultations médicales. 

Le projet est structuré par tâche et par langue afin de fournir des composants réutilisables pour :
* **ASR** (Automatic Speech Recognition) : Reconnaissance automatique de la parole.
* **MT** (Machine Translation) : Traduction automatique.
* **TTS** (Text-to-Speech) : Synthèse vocale.
* **LID** (Language Identification) : Identification automatique de la langue.

Ce dépôt centralise la version finale et structurée des travaux du projet MedBridge.

---

## 2. High-Level Architecture

Le système suit un flux de traitement séquentiel permettant de convertir une entrée vocale en une sortie compréhensible pour le clinicien ou le patient.

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

L'organisation du dépôt repose sur une séparation stricte des domaines techniques et linguistiques.

```text
medbridge-speech-stack/
├── asr/                    # Reconnaissance vocale
│   ├── ewe/                # Pipelines pour l'Ewe (Omni, Whisper)
│   ├── arabic/             # Pipelines pour l'Arabe Égyptien
│   └── README.md
├── mt/                     # Traduction automatique
│   ├── ewe_fr/             # Ewe ↔ Français
│   ├── ar_fr/              # Arabe ↔ Français
│   └── README.md
├── tts/                    # Synthèse vocale
│   ├── arabic/             # Génération vocale Arabe
│   └── README.md
├── lid/                    # Identification de la langue
├── shared/                 # Configurations, scripts et assets communs
├── docs/                   # Documentation technique détaillée
└── README.md               # Point d'entrée principal
```

---

## 4. Module Details

### 4.1 ASR (Automatic Speech Recognition)
* **Ewe** : Deux approches complémentaires sont maintenues dans `asr/ewe/` :
    * `omni/` : Pipeline basé sur OmniASR.
    * `whisper/` : Scripts de fine-tuning et évaluation basés sur Whisper.
* **Egyptian Arabic** : Le module `asr/arabic/omni/` est traité indépendamment pour répondre aux spécificités linguistiques, aux jeux de données distincts et aux contraintes de prétraitement propres à ce dialecte.

### 4.2 MT (Machine Translation)
Les modules de traduction (`mt/ewe_fr/` et `mt/ar_fr/`) sont bidirectionnels. Chaque sous-répertoire contient la logique d'entraînement, le tokenizer et les scripts d'évaluation spécifiques.

### 4.3 TTS (Text-to-Speech)
Le module `tts/arabic/` est conçu pour la génération de parole dans le cadre du pipeline de consultation, permettant notamment de restituer oralement au patient le contenu traduit.

### 4.4 LID (Language Identification)
Composant critique pour le routage multilingue, il détecte la langue source afin d'orienter les données vers les branches ASR et MT appropriées.

---

## 5. Design Principles

* **Modularité** : Chaque couple tâche-langue est isolé pour faciliter la maintenance.
* **Reproductibilité** : Les fichiers de configuration (`configs/`) et les environnements (`environment.yml`) sont inclus dans chaque module.
* **Scalabilité** : L'architecture permet l'ajout de nouvelles langues sans modification structurelle majeure.
* **Gestion des fichiers volumineux** : Conformément à la politique du projet, les fichiers lourds (checkpoints de modèles `.pt`, datasets bruts) ne sont pas versionnés sur GitHub. Les liens vers les assets externes sont fournis dans les README de chaque module.

---

## 6. Current Status

Le dépôt est en phase de structuration finale.
* **Composants intégrés** : Ewe ASR (OmniASR & Whisper).
* **Composants en cours d'intégration** : Egyptian Arabic ASR, MT (Ewe/FR & Arabe/FR), Arabic TTS et LID.

---

## 7. Authors & Contributors

L'équipe du projet MedBridge est composée de :

* **Keli Kekeli**
* **Ahmad Aldenawi**
* **Patrice Sebastiano**
* **Steve Sanogo**

---

## 8. License

Usage académique et recherche uniquement, sauf mention contraire.
