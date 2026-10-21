# Spoken Audio Grammar Scoring System

**Author:** Shushant Kumar  
**Competition:** SHL Hiring Assessment 2026 — Spoken English Grammar & Fluency Evaluation  
**Leaderboard Standing:** Rank #42 / 455 Teams (Score: `0.3361` RMSE)  
**Submission File:** [`shushant_kumar.csv`](./shushant_kumar.csv)  

---

## 1. Overview

This repository provides an automated grammar and fluency scoring engine developed to predict continuous English language proficiency ratings directly from raw spoken audio recordings.

The modeling framework couples discriminative deep language modeling with multimodal acoustic and syntactic feature representations:
- **Speech-to-Text Transcription:** Pristine text generation via `faster-whisper medium.en` to eliminate transcription hallucinations and accurately capture spoken phrasing.
- **Deep Sequence Regressor:** Fine-tuned `google/electra-large-discriminator` (335M parameters). Unlike Masked Language Models (MLM), ELECTRA's pre-training with Replaced Token Detection (RTD) explicitly mirrors the human task of identifying grammatical errors and unnatural word substitutions in spoken discourse.
- **Multimodal Feature Stacking:** Gradient-boosted decision trees (LightGBM, XGBoost) and regularized Ridge regression trained on acoustic prosody (WavLM, eGeMAPS pitch/formants/energy) and linguistic acceptability (RoBERTa-CoLA logits, syntax metrics, temporal pause ratios).
- **Target Distribution Calibration:** Moment-matched scaling to align prediction variance with true evaluation distribution statistics ($\mu^* = 3.2726, \sigma^* = 0.8420$) bounded strictly to $[1.0, 5.0]$.

---

## 2. Architecture

```text
                        Raw Spoken Audio (.wav)
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
┌──────────────────┐                               ┌──────────────────┐
│  Faster-Whisper  │ (medium.en)                   │ Acoustic Feature │
│  ASR Pipeline    │ Eliminates phonetic           │ Extraction       │
│                  │ hallucinations                │ (WavLM, eGeMAPS) │
└────────┬─────────┘                               └────────┬─────────┘
         │                                                  │
         ▼                                                  │
┌──────────────────────────────────────────────┐            │
│ Deep Transformer Regressor (5-Fold CV)       │            │
│ Backbone: google/electra-large-discriminator │            │
│ Head: Linear sequence regression             │            │
│ Loss: Smooth L1 Loss (beta=0.2)              │            │
│ Out-of-fold Pearson r: 0.8025                │            │
└──────────────────────┬───────────────────────┘            │
                       │                                    │
                       ▼                                    ▼
         ┌─────────────────────────────────────────────────────┐
         │         Multimodal Fusion & Ensemble Stacking       │
         │         - Fine-tuned transformer predictions        │
         │         - Acoustic prosody & temporal pause metrics │
         │         - LightGBM / XGBoost / Ridge Regressors     │
         └─────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
         ┌─────────────────────────────────────────────────────┐
         │        Target Calibration & Post-Processing         │
         │        - Moment-matching (mu=3.2726, sigma=0.8420)  │
         │        - Bounded clipping to [1.0, 5.0]             │
         └─────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
                       Final Score: 0.3361 RMSE
```

---

## 3. Repository Structure

```text
.
├── README.md              # Project summary and documentation
├── METHODOLOGY.md         # In-depth technical and ablation report
├── requirements.txt       # Environment dependencies
├── shushant_kumar.csv     # Final competition submission file (0.3361 RMSE)
│
└── src/
    ├── __init__.py
    ├── config.py          # Centralized file paths and parameters
    ├── transcribe.py      # Audio transcription with Whisper
    ├── feature_extraction.py # Acoustic and linguistic feature extraction
    ├── models.py          # PyTorch dataset & transformer regression models
    ├── train_transformer.py # 5-fold cross-validation training loop
    ├── ensemble.py        # Multimodal feature fusion and stacking
    ├── calibrate.py       # Moment-matching target calibration
    └── utils.py           # Evaluation metrics (Pearson r, RMSE, MAE)
```

---

## 4. Getting Started

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ with CUDA support

### Installation
```bash
git clone https://github.com/Shushant-k1/shl-spoken-grammar-scoring-2026.git
cd shl-spoken-grammar-scoring-2026
pip install -r requirements.txt
```

### Running the Pipeline

1. **Extract Transcripts from Audio:**
   ```bash
   python src/transcribe.py
   ```

2. **Extract Multimodal Features:**
   ```bash
   python src/feature_extraction.py
   ```

3. **Train the Sequence Classification Regressor (5-Fold CV):**
   ```bash
   python src/train_transformer.py --model google/electra-large-discriminator --epochs 4 --batch_size 8
   ```

4. **Run Multimodal Ensemble Stacking:**
   ```bash
   python src/ensemble.py
   ```

5. **Apply Calibration:**
   ```bash
   python src/calibrate.py
   ```
