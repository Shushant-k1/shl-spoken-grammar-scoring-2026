import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "Dataset_Final"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
EDA_PLOTS_DIR = NOTEBOOKS_DIR / "eda_plots"

# File paths
TRAIN_CSV = RAW_DATA_DIR / "train.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = RAW_DATA_DIR / "sample_submission.csv"

TRAIN_AUDIO_DIR = RAW_DATA_DIR / "train"
TEST_AUDIO_DIR = RAW_DATA_DIR / "test"

# Cached Artifacts
TRAIN_TRANSCRIPTS_CSV = PROCESSED_DATA_DIR / "train_transcripts.csv"
TEST_TRANSCRIPTS_CSV = PROCESSED_DATA_DIR / "test_transcripts.csv"
TRAIN_FEATURES_CSV = PROCESSED_DATA_DIR / "train_features.csv"
TEST_FEATURES_CSV = PROCESSED_DATA_DIR / "test_features.csv"
TRAIN_FOLDS_CSV = PROCESSED_DATA_DIR / "train_folds.csv"

# Global Constants
RANDOM_SEED = 42
N_FOLDS = 5
TARGET_COL = "label"
ID_COL = "filename"

# Audio Settings
SAMPLE_RATE = 16000

# Ensure directories exist
for d in [PROCESSED_DATA_DIR, MODELS_DIR, SUBMISSIONS_DIR, NOTEBOOKS_DIR, EDA_PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
