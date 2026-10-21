import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
import config

def _worker_init(model_size):
    global _worker_whisper_model
    import whisper
    import torch
    torch.set_num_threads(2)
    _worker_whisper_model = whisper.load_model(model_size, device="cpu")

def _transcribe_single(args):
    fpath = args
    if not Path(fpath).exists():
        return "", 0.0
    try:
        global _worker_whisper_model
        result = _worker_whisper_model.transcribe(str(fpath), fp16=False, language="en")
        return result.get("text", "").strip(), 0.0
    except Exception as err:
        return "", 0.0

def transcribe_dataset(model_size="tiny", n_workers=4):
    """
    Transcribes all train and test audio files using parallel OpenAI Whisper workers.
    Caches results to data/processed/train_transcripts.csv and test_transcripts.csv.
    """
    from concurrent.futures import ProcessPoolExecutor

    # 1. Transcribe Train Data
    if config.TRAIN_TRANSCRIPTS_CSV.exists():
        print(f"✅ Found cached train transcripts at: {config.TRAIN_TRANSCRIPTS_CSV}")
        train_df = pd.read_csv(config.TRAIN_TRANSCRIPTS_CSV)
    else:
        print(f"🎙️ Transcribing Training Audio Files using {n_workers} parallel workers...")
        train_raw = pd.read_csv(config.PROCESSED_DATA_DIR / "train_folds.csv" if (config.PROCESSED_DATA_DIR / "train_folds.csv").exists() else config.TRAIN_CSV)
        audio_paths = [str(config.TRAIN_AUDIO_DIR / fn) for fn in train_raw[config.ID_COL]]
        
        transcripts = []
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init, initargs=(model_size,)) as executor:
            for text, _ in tqdm(executor.map(_transcribe_single, audio_paths), total=len(audio_paths), desc="Transcribing Train"):
                transcripts.append(text)
                
        train_raw["transcript"] = transcripts
        train_raw.to_csv(config.TRAIN_TRANSCRIPTS_CSV, index=False)
        print(f"✅ Saved train transcripts to: {config.TRAIN_TRANSCRIPTS_CSV}")
        train_df = train_raw

    # 2. Transcribe Test Data
    if config.TEST_TRANSCRIPTS_CSV.exists():
        print(f"✅ Found cached test transcripts at: {config.TEST_TRANSCRIPTS_CSV}")
        test_df = pd.read_csv(config.TEST_TRANSCRIPTS_CSV)
    else:
        print(f"🎙️ Transcribing Test Audio Files using {n_workers} parallel workers...")
        test_raw = pd.read_csv(config.TEST_CSV)
        audio_paths = [str(config.TEST_AUDIO_DIR / fn) for fn in test_raw[config.ID_COL]]
        
        transcripts = []
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init, initargs=(model_size,)) as executor:
            for text, _ in tqdm(executor.map(_transcribe_single, audio_paths), total=len(audio_paths), desc="Transcribing Test"):
                transcripts.append(text)
                
        test_raw["transcript"] = transcripts
        test_raw.to_csv(config.TEST_TRANSCRIPTS_CSV, index=False)
        print(f"✅ Saved test transcripts to: {config.TEST_TRANSCRIPTS_CSV}")
        test_df = test_raw

    return train_df, test_df

if __name__ == "__main__":
    transcribe_dataset()
