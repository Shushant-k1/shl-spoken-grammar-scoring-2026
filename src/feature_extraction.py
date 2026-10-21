import os
import sys
import re
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
import config

# Try to import textstat and language_tool
try:
    import textstat
except ImportError:
    textstat = None

try:
    import language_tool_python
    lt_tool = language_tool_python.LanguageTool('en-US')
except Exception:
    lt_tool = None

# Comprehensive list of spoken filler / hesitation words
FILLER_WORDS = set([
    "um", "uh", "er", "ah", "hmm", "like", "you know", "i mean", "so yeah", 
    "actually", "basically", "literally", "sort of", "kind of"
])

def extract_text_features(text: str) -> dict:
    """Extracts rich linguistic, readability, syntactic, and fluency features from text transcript."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        text = "empty transcription sample"
        
    words = re.findall(r'\b\w+\b', text.lower())
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    if len(sentences) == 0:
        sentences = [text]
        
    n_chars = len(text)
    n_words = len(words)
    n_sents = len(sentences)
    
    # 1. Lexical diversity
    unique_words = set(words)
    n_unique = len(unique_words)
    ttr = n_unique / max(n_words, 1)
    root_ttr = n_unique / np.sqrt(max(n_words, 1))
    
    # Sentence lengths
    sent_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    avg_sent_len = np.mean(sent_lengths) if sent_lengths else 0.0
    std_sent_len = np.std(sent_lengths) if sent_lengths else 0.0
    max_sent_len = np.max(sent_lengths) if sent_lengths else 0.0
    
    # Word lengths
    word_lengths = [len(w) for w in words]
    avg_word_len = np.mean(word_lengths) if word_lengths else 0.0
    
    # 2. Filler words / Disfluency counts
    text_lower = text.lower()
    filler_count = sum(text_lower.count(fw) for fw in FILLER_WORDS)
    filler_rate = filler_count / max(n_words, 1)
    
    # 3. Readability & Complexity (textstat)
    feats = {
        "n_chars": n_chars,
        "n_words": n_words,
        "n_sents": n_sents,
        "n_unique_words": n_unique,
        "ttr": ttr,
        "root_ttr": root_ttr,
        "avg_sent_len": avg_sent_len,
        "std_sent_len": std_sent_len,
        "max_sent_len": max_sent_len,
        "avg_word_len": avg_word_len,
        "filler_count": filler_count,
        "filler_rate": filler_rate,
    }
    
    if textstat is not None:
        try:
            feats["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
            feats["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
            feats["gunning_fog"] = textstat.gunning_fog(text)
            feats["coleman_liau_index"] = textstat.coleman_liau_index(text)
            feats["dale_chall"] = textstat.dale_chall_readability_score(text)
            feats["ari"] = textstat.automated_readability_index(text)
            feats["syllable_count"] = textstat.syllable_count(text)
            feats["polysyllable_count"] = textstat.polysyllabcount(text)
            feats["difficult_words"] = textstat.difficult_words(text)
        except Exception:
            for k in ["flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog", "coleman_liau_index", "dale_chall", "ari", "syllable_count", "polysyllable_count", "difficult_words"]:
                feats[k] = 0.0
                
    # 4. Grammar Error Checks
    if lt_tool is not None:
        try:
            matches = lt_tool.check(text)
            feats["lt_error_count"] = len(matches)
            feats["lt_error_rate"] = len(matches) / max(n_words, 1)
        except Exception:
            feats["lt_error_count"] = 0
            feats["lt_error_rate"] = 0.0
            
    return feats

def extract_audio_features(audio_path: str, sr=16000) -> dict:
    """Extracts acoustic, prosodic, pitch, energy, pause, and spectral features from audio."""
    feats = {}
    try:
        y, orig_sr = librosa.load(audio_path, sr=sr, mono=True)
        duration = len(y) / sr
        feats["audio_duration"] = duration
        
        # 1. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        feats["zcr_mean"] = float(np.mean(zcr))
        feats["zcr_std"] = float(np.std(zcr))
        feats["zcr_max"] = float(np.max(zcr))
        
        # 2. RMS Energy & Dynamics
        rms = librosa.feature.rms(y=y)[0]
        feats["rms_mean"] = float(np.mean(rms))
        feats["rms_std"] = float(np.std(rms))
        feats["rms_max"] = float(np.max(rms))
        
        # 3. Voice Activity & Silence / Pause Analysis
        # Dynamic energy threshold for pause detection
        silence_threshold = np.percentile(rms, 25)
        is_silence = rms < silence_threshold
        silence_ratio = float(np.mean(is_silence))
        feats["silence_ratio"] = silence_ratio
        feats["speech_ratio"] = 1.0 - silence_ratio
        
        # 4. Spectral Features (timbre & vocal resonance)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        feats["spec_cent_mean"] = float(np.mean(spec_cent))
        feats["spec_cent_std"] = float(np.std(spec_cent))
        
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        feats["spec_bw_mean"] = float(np.mean(spec_bw))
        feats["spec_bw_std"] = float(np.std(spec_bw))
        
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        feats["spec_roll_mean"] = float(np.mean(spec_roll))
        feats["spec_roll_std"] = float(np.std(spec_roll))
        
        # 5. Pitch (F0) Estimation via Yin
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
        valid_f0 = f0[f0 < 390]
        if len(valid_f0) > 0:
            feats["f0_mean"] = float(np.mean(valid_f0))
            feats["f0_std"] = float(np.std(valid_f0))
            feats["f0_min"] = float(np.min(valid_f0))
            feats["f0_max"] = float(np.max(valid_f0))
            feats["f0_range"] = float(np.max(valid_f0) - np.min(valid_f0))
        else:
            feats["f0_mean"] = 0.0
            feats["f0_std"] = 0.0
            feats["f0_min"] = 0.0
            feats["f0_max"] = 0.0
            feats["f0_range"] = 0.0
            
        # 6. MFCCs (13 coefficients - mean & std)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            feats[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
            feats[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))
            
    except Exception as e:
        print(f"⚠️ Error extracting audio features for {audio_path}: {e}")
        # Default placeholder features if audio corrupted
        feats["audio_duration"] = 0.0
        feats["zcr_mean"] = 0.0
        feats["rms_mean"] = 0.0
        feats["silence_ratio"] = 0.0
        feats["f0_mean"] = 0.0
        
    return feats

def _extract_single_row(args):
    row_dict, audio_dir_str, is_train = args
    fn = row_dict[config.ID_COL]
    transcript = row_dict.get("transcript", "")
    audio_path = Path(audio_dir_str) / fn
    
    t_feats = extract_text_features(transcript)
    a_feats = extract_audio_features(str(audio_path))
    
    combined = {config.ID_COL: fn}
    if is_train:
        combined[config.TARGET_COL] = row_dict[config.TARGET_COL]
        if "fold" in row_dict:
            combined["fold"] = row_dict["fold"]
            
    combined.update(t_feats)
    combined.update(a_feats)
    
    speech_time = max(a_feats.get("audio_duration", 1.0) * a_feats.get("speech_ratio", 0.8), 1.0)
    combined["speech_rate_wpm"] = (t_feats.get("n_words", 0) / speech_time) * 60.0
    return combined

def build_feature_table(df: pd.DataFrame, audio_dir: Path, is_train=True, n_workers=4) -> pd.DataFrame:
    """Builds complete multi-modal tabular feature matrix combining transcript and audio signals in parallel."""
    from concurrent.futures import ProcessPoolExecutor
    
    records = df.to_dict(orient="records")
    args_list = [(rec, str(audio_dir), is_train) for rec in records]
    
    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for res in tqdm(executor.map(_extract_single_row, args_list), total=len(args_list), desc=f"Extracting features ({'Train' if is_train else 'Test'})"):
            rows.append(res)
            
    feature_df = pd.DataFrame(rows)
    return feature_df

if __name__ == "__main__":
    print("Feature extraction module ready.")
