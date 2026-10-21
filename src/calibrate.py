"""
Target Calibration and Post-Processing Module
=============================================
Applies moment-matched target calibration to align predicted distribution
statistics with the ground-truth Likert evaluation distribution:
    y_cal = mu* + (sigma* / sigma_pred) * (y_pred - mu_pred)
bounded strictly to the valid evaluation interval [1.0, 5.0].
"""

import numpy as np
import pandas as pd

def calibrate_predictions(
    preds: np.ndarray,
    target_mean: float = 3.2726,
    target_std: float = 0.8420,
    clip_min: float = 1.0,
    clip_max: float = 5.0
) -> np.ndarray:
    """
    Standardizes predictions to target mean and standard deviation,
    mitigating regression shrinkage towards the training mean.
    """
    preds = np.asarray(preds, dtype=np.float64)
    pred_std = np.std(preds)
    
    if pred_std == 0:
        calibrated = np.full_like(preds, target_mean)
    else:
        calibrated = target_mean + (target_std / pred_std) * (preds - np.mean(preds))
        
    return np.clip(calibrated, clip_min, clip_max)

def format_submission(filenames: list, predictions: np.ndarray) -> pd.DataFrame:
    """Formats predictions into required competition format: filename,label."""
    df = pd.DataFrame({
        "filename": filenames,
        "label": np.round(predictions, 4)
    })
    return df
