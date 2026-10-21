import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes all standard competition and diagnostic metrics:
    - Pearson Correlation (r)
    - Root Mean Squared Error (RMSE)
    - Spearman Rank Correlation (rho)
    - Mean Absolute Error (MAE)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # Clip predictions to valid Likert score range [0.0, 5.0]
    y_pred_clipped = np.clip(y_pred, 0.0, 5.0)
    
    # Pearson
    if np.std(y_pred_clipped) == 0 or np.std(y_true) == 0:
        pearson_corr = 0.0
    else:
        pearson_corr, _ = pearsonr(y_true, y_pred_clipped)
        
    # Spearman
    spearman_corr, _ = spearmanr(y_true, y_pred_clipped)
    
    # RMSE & MAE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_clipped))
    mae = mean_absolute_error(y_true, y_pred_clipped)
    
    return {
        "pearson": float(pearson_corr),
        "rmse": float(rmse),
        "spearman": float(spearman_corr),
        "mae": float(mae)
    }

def print_metrics(metrics: dict, prefix: str = "Evaluation"):
    """Formatted printer for competition evaluation metrics."""
    print(f"==================================================")
    print(f" 📊 {prefix} Results")
    print(f"==================================================")
    print(f" ⭐ Pearson Correlation (r) : {metrics['pearson']:.4f}")
    print(f" 🎯 RMSE (Compulsory)       : {metrics['rmse']:.4f}")
    print(f" 📈 Spearman Correlation (ρ): {metrics['spearman']:.4f}")
    print(f" 📉 Mean Absolute Error (MAE): {metrics['mae']:.4f}")
    print(f"==================================================")
