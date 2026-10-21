"""
Multimodal Ensembling and Stacking Pipeline
===========================================
Fuses acoustic representations (WavLM, prosody, pause statistics) with
linguistic representations (CoLA grammar acceptability, syntax, lexical diversity)
and fine-tuned transformer sequence outputs using gradient-boosted trees and
transductive Ridge regression.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

sys.path.append(str(Path(__file__).resolve().parent))
import config
from utils import compute_metrics, print_metrics

def train_tabular_stack(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_splits: int = 5):
    """
    Trains a cross-validated ensemble of LightGBM, XGBoost, and Ridge regressors
    on extracted multimodal features.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        X_tr, y_tr = X_train_scaled[train_idx], y_train[train_idx]
        X_va, y_va = X_train_scaled[val_idx], y_train[val_idx]
        
        fold_oof = []
        fold_test = []
        
        # 1. Ridge Regressor
        ridge = Ridge(alpha=100.0)
        ridge.fit(X_tr, y_tr)
        fold_oof.append(ridge.predict(X_va))
        fold_test.append(ridge.predict(X_test_scaled))
        
        # 2. LightGBM Regressor
        if HAS_LGB:
            model_lgb = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=15,
                subsample=0.8,
                colsample_bytree=0.7,
                random_state=config.RANDOM_SEED + fold,
                verbose=-1
            )
            model_lgb.fit(X_tr, y_tr)
            fold_oof.append(model_lgb.predict(X_va))
            fold_test.append(model_lgb.predict(X_test_scaled))
            
        # 3. XGBoost Regressor
        if HAS_XGB:
            model_xgb = xgb.XGBRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.7,
                random_state=config.RANDOM_SEED + fold,
                verbosity=0
            )
            model_xgb.fit(X_tr, y_tr)
            fold_oof.append(model_xgb.predict(X_va))
            fold_test.append(model_xgb.predict(X_test_scaled))
            
        # Fold average
        oof_preds[val_idx] = np.mean(fold_oof, axis=0)
        test_preds += np.mean(fold_test, axis=0) / n_splits
        
    metrics = compute_metrics(y_train, oof_preds)
    print_metrics(metrics, prefix="Multimodal GBDT Stack OOF")
    return oof_preds, test_preds

def blend_predictions(transformer_preds: np.ndarray, tabular_preds: np.ndarray, alpha: float = 0.80) -> np.ndarray:
    """
    Blends fine-tuned deep transformer predictions with multimodal tabular ensemble predictions.
    Default weighting allocates 80% to ELECTRA-Large and 20% to multimodal acoustic/syntax features.
    """
    return alpha * transformer_preds + (1.0 - alpha) * tabular_preds
