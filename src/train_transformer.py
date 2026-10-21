"""
Transformer Regressor Cross-Validation Training Pipeline
=======================================================
Performs 5-fold stratified cross-validation fine-tuning of sequence classification models
(ELECTRA-Large / RoBERTa-Large) for continuous grammar proficiency scoring.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import get_cosine_schedule_with_warmup

sys.path.append(str(Path(__file__).resolve().parent))
import config
from utils import compute_metrics, print_metrics
from models import SpeechTextDataset, get_transformer_regressor

def train_epoch(model, loader, optimizer, scheduler, criterion, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.squeeze(-1)
        loss = criterion(preds, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds.extend(outputs.logits.squeeze(-1).cpu().numpy().tolist())
    return np.array(preds)

def run_cross_validation(
    model_name="google/electra-large-discriminator",
    epochs=4,
    batch_size=8,
    lr_backbone=1.5e-5,
    lr_head=5e-5,
    device=None
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {model_name}")

    # Load data
    train_df = pd.read_csv(config.TRAIN_TRANSCRIPTS_CSV)
    test_df = pd.read_csv(config.TEST_TRANSCRIPTS_CSV)

    # Filter out empty/silent audio outliers
    valid_mask = (train_df[config.TARGET_COL] > 0.0) & (train_df["transcript"].fillna("").str.strip() != "")
    train_valid = train_df[valid_mask].reset_index(drop=True)
    y_valid = train_valid[config.TARGET_COL].values

    # Stratified K-Fold by target quintiles
    target_bins = pd.qcut(y_valid, q=config.N_FOLDS, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

    oof_preds = np.zeros(len(train_valid))
    test_preds = np.zeros(len(test_df))

    criterion = nn.SmoothL1Loss(beta=0.2)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_valid, target_bins)):
        print(f"\n--- Fold {fold + 1}/{config.N_FOLDS} ---")
        train_fold = train_valid.iloc[train_idx].reset_index(drop=True)
        val_fold = train_valid.iloc[val_idx].reset_index(drop=True)

        tokenizer, model = get_transformer_regressor(model_name=model_name, freeze_bottom_layers=16)
        model.to(device)

        train_ds = SpeechTextDataset(train_fold["transcript"].values, train_fold[config.TARGET_COL].values, tokenizer)
        val_ds = SpeechTextDataset(val_fold["transcript"].values, val_fold[config.TARGET_COL].values, tokenizer)
        test_ds = SpeechTextDataset(test_df["transcript"].values, tokenizer=tokenizer)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

        # Differential learning rates
        classifier_params = [p for n, p in model.named_parameters() if "classifier" in n]
        encoder_params = [p for n, p in model.named_parameters() if "classifier" not in n and p.requires_grad]

        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": lr_backbone, "weight_decay": 0.01},
            {"params": classifier_params, "lr": lr_head, "weight_decay": 0.01}
        ])

        total_steps = len(train_loader) * epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

        best_val_r = -1.0
        best_fold_preds = None

        for epoch in range(1, epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
            val_preds = evaluate(model, val_loader, device)
            val_r, _ = pearsonr(val_fold[config.TARGET_COL].values, val_preds)
            val_rmse = np.sqrt(np.mean((val_fold[config.TARGET_COL].values - val_preds) ** 2))
            print(f"Epoch {epoch}/{epochs} | Train Loss: {loss:.4f} | Val r: {val_r:.4f} | Val RMSE: {val_rmse:.4f}")

            if val_r > best_val_r:
                best_val_r = val_r
                best_fold_preds = val_preds

        oof_preds[val_idx] = best_fold_preds
        test_preds += evaluate(model, test_loader, device) / config.N_FOLDS

    metrics = compute_metrics(y_valid, oof_preds)
    print_metrics(metrics, prefix=f"{model_name} OOF")

    return oof_preds, test_preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer grammar regressor.")
    parser.add_argument("--model", type=str, default="google/electra-large-discriminator")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    run_cross_validation(model_name=args.model, epochs=args.epochs, batch_size=args.batch_size)
