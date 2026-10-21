"""
Deep Transformer Models for Spoken Grammar & Fluency Evaluation
==============================================================
Implements sequence classification regressors fine-tuned on spoken transcripts:
1. ELECTRA-Large Discriminator (335M): Pre-trained with Replaced Token Detection (RTD),
   directly modeling grammatical token plausibility and error detection.
2. RoBERTa-Large (355M): Contextual masked language model baseline.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SpeechTextDataset(Dataset):
    """PyTorch Dataset for spoken utterance transcripts."""
    def __init__(self, texts, labels=None, tokenizer=None, max_len=256):
        self.texts = [str(t).strip() if len(str(t).strip()) > 0 else "Speech utterance." for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

def get_transformer_regressor(model_name: str = "google/electra-large-discriminator", freeze_bottom_layers: int = 16):
    """
    Initializes a pre-trained transformer model with a single continuous regression head.
    Optionally freezes bottom transformer layers for stable parameter-efficient fine-tuning.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    # Freeze lower feature extraction layers if specified
    if freeze_bottom_layers > 0:
        encoder = getattr(model, "electra", None) or getattr(model, "roberta", None) or getattr(model, "bert", None)
        if encoder and hasattr(encoder, "encoder"):
            for layer in encoder.encoder.layer[:freeze_bottom_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    return tokenizer, model
