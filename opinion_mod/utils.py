"""Utility functions."""
import os
from pathlib import Path
import shutil
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, Trainer
from torchmetrics.classification import F1Score

from opinion_mod.datamodules.medicine import DataModuleMedicine


def save_best_model(
    trainer: Trainer,
    version: str
):
    """Save best model from trainer. Copy all information."""
    src_dir = trainer.state.best_model_checkpoint
    dest_dir = f"{str(Path(trainer.state.best_model_checkpoint).parent)}" \
               f"\\best\\{version}"

    shutil.rmtree(dest_dir, ignore_errors=True)
    shutil.copytree(src_dir, dest_dir)


def calculate_metrics(
    model: AutoModelForSequenceClassification,
    datamodule: DataModuleMedicine,
    model_params: Dict,
    datamodule_params: Dict,
):
    """Calculate F1 metrics for test dataset.

    To prediction used {model}. Best model is used based on dev_set.
    """
    f1score_macro = F1Score(
        task="multiclass",
        num_classes=model_params['kwargs']['output_dim'],
        average='macro'
    )

    f1score = F1Score(
        task="multiclass",
        num_classes=model_params['kwargs']['output_dim'],
        average=None
    )

    model.to('cpu')
    y_pred_all = []
    y_all = []
    for data in datamodule:

        y = data['labels'].to('cpu')
        tmp_emb = {
            'input_ids': data['input_ids'].to('cpu'),
            'attention_mask': data['attention_mask'].to('cpu'),
            'token_type_ids': data['token_type_ids'].to('cpu')
        }

        y_logits = model(**tmp_emb, labels=y).logits
        y_pred = torch.argmax(y_logits, dim=1)
        y_pred_all.append(y_pred)
        y_all.append(y)

    y_pred_all = torch.hstack(y_pred_all)
    y_true = torch.hstack(y_all)
    metrics = {
        'f1': f1score_macro(y_pred_all, y_true),
        'f1_class': f1score(y_pred_all, y_true),
    }

    metric_file = Path(os.path.join(
        datamodule_params['metric_files_dir'],
        "results.pt"
    ))
    metric_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(metrics, metric_file)
