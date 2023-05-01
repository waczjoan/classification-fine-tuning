"""Classification using AutoModelForSequenceClassification."""
from typing import Dict

from opinion_mod.datamodules.medicine import DataModuleMedicine
from transformers import AutoModelForSequenceClassification
from opinion_mod.trainer.trainer import transformer_trainer
from opinion_mod.utils import save_best_model


def transformer_classification(
    datamodule: DataModuleMedicine,
    model_params: Dict,
):
    """Classification task. Load data nad train model. Save best model."""

    model = AutoModelForSequenceClassification.from_pretrained(
        model_params["kwargs"]['model_name'],
        num_labels=model_params["kwargs"]['output_dim']
    ).cuda()

    trainer = transformer_trainer(
        model_params["kwargs"]['model_name'],
        model,
        datamodule.train_data,
        datamodule.valid_data,
        model_params['max_epochs']
    )
    trainer.train()
    save_best_model(
        trainer=trainer,
        version=model_params["kwargs"]['model_name']
    )

    return model
