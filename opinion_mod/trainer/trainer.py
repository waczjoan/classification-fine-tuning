"""Get default trainer used fine-tuning transformer."""
from transformers import Trainer, TrainingArguments


def transformer_trainer(
    model_name,
    model,
    train_data,
    dev_data,
    num_train_epochs=10
) -> Trainer:
    """Get default trainer used fine-tuning transformer.

    Used TrainingArguments to define train hyperparameters.
    """
    training_args = TrainingArguments(
        output_dir=f"output/checkpoints/{model_name.replace('/','-')}/",
        num_train_epochs=num_train_epochs,
        learning_rate=0.00005,
        weight_decay=0.01,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
    )

    return trainer
