"""Tokenize data and save results."""
from pathlib import Path

import click
import yaml

from opinion_mod.datamodules.medicine import DataModuleMedicine
from opinion_mod.models.transformer_finetuning import transformer_classification
from opinion_mod.utils import calculate_metrics


@click.command()
@click.option(
    "--model_name",
    help="Name of the model from config.",
    type=str,
    default="allegro_herbert",
)
@click.option(
    "--hparams_path",
    help="Path to config file.",
    type=click.Path(exists=True, path_type=Path),
    default=Path("experiments/configs/models.yaml"),
)
def main(
    model_name: str,
    hparams_path: str,
):
    """Tokenize data and save results. Main function."""
    with open(hparams_path, "r") as fin:
        hparams = yaml.safe_load(fin)[model_name]

    datamodule_params = hparams["datamodule"]

    datamodule = DataModuleMedicine(
        token_dir=datamodule_params["data_dir"],
        batch_size=datamodule_params["batch_size"],
    )
    datamodule.setup()

    model = transformer_classification(
        datamodule=datamodule,
        model_params=hparams["model"],
    )

    calculate_metrics(
        model=model,
        datamodule=datamodule.test_dataloader(),
        model_params=hparams["model"],
        datamodule_params=hparams["datamodule"],
    )


if __name__ == "__main__":
    main()
