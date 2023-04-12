"""Tokenize data and save results."""
from pathlib import Path
import os

import click
import torch
from tqdm import tqdm

from opinion_mod.datamodules.preprocess import prepare
from opinion_mod.tokenizer.tokenizer import tokenize_text


@click.command()
@click.option(
    "--model_name",
    help="Name of the model used to tokenized.",
    type=str,
    default="allegro/herbert-base-cased",
)
@click.option(
    "--dataset_name",
    help="Name of the used dataset to experiments.",
    type=str,
    default="medicine",
)
@click.option(
    "--preprocessed_files_dir",
    help="Directory to preprocessed data.",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/preprocessed"),
)
@click.option(
    "--tokenized_files_dir",
    help="Directory to tokenized data.",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/tokenized"),
)
def main(
    model_name: str,
    dataset_name: str,
    preprocessed_files_dir: Path,
    tokenized_files_dir: Path
):
    """Tokenize data and save results. Main function."""
    tokenize_data(
        model_name=model_name,
        dataset_name=dataset_name,
        preprocessed_file_dir=preprocessed_files_dir,
        tokenized_files_dir=tokenized_files_dir
    )


def tokenize_data(
    model_name: str,
    dataset_name: str,
    preprocessed_file_dir: Path,
    tokenized_files_dir: Path
):
    """Tokenize data and save results.

    Args:
    model_name: language model used to create text tokens.
    dataset_name: name of the used sentence corpora in data/raw.
    preprocessed_file_dir: Dir to tokenized texts.
    tokenized_files_dir: Dir to tokenized texts.

    """

    for dataset_split in tqdm(["train", "dev", "test"]):

        out_file = Path(os.path.join(
            preprocessed_file_dir,
            model_name.replace('/', '-'),
            f"{dataset_name}_{dataset_split}.pt"
        ))

        if ~check_if_file_exists(out_file):
            prepare(
                f'data/raw/{dataset_name}.sentence.{dataset_split}.txt',
                out_file=out_file
            )
        texts, labels = torch.load(out_file)
        inputs = tokenize_text(texts, model_name)
        dataset = {
            'tokenize': inputs,
            'y': labels
        }

        dataset_file = Path(os.path.join(
            tokenized_files_dir,
            model_name.replace('/', '-'),
            f"{dataset_name}_{dataset_split}.pt"
        ))
        dataset_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, dataset_file)


def check_if_file_exists(
    file: str
):
    """Check if file exists. Return True if file exists."""
    path = Path(file)
    return path.is_file()


if __name__ == "__main__":
    main()
