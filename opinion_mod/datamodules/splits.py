"""Create evaluated dataset used selected test and dev set."""
from typing import Dict
from tqdm import tqdm
import torch


def prepare_splits(
    input_file: str,
    n_splits: int,
    test_idx: int,
    dev_idx: int,
) -> (Dict, Dict):
    """Create dataset.

    To create test set use test_idx fold,
    to create dev det use dev_idx.
    """
    dataset_folds = torch.load(input_file)

    train_idx = list(range(n_splits))
    train_idx.remove(test_idx)
    train_idx.remove(dev_idx)

    dataset = {}
    labels = {}
    for dataset_split, data_idx in tqdm(zip(
            ("train", "dev", "test"),
            (train_idx, [test_idx], [dev_idx]),
    )):
        input_ids = [dataset_folds[i]['tokenize']['input_ids']
                     for i in data_idx]
        token_type_ids = [dataset_folds[i]['tokenize']['token_type_ids']
                          for i in data_idx]
        attention_mask = [dataset_folds[i]['tokenize']['attention_mask']
                          for i in data_idx]

        data_y = [dataset_folds[i]['y'] for i in data_idx]

        dataset[dataset_split] = {
            'input_ids': torch.cat(input_ids),
            'token_type_ids': torch.cat(token_type_ids),
            'attention_mask': torch.cat(attention_mask)
        }

        labels[dataset_split] = torch.cat(data_y)

    return dataset, labels
