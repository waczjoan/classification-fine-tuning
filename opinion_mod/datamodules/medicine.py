import os
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch


class CustomTextDataset(Dataset):
    def __init__(self, x, labels, out_dict: bool = True):
        self.labels = labels
        self.x = x
        self.device = 'cuda'
        self.out_dict = out_dict

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.x[idx]

        sample = {
            "input_ids": torch.tensor(data.ids),
            "token_type_ids": torch.tensor(data.type_ids),
            "attention_mask": torch.tensor(data.attention_mask),
            "labels": torch.tensor(label).long()
        }

        return sample


class DataModuleMedicine(pl.LightningDataModule):
    def __init__(
        self,
        token_dir: Path,
        batch_size: int = 128,
    ):
        super().__init__()
        self.token_dir = token_dir

        # Defining batch size of our data
        self.batch_size = batch_size

        #
        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def setup(self, stage=None):
        data = {}
        labels = {}
        for dataset_split in ["train", "dev", "test"]:
            path = Path(os.path.join(
                self.token_dir,
                f"{dataset_split}.pt"
            ))
            _data = torch.load(path)
            data[dataset_split] = _data['tokenize']
            labels[dataset_split] = _data['y']

        self.train_data = CustomTextDataset(data['train'], labels['train'])
        self.valid_data = CustomTextDataset(data['dev'], labels['dev'])
        self.test_data = CustomTextDataset(data['test'], labels['test'])

    def train_dataloader(self):
        """ Generating train_dataloader."""
        return DataLoader(self.train_data,
                          batch_size=self.batch_size)

    def dev_dataloader(self):
        """ Generating val_dataloader."""
        return DataLoader(self.valid_data,
                          batch_size=self.batch_size)

    def test_dataloader(self):
        """ Generating test_dataloader."""
        return DataLoader(self.test_data,
                          batch_size=self.batch_size)