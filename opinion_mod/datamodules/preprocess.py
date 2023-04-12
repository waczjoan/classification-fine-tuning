"""Prepare dataset to processed. Split texts and labels."""
import json
from pathlib import Path

from sklearn import preprocessing
import torch


def label_encoder_default():
    labels = [
        'z_minus_m',
        'z_zero',
        'z_plus_m',
        'z_amb'
    ]
    le = preprocessing.LabelEncoder()
    le.fit(labels)

    classes_ = list(le.classes_)
    y_map = {}
    for i in range(len(classes_)):
        y_map[i] = classes_[i]

    label_encoder_file_out = Path("data/preprocessed/LabelEncoder.json")
    label_encoder_file_out.parent.mkdir(parents=True, exist_ok=True)
    with label_encoder_file_out.open("w") as fout:
        json.dump(obj=y_map, fp=fout, indent=4)

    return le


def splitting(lines):
    """Split line to text and label"""
    texts = []
    labels = []
    for i in range(len(lines)):
        text, label = lines[i].split(' __label__')
        texts.append(text)
        labels.append(label[:-1])
    return texts, labels


def prepare(
    data_path: str,
    out_file: Path
):
    """Split texts and labels and save output."""
    with open(data_path, encoding="utf-8") as f:
        lines = f.readlines()

    texts, labels = splitting(lines)

    le = label_encoder_default()
    labels = le.transform(labels)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save((texts, labels), out_file)
