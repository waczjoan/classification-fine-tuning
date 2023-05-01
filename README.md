# 
Text classification using transformer fine-tuning.
The data relates to patients' opinions on medical services, mainly doctors' opinions. There are four classes:
- neutral opinion 
- positive
- negative
- ambivalent

The description of the corpus and the task refers to the following paper:


[Multi-Level Sentiment Analysis of PolEmo 2.0: Extended Corpus of Multi-Domain Consumer Reviews](https://aclanthology.org/K19-1092) (Kocoń et al., CoNLL 2019)

The repository was created mainly as a base and to show how to easily use transformer.Trainer and AutoModelForSequenceClassification. 

### Setup env
```bash
$ python3.9 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

If you do not familiar with dvc, and you do not want to be, you can just find data described in paper, or use your own, and put them in `data\{dataset_name}\raw\{split}.txt`.

File should contain sentences with labels:
```bash
Example sentence . __label__z_plus_m
Different sentence . __label__z_minus_m
```