import os
import numpy as np
from pathlib import Path
import sklearn.metrics as skm

def read_gold_labels_from_file(path):
    gold_labels = []
    for filename in os.listdir(path):
        f = open(path / filename, encoding="utf-8")
        for gold_label in f.readlines():
            gold_labels.append(gold_label.replace("\n", ""))
    return gold_labels



def print_score(y_pred, y_true):
    print(skm.f1_score(y_true, y_pred, average='macro', labels=np.unique(y_pred)))
