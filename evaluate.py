import os
import numpy as np
from pathlib import Path
import sklearn.metrics as skm
import dictionaries as dt

def read_gold_labels_from_file(path):
    gold_labels = []
    for filename in os.listdir(path):
        f = open(path / filename, encoding="utf-8")
        for gold_label in f.readlines():
            gold_labels.append(gold_label.replace("\n", ""))
    return gold_labels



def print_score(y_pred, y_true):
    print()
    print("Precision:")
    print()
    for n, value in enumerate(skm.precision_score(y_true, y_pred, average=None)):
        print("{:40s}: {:5f}".format(dt.number_to_category_dict[n], value))
    print()
    print()
    print("Recall:")
    print()
    for n, value in enumerate(skm.recall_score(y_true, y_pred, average=None)):
        print("{:40s}: {:5f}".format(dt.number_to_category_dict[n], value))
    print()
    print("F1-score: {}".format(skm.f1_score(y_true, y_pred, average='micro')))
    print()
    print(skm.confusion_matrix(y_true, y_pred))
