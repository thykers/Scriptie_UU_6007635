import os
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import features as ft
import evaluate as ev
from sklearn.feature_extraction.text import CountVectorizer
import dictionaries as dt

def read_from_file(path):
    spans = []
    filename_of_article = []
    for filename in os.listdir(path):
        f = open(path / filename, encoding="utf-8")
        for span in f.readlines():
            spans.append(span.replace("\n", ""))
            filename_of_article.append(filename[18:])
    f.close()
    return spans, filename_of_article

training_data_folder = Path("./Data/datasets/alternative-train-spans-task2/")
training_label_folder = Path("./Data/datasets/alternative-train-spans-labels-task2/")
prediction_folder = Path("./Data/datasets/alternative-dev-spans-task2/")
gold_label_folder = Path("./Data/datasets/alternative-dev-spans-labels-task2/")
article_folder = Path("./Data/datasets/train-articles/")

logreg_classifier = LogisticRegression(penalty='l2', class_weight='balanced', solver="lbfgs", multi_class='auto')

if __name__ == "__main__":

    training_spans, filename_of_article_per_span = read_from_file(training_data_folder)
    X = ft.get_features(training_spans, article_folder, filename_of_article_per_span, True)

    training_labels, filename_of_label = read_from_file(training_label_folder)
    Y = np.array([dt.category_to_number_dict[y] for y in training_labels])

    logreg_classifier.fit(X, Y)

    spans, filenames_of_article = read_from_file(prediction_folder)
    X = ft.get_features(spans, article_folder, filenames_of_article, False)

    y_pred = logreg_classifier.predict(X)
    gold_labels, filename_of_label = read_from_file(gold_label_folder)
    y_true = np.array([dt.category_to_number_dict[x] for x in gold_labels])

    ev.print_score(y_pred, y_true)

#X = np.hstack((X, vector.toarray()))
#X = np.hstack((X, vectors.toarray()))
