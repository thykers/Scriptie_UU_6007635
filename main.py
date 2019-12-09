import os
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import features as ft
import evaluate as ev
from sklearn.feature_extraction.text import CountVectorizer

def read_from_file(path):
    spans = []
    for filename in os.listdir(path):
        f = open(path / filename, encoding="utf-8")
        for span in f.readlines():
            spans.append(span.replace("\n", ""))
    f.close()
    return spans

category_to_number_dict = {"Whataboutism,Straw_Men,Red_Herring": 0, "Loaded_Language":1, "Name_Calling,Labeling":2, "Flag-Waving":3,
"Exaggeration,Minimisation":4, "Causal_Oversimplification":5, "Repetition":6, "Slogans":7, "Black-and-White_Fallacy":8,
"Appeal_to_Authority":9, "Appeal_to_fear-prejudice":10, "Doubt":11, "Bandwagon,Reductio_ad_hitlerum":12,
"Thought-terminating_Cliches":13 }

number_to_category_dict = {0:"Whataboutism,Straw_Men,Red_Herring", 1:"Loaded_Language", 2:"Name_Calling,Labeling", 3:"Flag-Waving",
4:"Exaggeration,Minimisation", 5:"Causal_Oversimplification", 6:"Repetition", 7:"Slogans", 8:"Black-and-White_Fallacy",
9:"Appeal_to_Authority", 10:"Appeal_to_fear-prejudice", 11:"Doubt", 12:"Bandwagon,Reductio_ad_hitlerum",
13:"Thought-terminating_Cliches"}

training_data_folder = Path("./Data/datasets/alternative-train-spans-task2/")
training_label_folder = Path("./Data/datasets/alternative-train-spans-labels-task2/")
prediction_folder = Path("./Data/datasets/alternative-dev-spans-task2/")
gold_label_folder = Path("./Data/datasets/alternative-dev-spans-labels-task2/")

logreg_classifier = LogisticRegression(penalty='l2', class_weight='balanced', solver="lbfgs", multi_class='auto')
vectorizer = CountVectorizer()

training_spans = read_from_file(training_data_folder)
vectors = vectorizer.fit_transform(training_spans)
#X = ft.get_features(training_spans)
#X = np.hstack((X, vectors.toarray()))
training_labels = read_from_file(training_label_folder)
Y = np.array([category_to_number_dict[y] for y in training_labels])
logreg_classifier.fit(vectors, Y)

spans = read_from_file(prediction_folder)
vector = vectorizer.transform(spans)
#X = ft.get_features(spans)
#X = np.hstack((X, vector.toarray()))
y_pred = logreg_classifier.predict(vector)

gold_labels = read_from_file(gold_label_folder)
y_true = np.array([category_to_number_dict[x] for x in gold_labels])
ev.print_score(y_pred, y_true)
