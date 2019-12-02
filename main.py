import os
from pathlib import Path
from sklearn.dummy import DummyClassifier
import features as ft
import evaluate as ev

category_to_number_dict = {"Whataboutism,Straw_Men,Red_Herring": 0, "Loaded_Language":1, "Name_Calling,Labeling":2, "Flag-Waving":3,
"Exaggeration,Minimisation":4, "Causal_Oversimplification":5, "Repetition":6, "Slogans":7, "Black-and-White_Fallacy":8,
"Appeal_to_Authority":9, "Appeal_to_fear-prejudice":10, "Doubt":11, "Bandwagon,Reductio_ad_hitlerum":12,
"Thought-terminating_Cliches":13 }

number_to_category_dict = {0:"Whataboutism,Straw_Men,Red_Herring", 1:"Loaded_Language", 2:"Name_Calling,Labeling", 3:"Flag-Waving",
4:"Exaggeration,Minimisation", 5:"Causal_Oversimplification", 6:"Repetition", 7:"Slogans", 8:"Black-and-White_Fallacy",
9:"Appeal_to_Authority", 10:"Appeal_to_fear-prejudice", 11:"Doubt", 12:"Bandwagon,Reductio_ad_hitlerum",
13:"Thought-terminating_Cliches"}

data_folder = Path("./Data/datasets/alternative-train-spans-task2/")
label_folder = Path("./Data/datasets/alternative-train-spans-labels-task2/")
prediction_folder = Path("./Data/datasets/alternative-dev-spans-task2/")
gold_label_folder = Path("./Data/datasets/alternative-dev-spans-labels-task2/")

dummyclassifier = DummyClassifier(strategy="most_frequent")

lines = []
labels = []
for filename in os.listdir(data_folder):
    f = open(data_folder / filename, encoding="utf-8")
    for line in f.readlines():
        lines.append(line)
    f.close()

X = ft.get_features(lines)

for filename in os.listdir(label_folder):
    f = open(label_folder / filename, encoding="utf-8")
    for label in f.readlines():
        labels.append(label.replace("\n", ""))

Y = [category_to_number_dict[y] for y in labels]

dummyclassifier.fit(X, Y)

lines=[]
for filename in os.listdir(prediction_folder):
    f = open(prediction_folder / filename, encoding="utf-8")
    for line in f.readlines():
        lines.append(line)
    f.close()
X = ft.get_features(lines)
y_pred = dummyclassifier.predict(X)
gold_labels = ev.read_gold_labels_from_file(gold_label_folder)
y_true = [str(category_to_number_dict[y]) for y in gold_labels]
correct = 0
count = 0
for y in y_true:
    if y == '1':
        correct += 1
    count += 1
print(correct / count)
ev.print_score(y_pred.astype(str), gold_labels)

"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
vectorizer.fit(lines)
vector = vectorizer.transform(lines)
print(vector.shape)
"""
