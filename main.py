import os
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

category_to_number_dict = {"Whataboutism,Straw_Men,Red_Herring": 0, "Loaded_Language":1, "Name_Calling,Labeling":2, "Flag-Waving":3,
"Exaggeration,Minimisation":4, "Causal_Oversimplification":5, "Repetition":6, "Slogans":7, "Black-and-White_Fallacy":8,
"Appeal_to_Authority":9, "Appeal_to_fear-prejudice":10, "Doubt":11, "Bandwagon,Reductio_ad_hitlerum":12,
"Thought-terminating_Cliches":13 }

number_to_category_dict = {0:"Whataboutism,Straw_Men,Red_Herring", 1:"Loaded_Language", 2:"Name_Calling,Labeling", 3:"Flag-Waving",
4:"Exaggeration,Minimisation", 5:"Causal_Oversimplification", 6:"Repetition", 7:"Slogans", 8:"Black-and-White_Fallacy",
9:"Appeal_to_Authority", 10:"Appeal_to_fear-prejudice", 11:"Doubt", 12:"Bandwagon,Reductio_ad_hitlerum",
13:"Thought-terminating_Cliches"}

dummyclassifier = DummyClassifier(strategy="most_frequent")
logreg = LogisticRegression()
vectorizer = CountVectorizer()
data_folder = Path("./Data/datasets/alternative-train-spans-task2/")
label_folder = Path("./Data/datasets/alternative-train-spans-labels-task2/")
lines = []
labels = []
for filename in os.listdir(data_folder):
    f = open(data_folder / filename, encoding="utf-8")
    for line in f.readlines():
        lines.append(line)
    f.close()

X = [x for x in range(len(lines))]
for filename in os.listdir(label_folder):
    f = open(label_folder / filename, encoding="utf-8")
    for label in f.readlines():
        labels.append(label.replace("\n", ""))
Y = [category_to_number_dict[y] for y in labels]
dummyclassifier.fit(X, Y)
predictions = dummyclassifier.predict([x for x in range(8)])
for p in predictions:
    print(number_to_category_dict[p])

vectorizer.fit(lines)
vector = vectorizer.transform(lines)
print(vector.shape)
