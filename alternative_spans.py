from pathlib import Path
import os

data_folder_articles = Path("./Data/datasets/train-articles/")
data_folder_labels = Path("./Data/datasets/train-labels-task2-technique-classification/")
data_folder_alternative_articles = Path("./Data/datasets/alternative-train-spans-task2/")
data_folder_alternative_labels = Path("./Data/datasets/alternative-train-spans-labels-task2/")

articles = dict()
for filename in os.listdir(data_folder_labels):
    f = open(data_folder_labels / filename, encoding="utf-8")
    lines = []
    for line in f:
        x = line.split()
        lines.append(x[1:])
    articles[filename[7:16]] = lines
    f.close()

for filename in os.listdir(data_folder_articles):
    f = open(data_folder_articles / filename, encoding="utf-8")
    count = 0
    train_article_lines = f.readlines()
    article = articles[filename[7:16]]
    sorted_article = sorted(article, key = lambda tup: int(tup[1]))
    newfile = []
    newlabel = []
    text = ""
    for line in train_article_lines:
        text += line
    for category, start, end in sorted_article:
        newfile.append(text[int(start):int(end)].replace("\n", " ")+"\n")
        newlabel.append(category+"\n")
    f.close()

    new_name = "alternative_spans_" + str(filename)
    new_filename_label = "aternative_spans_labels_" + str(filename)

    w = open(data_folder_alternative_articles / new_name, "x", encoding="utf-8")
    w.writelines(newfile)
    w.close()

    w = open(data_folder_alternative_labels / new_filename_label, "x", encoding="utf-8")
    w.writelines(newlabel)
    w.close()
