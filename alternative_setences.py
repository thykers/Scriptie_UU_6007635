from pathlib import Path
import os

data_folder_articles = Path("./Data/datasets/train-articles/")
data_folder_labels = Path("./Data/datasets/train-labels-task2-technique-classification/")
data_folder_alternative_articles = Path("./Data/datasets/alternative-train-articles-task2/")
data_folder_alternative_labels = Path("./Data/datasets/alternative-train-labels-task2/")

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
    same_line = False
    if sorted_article:
        category, start, end = sorted_article[0]
        for line in train_article_lines:
            if len(line) > 1:   #geen lege regels
                count += len(line)
                while count >= int(start) and sorted_article:
                    if same_line:
                        temp = newlabel.pop()
                        newlabel.append(temp + " " + category)
                    else:
                        newfile.append(line)
                        newlabel.append(category)
                        same_line = True
                    sorted_article.pop(0)
                    if sorted_article:
                        category, start, end = sorted_article[0]
                    else:
                        break

                same_line = False
    f.close()
    new_name = "alternative_" + str(filename)
    new_filename_label = "aternative_labels_" + str(filename)
    #"""
    w = open(data_folder_alternative_articles / new_name, "x", encoding="utf-8")
    for line in newfile:
        w.write(line)
    w.close()
    
    w = open(data_folder_alternative_labels / new_filename_label, "x", encoding="utf-8")
    for category in newlabel:
        w.write(category+"\n")
    w.close()
    #"""
