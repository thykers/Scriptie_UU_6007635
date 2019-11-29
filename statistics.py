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

category_dict = dict()
total_span_length = 0
span_count = 0
biggest_span = 0
smallest_span = 9999
for article in articles.values():
    for category, start, end in article:
        if category in category_dict:
            category_dict[category] += 1
        else:
            category_dict[category] = 1
        span_length = int(end) - int(start)
        if(span_length >  biggest_span):
            biggest_span = span_length
        elif(span_length < smallest_span):
            smallest_span = span_length
        span_count += 1
        total_span_length += (span_length)

print(total_span_length / span_count)
print(biggest_span)
print(smallest_span)
