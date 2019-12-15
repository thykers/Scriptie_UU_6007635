"""
This file is for all the feature-methods
"""
import numpy as np
import dictionaries as dt
from sklearn.feature_extraction.text import CountVectorizer

def get_features(data, path, filenames_list):
    X = number_of_words(data, path, filenames_list)
    X = np.hstack((X, length_of_span(data)))
    return X

def length_of_span(data):
    return np.array([len(span) for span in data]).reshape(-1, 1)

def number_of_words(data, path, filenames):
    article_dict_of_word_count = dict()
    X = []
    for line, filename in zip(data, filenames):
        if(filename in article_dict_of_word_count.keys()):
            X.append(article_dict_of_word_count[filename])
        else:
            f = open(path / filename, encoding="utf-8")
            text = f.read()
            words = text.split(" ")
            article_dict_of_word_count[filename] = len(words)
            X.append(len(words))
    return np.array(X).reshape(-1, 1)

def initialize_vectoriser(data):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(data)

def transform_data_to_countvector(data, vector):
    return vector.transform(data)

def emotion_annontation(data):
    X = []
    for span in data:
        data_words = span.split(" ")
        annotation = 0
        for word in data_words:
            if(word in dt.NRC_pos_neg.keys()):
                if dt.NRC_pos_neg[word] == "postive":
                    annotation += 1
                else:
                    annotation -= 1
        if(annotation > 0 ):
            X.append(1)
        elif(annotation < 0):
            X.append(-1)
        else:
            X.append(0)
    return np.array(X).reshape(-1, 1)

def emotion_category(data):
    data_words = data.split(" ")
    for word in data_words:
        if(word in dt.NRC_lexicon):
            None

if __name__ == "__main__":
    print(emotion_annontation("The next transmission could be more pronounced or stronger abandon"))
