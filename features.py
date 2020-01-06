"""
This file is for all the feature-methods
"""
import numpy as np
import dictionaries as dt
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer

vector = None
vectorizer = CountVectorizer()

def get_features(data, path_to_articles, filenames_list, is_training):
    union = FeatureUnion([("A", length_of_span(data)), ("B", number_of_words(data, path_to_articles, filenames_list))])
    """
    if is_training:
        X = initialize_vectoriser(data)
    else:
        X = transform_data_to_countvector(data)

    X = np.hstack((X, relative_capitals_frequency(data, path_to_articles, filenames_list)))
    X = np.hstack((X, relative_punctuation_frequency(data, path_to_articles, filenames_list)))
    X = np.hstack((X, average_number_of_words_per_sentence(data, path_to_articles, filenames_list)))
    X = np.hstack((X, number_of_words(data, path_to_articles, filenames_list)))
    X = np.hstack((X, length_of_span(data)))

    return X
    """
    return union

def relative_punctuation_frequency(data, path, filenames):
    punctuation_list = ['.', ',', '?', '!', ';', ':', '\"', '\'', '[', ']', '(', ')', '/', '{', '}']
    dict_rel_punc_frequency = dict()
    X = []
    for line, filename in zip(data, filenames):
        if(filename in dict_rel_punc_frequency.keys()):
            X.append(dict_rel_punc_frequency[filename])
        else:
            f = open(path / filename, encoding="utf-8")
            text = f.read()
            punctuation_count = 0
            for n, c in enumerate(text):
                if(c in punctuation_list):
                    punctuation_count += 1
            dict_rel_punc_frequency[filename] = punctuation_count / n
            X.append(punctuation_count / n)
    return np.array(X).reshape(-1, 1)

def averaged_word_length(data, path, filenames):
    dict_of_avg_word_len = dict()
    X = []
    for line, filename in zip(data, filenames):
        if(filename in dict_of_avg_sent_len.keys()):
            X.append(dict_of_avg_sent_len[filename])
        else:
            character_count = 0
            f = open(path / filename, encoding="utf-8")
            text = f.read()
            words = text.split(" ")
            for n, word in enumerate(words):
                character_count += len(word.replace("\n", ""))
            dict_of_avg_word_len[filename] = character_count / n
            X.append(character_count / n)
    return np.array(X).reshape(-1, 1)

def length_of_span(data):
    return np.array([len(span) for span in data]).reshape(-1, 1)

def average_number_of_words_per_sentence(data, path, filenames):
    dict_of_avg_sent_len = dict()
    X = []
    for line, filename in zip(data, filenames):
        if(filename in dict_of_avg_sent_len.keys()):
            X.append(dict_of_avg_sent_len[filename])
        else:
            total_words = 0
            f = open(path / filename, encoding="utf-8")
            lines = f.readlines()
            for n, line in enumerate(lines):
                total_words += len(line)
            dict_of_avg_sent_len[filename] = total_words / n
            X.append(total_words / n)
    return np.array(X).reshape(-1, 1)

def number_of_words(data, path, filenames):
    """
    Number of Words in the Article
    """
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

def relative_capitals_frequency(data, path, filenames):
    dict_rel_cap_frequency = dict()
    X = []
    for line, filename in zip(data, filenames):
        if(filename in dict_rel_cap_frequency.keys()):
            X.append(dict_rel_cap_frequency[filename])
        else:
            f = open(path / filename, encoding="utf-8")
            text = f.read()
            capital_count = 0
            for n, c in enumerate(text):
                if(c.isupper()):
                    capital_count += 1
            dict_rel_cap_frequency[filename] = capital_count / n
            X.append(capital_count / n)
    return np.array(X).reshape(-1, 1)

def initialize_vectoriser(data):
    global vector
    vector = vectorizer.fit_transform(data)
    return np.array(vector).reshape(-1, 1)

def transform_data_to_countvector(data):
    return vectorizer.transform(data)

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
