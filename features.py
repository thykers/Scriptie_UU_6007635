"""
This file is for all the feature-methods
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def get_features(data):
    X = length_of_span(data)
    return X


def length_of_span(data):
    return np.array([len(span) for span in data]).reshape(-1, 1)

def countvectorizer(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit(data)
    return X
