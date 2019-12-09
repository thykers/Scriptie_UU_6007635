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

def initialize_vectoriser(data):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(data)

def transform_data_to_countvector(data, vector):
    return vector.transform(data)
