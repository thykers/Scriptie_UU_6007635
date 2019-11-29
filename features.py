"""
This file is for all the feature-methods
"""
def get_features(data):
    X = length_of_span(data)
    return X


def length_of_span(data):
    return [len(span) for span in data]
