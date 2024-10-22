from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from typing import Any
import logging as log
import joblib

class TFIDFVectorizer:
    def __init__(self, checkpoint_path="checkpoints/tfidf.pkl"):
        self._vectorizer = TfidfVectorizer()
        self._is_trained = False
        self._checkpoint_path = checkpoint_path

    def fit_X(self, data):
        data = data.map(' '.join)
        self._vectorizer.fit(data)
        self._is_trained = True
        joblib.dump(self._vectorizer, self._checkpoint_path)

    def transform_X(self, data):
        data = data.map(' '.join)
        if not self._is_trained:
            self._vectorizer = joblib.load(self._checkpoint_path)
            self._is_trained = True

        X = self._vectorizer.transform(data)
        return X

    def fit_transform_y(self, data):
        enc = LabelEncoder()
        enc.fit(data)
        y = enc.transform(data)

        print(f"{enc.transform(["Anxiety"])=}")
        print(f"{enc.transform(["Depression"])=}")
        print(f"{enc.transform(["Normal"])=}")
        print(f"{enc.transform(["Suicidal"])=}")
        print(f"{enc.transform(["Stress"])=}")
        print(f"{enc.transform(["Bipolar"])=}")
        print(f"{enc.transform(["Personality disorder"])=}")

        return y
