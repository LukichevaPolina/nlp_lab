from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from typing import Any
import logging as log
import joblib
import os
import numpy as np


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
        print(X.shape)
        return X

    def fit_transform_y(self, data):
        enc = LabelEncoder()
        enc.fit(data)
        y = enc.transform(data)

        return y


class Word2vecVectorizer:
    def __init__(self, checkpoint_path="checkpoints/word2vec.pkl"):
        self._is_trained = False
        self._checkpoint_path = checkpoint_path
        if os.path.isfile(checkpoint_path):
            self._vectorizer = joblib.load(self._checkpoint_path)
            self._is_trained = True

    def fit_X(self, data):
        if not self._is_trained:
            data = data.tolist()
            self._vectorizer = Word2Vec(
                sentences=data, vector_size=59406, window=7)
            self._is_trained = True
            joblib.dump(self._vectorizer, self._checkpoint_path)

    def get_document_vector(self, document):
        word_vectors = [self._vectorizer.wv[word]
                        for word in document if word in self._vectorizer.wv]
        if not word_vectors:
            return np.zeros(self._vectorizer.vector_size)
        return np.mean(word_vectors, axis=0)

    def transform_X(self, data):
        data = data.tolist()

        X = [self.get_document_vector(doc) for doc in data]
        X = np.array([np.array(el) for el in X])

        return X

    def fit_transform_y(self, data):
        enc = LabelEncoder()
        enc.fit(data)
        y = enc.transform(data)

        return y
