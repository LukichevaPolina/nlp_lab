from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from typing import Any
import logging as log

# TODO add word2vec
def tfidf_embeddings(data) -> Any:
    log.info("Embeddings: recived tfidf embeddings")
    vectorizer = TfidfVectorizer()
    data["statement"] = data["statement"].map(' '.join)
    X = vectorizer.fit_transform(data["statement"])
    if "status" in data.columns:
        enc = LabelEncoder()
        enc.fit(data["status"])
        y = enc.transform(data["status"])

        return X, y
    return X
