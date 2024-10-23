from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from typing import Any
import logging as log

# TODO add word2vec

# TODO add saving of vectorizer to apply for input(just a sentence) in infer or eval
# return vectorizer and save it in text classification pipeline class
def tfidf_embeddings(data) -> Any:
    log.info("Embeddings: recived tfidf embeddings")
    vectorizer = TfidfVectorizer()
    data["statement"] = data["statement"].map(' '.join)
    X = vectorizer.fit_transform(data["statement"])
    if "status" in data.columns:
        enc = LabelEncoder()
        enc.fit(data["status"])
        y = enc.transform(data["status"])
        print(f"{enc.transform(["Anxiety"])=}")
        print(f"{enc.transform(["Depression"])=}")
        print(f"{enc.transform(["Normal"])=}")
        print(f"{enc.transform(["Suicidal"])=}")
        print(f"{enc.transform(["Stress"])=}")
        print(f"{enc.transform(["Bipolar"])=}")
        print(f"{enc.transform(["Personality disorder"])=}")
        return X, y
    return X
