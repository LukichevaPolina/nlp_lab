import sys
import argparse
import random
from pathlib import Path
from enum import Enum
from string import punctuation

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
import json

from typing import Any, Dict

from src.models.svc import SVC
from src.models.decision_tree import DecisionTree
from src.utils.get_data import get_data
from src.eda.rendering.data import plot_data_information
from src.eda.preprocessing.preprocessing import (
    drop_nan, remove_punctuation, remove_digits, remove_stop_words,
    tokenize, stemming, lemmatization
)
from src.eda.preprocessing.embeddings import TFIDFVectorizer
from src.eda.rendering.statistics import (
    class_features_distribution, plot_class_distribution
)

SEED = 4200
random.seed(SEED)
np.random.seed(SEED)
# TODO add torch random seed

class Preprocessor(Enum):
    REMOWE_ALL_STOP_WORDS_AND_PUNCTUATION=1

class ClassBalancer(Enum):
    CLASS_WEIGHT=1
    UNION=2 # experemental union some classes into one

class Embeddings(Enum):
    TFIDF=1
    WORD2VEC=2

class Algorithm(Enum):
    SVM=1
    DECISION_TREE=2
    LSTM=3
    CNN=4

class Mode(Enum):
    TRAIN=1
    EVAL=2
    INFER=3

TARGET2ENUM = {
    "cnn": Algorithm.CNN,
    "lstm": Algorithm.LSTM,
    "decision_tree": Algorithm.DECISION_TREE,
    "svm": Algorithm.SVM,
    "tfidf": Embeddings.TFIDF,
    "word2vec": Embeddings.WORD2VEC,
    "class_weight": ClassBalancer.CLASS_WEIGHT,
    "union": ClassBalancer.UNION,
    "remove_all": Preprocessor.REMOWE_ALL_STOP_WORDS_AND_PUNCTUATION,
    "train": Mode.TRAIN,
    "eval": Mode.EVAL,
    "infer": Mode.INFER
}

ENUM2TARGET = dict(zip(TARGET2ENUM.values(), TARGET2ENUM.keys()))

class TextClassificationPipeline:
    def __init__(
            self, 
            dataset_path: str,
            algorithm: str,
            embeddings: str,
            class_balancer: str,
            preprocessor: str,
            mode: str,

        ) -> None:
        self._dataset = get_data(path=dataset_path)

        # TODO add dict for classic algorithm results
        # TODO add dict for dl algorithm results
        # TODO add metrics
        self._algorithm = self.str2enum(algorithm)
        self._preprocessor = self.str2enum(preprocessor)
        self._embedder = self.str2enum(embeddings)
        self._checkpoint_path = self.get_checkpoint(algorithm, embeddings, class_balancer, preprocessor)
        self._mode = self.str2enum(mode)
        self._preprocessed_data = None
        self._vectorized_data = None
        self._metrics = {}
    
    def run(self) -> None:
        self.preprocess()

        if self._mode == Mode.INFER:
            self.infer()
        else:
            X_train, X_test, y_train, y_test = self.vectorize()
            if self._mode == Mode.TRAIN:
                self.train(X_train, y_train)
                self.eval(X_test, y_test)
            elif self._mode == Mode.EVAL:
                self.eval(X_test, y_test)
            else:
                ValueError("Wrong mode")

    def train(self, X_train, y_train) -> None:
        match self._algorithm:
            case Algorithm.SVM:
                svc = SVC(self._checkpoint_path)
                svc.train(X_train, y_train)
            case Algorithm.DECISION_TREE:
                dt = DecisionTree(self._checkpoint_path)
                dt.train(X_train, y_train)
            case Algorithm.CNN:
                raise NotImplementedError
            case Algorithm.LSTM:
                raise NotImplementedError
            case _:
                raise ValueError(f"Given algorithm: {target} does not exist")

    def eval(self, X, y) -> None:
        y_pred = self.infer(X)
        f1 = f1_score(y, y_pred, average=None)
        self._metrics["f1-score"] = dict((i,j) for i,j in enumerate(f1))
        self._metrics["f1-score-weighted"] = f1_score(y, y_pred, average="weighted")
        self._metrics["balanced_accuracy_score"] = balanced_accuracy_score(y, y_pred)

        print("Classification report:")
        print(self._metrics)

        self.save_results()

    def infer(self, X) -> None:
        if not self._checkpoint_path.exists():
            raise RuntimeError(f"The checkpoint: {str(self._checkpoint_path)} does not exist, train first")

        if self._algorithm == Algorithm.SVM or self._algorithm == Algorithm.DECISION_TREE:
            model = joblib.load(self._checkpoint_path)
            y = model.predict(X)
            return y
        else:
            NotImplementedError


    def preprocess(self) -> pd.DataFrame:
        if self._preprocessor == Preprocessor.REMOWE_ALL_STOP_WORDS_AND_PUNCTUATION:
            preprocess_data = drop_nan(self._dataset)
            plot_class_distribution(preprocess_data)
            class_features_distribution(preprocess_data, "Length distribution", lambda x: len(x), "length")
            class_features_distribution(preprocess_data, "Punctuation length distribution", lambda x: len([let for let in x if let in punctuation]), "punctuation_length")
            class_features_distribution(preprocess_data, "Digit length distribution", lambda x: len([let for let in x if let.isdigit()]), "digit_length")
            preprocess_data = remove_punctuation(preprocess_data)
            preprocess_data = remove_digits(preprocess_data)
            preprocess_data = remove_stop_words(preprocess_data)
            preprocess_data = tokenize(preprocess_data)
            preprocess_data = stemming(preprocess_data)
            preprocess_data = lemmatization(preprocess_data)
            plot_data_information(preprocess_data)
            self._preprocessed_data = preprocess_data
        else:
            raise NotImplementedError

    def vectorize(self) -> Any:
        if self._embedder == Embeddings.TFIDF:
            vectorizer = TFIDFVectorizer()
            if self._mode == Mode.INFER:
                return vectorizer.transform_X(self._preprocessed_data["statement"])
            else:
                # split dataset on train and test
                X_train, X_test, y_train, y_test = train_test_split(self._preprocessed_data["statement"], self._preprocessed_data["status"],
                                                                    test_size=0.3, random_state=42, stratify=self._preprocessed_data["status"])

                vectorizer.fit_X(X_train)

                X_train = vectorizer.transform_X(X_train)
                X_test = vectorizer.transform_X(X_test)

                y_train = vectorizer.fit_transform_y(y_train)
                y_test = vectorizer.fit_transform_y(y_test)

                return X_train, X_test, y_train, y_test
        else:
            raise NotImplementedError

    def save_results(self) -> None:
        file = "results/" + self.enum2str(self._algorithm) + ".json"
        with open(file, 'w') as fp:
            json.dump(self._metrics, fp)

    def str2enum(self, target: str) -> Algorithm:
        try:
            return TARGET2ENUM[target]
        except:
            raise ValueError(f"Given algorithm: {target} does not exist")
        
    def enum2str(self, target: Algorithm) -> str:
        try:
            return ENUM2TARGET[target]
        except:
            raise ValueError(f"Given algorithm: {target} does not exist")

    def get_checkpoint_name(
        self, 
        algorithm: str, 
        embeddigns: str,
        class_balancer: str,
        preprocessor: str
    ) -> str:
        if algorithm in ["svm", "decision_tree"]:
            return f"{algorithm}_{embeddigns}_{class_balancer}_{preprocessor}.pkl"
        elif algorithm in ["lstm", "cnn"]:
            return f"{algorithm}_{embeddigns}_{class_balancer}_{preprocessor}.pt"
        else:
            raise ValueError(f"Given algorithm: {algorithm} does not exist")
    
    def get_checkpoint(
        self, 
        algorithm: str,
        embeddings: str, 
        class_balancer: str, 
        preprocessor: str
    ) -> Path:
        checkpoints_dir = Path("checkpoints/")
        checkpoint_name = Path(self.get_checkpoint_name(algorithm, embeddings, class_balancer, preprocessor))
        return (checkpoints_dir.joinpath(checkpoint_name))
