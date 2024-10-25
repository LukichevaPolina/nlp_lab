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
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report, accuracy_score
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

import torch

from src.models.cnn.train import cnn_train
from src.models.cnn.infer import cnn_infer

from src.models.linear.train import linear_train
from src.models.linear.infer import linear_infer

import pandas as pd

from torcheval.metrics.classification.accuracy import MulticlassAccuracy
from torcheval.metrics.classification.f1_score import MulticlassF1Score

from src.utils.rendering import plot_accuracy_curve, plot_f1_curve, plot_learning_curve
from src.models.cnn.infer import cnn_infer


SEED = 4200
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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
    LINEAR=3
    CNN=4

class Mode(Enum):
    TRAIN=1
    EVAL=2
    INFER=3

TARGET2ENUM = {
    "cnn": Algorithm.CNN,
    "linear": Algorithm.LINEAR,
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

        self._algorithm = self.str2enum(algorithm)
        self._preprocessor = self.str2enum(preprocessor)
        self._embedder = self.str2enum(embeddings)
        self._mode = self.str2enum(mode)
        self._checkpoint_path = self.get_checkpoint(algorithm, embeddings, class_balancer, preprocessor)
        self._metrics_path = self.get_metrics_name(algorithm, embeddings, class_balancer, preprocessor)
        self._preprocessed_data = None
        self._vectorized_data = None
    
    def run(self) -> None:
        self.preprocess()

        X_train, X_test, y_train, y_test = self.vectorize()
        if self._mode == Mode.TRAIN:
            self.train(X_train, y_train, X_test, y_test)
            #self.eval(X_test, y_test)
        elif self._mode == Mode.EVAL:
            self.eval(X_test, y_test)
        else:
            ValueError("Wrong mode")
    
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
            X_train, X_test, y_train, y_test = train_test_split(
                self._preprocessed_data["statement"],
                self._preprocessed_data["status"],
                test_size=0.3, 
                random_state=42, 
                stratify=self._preprocessed_data["status"]
            )

            vectorizer.fit_X(X_train)

            X_train = vectorizer.transform_X(X_train)
            X_test = vectorizer.transform_X(X_test)

            y_train = vectorizer.fit_transform_y(y_train)
            y_test = vectorizer.fit_transform_y(y_test)

            return X_train, X_test, y_train, y_test
        else:
            raise NotImplementedError

    def train(self, X_train, y_train, X_test, y_test) -> None:
        match self._algorithm:
            case Algorithm.SVM:
                svc = SVC(self._checkpoint_path)
                svc.train(X_train, y_train)
            case Algorithm.DECISION_TREE:
                dt = DecisionTree(self._checkpoint_path)
                dt.train(X_train, y_train)
            case Algorithm.CNN:
                train_metrics, val_metrics, train_losses, val_losses = cnn_train(self._checkpoint_path, X_train, y_train, X_test, y_test, num_epochs=100, batch_size=64)
                pd.DataFrame(train_metrics).to_csv(f"metrics/train_metrics_{self._metrics_path}")
                pd.DataFrame(val_metrics).to_csv(f"metrics/val_metrics_{self._metrics_path}")
                pd.DataFrame(train_losses).to_csv(f"metrics/train_losses_{self._metrics_path}")
                pd.DataFrame(val_losses).to_csv(f"metrics/val_losses_{self._metrics_path}")
                plot_learning_curve(train_losses["ce"], val_losses["ce"], name="cnn_learning_curve")
                plot_accuracy_curve(train_metrics["accuracy"], val_metrics["accuracy"], name="cnn_accuracy_curve")
                plot_f1_curve(train_metrics["f1score"], val_metrics["f1score"], name="cnn_f1_curve")
            case Algorithm.LINEAR:
                train_metrics, val_metrics, train_losses, val_losses = linear_train(self._checkpoint_path, X_train, y_train, X_test, y_test, num_epochs=100, batch_size=64)
                pd.DataFrame(train_metrics).to_csv(f"metrics/train_metrics_{self._metrics_path}")
                pd.DataFrame(val_metrics).to_csv(f"metrics/val_metrics_{self._metrics_path}")
                pd.DataFrame(train_losses).to_csv(f"metrics/train_losses_{self._metrics_path}")
                pd.DataFrame(val_losses).to_csv(f"metrics/val_losses_{self._metrics_path}")
                plot_learning_curve(train_losses["ce"], val_losses["ce"], name="linear_learning_curve")
                plot_accuracy_curve(train_metrics["accuracy"], val_metrics["accuracy"], name="linear_accuracy_curve")
                plot_f1_curve(train_metrics["f1score"], val_metrics["f1score"], name="linear_f1_curve")
            case _:
                raise ValueError(f"Given algorithm: {self._algorithm} does not exist")

    def eval(self, X, y) -> None:
        y_pred = self.infer(X)
        if self._algorithm == Algorithm.SVM or self._algorithm == Algorithm.DECISION_TREE:
            f1_per_class = f1_score(y, y_pred, average=None) # TODO add computation
            accuracy = balanced_accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average="weighted")
            eval_metrics = {"f1_per_class": f1_per_class, "accuracy": accuracy, "f1_weighted": f1}
            print("[INFO] Metrics report:")
            print("===================")
            print("f1 per classes")
            print(f1_per_class)
            print("===================")
            print(f"accuracy = {accuracy}, f1_weighted = {f1}")
            self.save_results("eval_result_metrics", eval_metrics)

        elif self._algorithm == Algorithm.CNN or self._algorithm == Algorithm.LINEAR:
            f1_per_class = f1_score(y, y_pred, average=None) # TODO add computation
            accuracy = balanced_accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average="weighted")
            eval_metrics = {"f1_per_class": f1_per_class, "accuracy": accuracy, "f1_weighted": f1}
            print("[INFO] Metrics report:")
            print("===================")
            print("f1 per classes")
            print(f1_per_class)
            print("===================")
            print(f"accuracy = {accuracy}, f1_weighted = {f1}")
            self.save_results("eval_result_metrics", eval_metrics)
        else:
            raise ValueError(f"Imposible eval, not algorithm = {self._algorithm}")

    def infer(self, X) -> None:
        if not self._checkpoint_path.exists():
            raise RuntimeError(f"The checkpoint: {str(self._checkpoint_path)} does not exist, train first")

        if self._algorithm == Algorithm.SVM or self._algorithm == Algorithm.DECISION_TREE:
            model = joblib.load(self._checkpoint_path)
            y = model.predict(X)
            return y
        elif self._algorithm == Algorithm.CNN:
            y = cnn_infer(self._checkpoint_path, X)
            return y
        elif self._algorithm == Algorithm.LINEAR:
            pass
        else:
            NotImplementedError

    def save_results(self, name, metrics) -> None:
        file = "metrics/" + name + f"_{self._metrics_path}"
        with open(file, 'w') as fp:
            json.dump(metrics, fp)

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
        elif algorithm in ["linear", "cnn"]:
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
    
    def get_metrics_name(
        self, 
        algorithm: str, 
        embeddigns: str,
        class_balancer: str,
        preprocessor: str,
    ) -> str:
        if algorithm in ["linear", "cnn"]:
            return f"{algorithm}_{embeddigns}_{class_balancer}_{preprocessor}.csv"
        elif algorithm in ["svm", "decision_tree"]:
            return f"{algorithm}_{embeddigns}_{class_balancer}_{preprocessor}.json"
        else:
            raise ValueError(f"Given algorithm: {algorithm} does not exist")