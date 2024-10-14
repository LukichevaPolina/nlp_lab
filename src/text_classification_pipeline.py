import sys
import argparse
import random
import numpy as np
from enum import Enum
from typing import Dict
from pathlib import Path
import pandas as pd
from string import punctuation

from src.utils.get_data import get_data
from src.eda.preprocessing.preprocessing import (drop_nan, remove_punctuation, remove_digits,
                                   remove_stop_words, tokenize, stemming, lemmatization)
from src.eda.preprocessing.embeddings import tfidf_embeddings
from src.eda.rendering.statistics import class_features_distribution, plot_class_distribution

SEED = 4200
random.seed(SEED)
np.random.seed(SEED)

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
}

# TODO change file chekpoint name
class TextClassificationPipeline:
    def __init__(
            self, 
            dataset_path: str,
            algorithm: str,
            embeddings: str,
            class_balancer: str,
            preprocessor: str,
            do_train: bool
        ) -> None:
        self._dataset = get_data(path=dataset_path)

        # TODO add dict for embeddings types
        # TODO add dict for classic algorithm results
        # TODO add dict for dl algorithm results
        # TODO add metrics
        self._algorithm = self.str2enum(algorithm)
        self._preprocessor = self.str2enum(preprocessor)
        self._checkpoint_path = self.get_checkpoint(algorithm, embeddings, class_balancer, preprocessor)
        self._do_train = do_train
    
    def run(self) -> None:
        preprocess_data = self._preprocessor()
        if self._do_train:
            self.train()
            self.infer()
        else:
            self.infer()

    def train(self) -> None:
        if self._algorithm == Algorithm.SVM:
            pass
        else:
            raise NotImplementedError

    def infer(self) -> None:
        if not self._checkpoint_path.exists():
            raise RuntimeError(f"The checkpoint: {str(self._checkpoint_path)} does not exist, train first")
        
        if self._algorithm == Algorithm.SVM:
            pass
        else:
            raise NotImplementedError
    
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
            # add plot of data
            return preprocess_data
        else:
            raise NotImplementedError

    def str2enum(self, target: str) -> Algorithm:
        try:
            return TARGET2ENUM[target]
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
        checkpoints_dir = Path("../checkpoints/")
        checkpoint_name = Path(self.get_checkpoint_name(algorithm, embeddings, class_balancer, preprocessor))
        return (checkpoints_dir.joinpath(checkpoint_name))

def parse_args() -> Dict[str, str]:
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset_path",
        required=True,
        type=str,
        help="Path to the dataset"
    )
    args.add_argument(
        "--algorithm",
        required=True,
        type=str,
        choices=["svm", "decision_tree", "lstm", "cnn"], #TODO add option for all
        help="algorithm to process"
    )
    args.add_argument(
        "--embeddigns",
        required=True,
        type=str,
        choices=["tfidf", "word2vec"], #TODO add option for all
        help="embedding which is used"
    )
    args.add_argument(
        "--class_balancer",
        required=True,
        type=str,
        choices=["class_weight", "union"], #TODO add option for all
        help="the strategy to deal with class disbalance"
    )
    args.add_argument(
        "--preprocessor",
        required=True,
        type=str,
        choices=["remove_all"], #TODO add option for all
        help="the strategy to do preprocess"
    )
    args.add_argument(
        "--do_train",
        required=False,
        type=bool,
        default=True,
        help="If true first train, then infer. If false only infer"
    )

    return vars(args.parse_args)

def main() -> None:
    args = parse_args
    pipeline = TextClassificationPipeline(
        dataset_path=args["dataset_path"], 
        algorithm=args["algorithm"], 
        embeddings=args["embeddigns"],
        class_balancer=args["class_balancer"],
        preprocessor=args["preprocessor"],
        train=args["do_train"]
    )

    pipeline.run()

if __name__ == "__main__":
    sys.exit(main() or 0)