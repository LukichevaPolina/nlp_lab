import sys
import argparse
import random
import numpy as np
from enum import Enum
from typing import Dict
from pathlib import Path

from src.utils.get_data import get_data

SEED = 4200
random.seed(SEED)
np.random.seed(SEED)

class Algorithm(Enum):
    SVM=1
    DECISION_TREE=2
    LSTM=3
    CNN=4

class TextClassificationPipeline:
    def __init__(
            self, 
            dataset_path: str,
            algorithm: str,
            do_train: bool
        ) -> None:
        self._dataset = get_data(path=dataset_path)
        # TODO add dict for embeddings types
        # TODO add preprocessing
        # TODO add dict for classic algorithm results
        # TODO add dict for dl algorithm results
        # TODO add metrics
        self._algorithm = self.str2enum(algorithm)
        self._checkpoint = self.get_checkpoint(algorithm)
        self._do_train = do_train
    
    def run(self) -> None:
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
        if not self._checkpoint.exists():
            raise RuntimeError(f"The checkpoint: {str(self._checkpoint)} does not exist, train first")
        
        if self._algorithm == Algorithm.SVM:
            pass
        else:
            raise NotImplementedError
    
    def str2enum(self, algorithm: str) -> Algorithm:
        if algorithm == "svm":
            return Algorithm.SVM
        elif algorithm == "decision_tree":
            return Algorithm.DECISION_TREE
        elif algorithm == "lstm":
            return Algorithm.LSTM
        elif algorithm == "cnn":
            return Algorithm.CNN
        else:
            raise ValueError(f"Given algorithm: {algorithm} does not exist")
    
    def get_checkpoint_name(self, algorithm: str) -> str:
        if algorithm == "svm":
            return "svm.pkl"
        elif algorithm == "decision_tree":
            return "decision_tree"
        elif algorithm == "lstm":
            return "lstm.pt"
        elif algorithm == "cnn":
            return "cnn.pt"
        else:
            raise ValueError(f"Given algorithm: {algorithm} does not exist")
    
    def get_checkpoint(self, algorithm: str) -> Path:
        checkpoints_dir = Path("../checkpoints/")
        checkpoint_name = Path(self.get_checkpoint_name(algorithm))
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
        "--do_train",
        required=False,
        type=bool,
        default=True,
        help="If true first train, then infer. If false only infer"
    )

    return vars(args.parse_args)

def main() -> None:
    args = parse_args
    pipeline = TextClassificationPipeline(dataset_path=args["dataset_path"], algorithm=args["algorithm"], train=args["do_train"])

    pipeline.run()

if __name__ == "__main__":
    sys.exit(main() or 0)