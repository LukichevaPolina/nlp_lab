import sys
import argparse
from src.text_classification_pipeline import TextClassificationPipeline
from typing import Dict


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
        choices=["svm", "decision-tree", "linear",
                 "cnn"],  # TODO add option for all
        help="algorithm to process"
    )
    args.add_argument(
        "--embeddigns",
        required=True,
        type=str,
        choices=["tfidf", "word2vec"],  # TODO add option for all
        help="embedding which is used"
    )
    args.add_argument(
        "--class_balancer",
        required=True,
        type=str,
        choices=["class-weight", "union"],  # TODO add option for all
        help="the strategy to deal with class disbalance"
    )
    args.add_argument(
        "--preprocessor",
        required=True,
        type=str,
        choices=["remove-all"],  # TODO add option for all
        help="the strategy to do preprocess"
    )
    args.add_argument(
        "--mode",
        required=True,
        type=str,
        choices=["train", "eval"],
        help="mode which used: \
              train - model will be trained and evaluated, \
              eval  - model will be evaluated"
    )

    return vars(args.parse_args())


def main() -> None:
    args = parse_args()
    pipeline = TextClassificationPipeline(
        dataset_path=args["dataset_path"],
        algorithm=args["algorithm"],
        embeddings=args["embeddigns"],
        class_balancer=args["class_balancer"],
        preprocessor=args["preprocessor"],
        mode=args["mode"]
    )

    pipeline.run()


if __name__ == "__main__":
    sys.exit(main() or 0)
