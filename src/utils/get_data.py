import logging as log
import pandas as pd

def get_data(path: str="/content/drive/MyDrive/nlp_lab/Combined Data.csv") -> pd.DataFrame:
    log.info(f"Get dataset from {path}")
    dataset = pd.read_csv(path).drop("Unnamed: 0", axis=1)
    log.info(dataset.head())
    log.info(dataset.info())
    return dataset
