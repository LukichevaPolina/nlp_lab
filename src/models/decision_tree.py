import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .base_model import BaseModel


class DecisionTree(BaseModel):
    def __init__(self, checkpoint_path):
        super().__init__(checkpoint_path)
        self.estimator = DecisionTreeClassifier(class_weight="balanced")
        self.param_grid = {'criterion': [
            'gini', 'entropy'], 'max_depth': np.arange(3, 15)}
