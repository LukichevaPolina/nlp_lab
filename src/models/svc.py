from .base_model import BaseModel
from sklearn.svm import LinearSVC

class SVC(BaseModel):
    def __init__(self, checkpoint_path):
        super().__init__(checkpoint_path)
        self.estimator = LinearSVC(class_weight="balanced")
        self.param_grid = {"C": [1, 10, 100, 1000],
                           "multi_class": ["ovr", "crammer_singer"]}
