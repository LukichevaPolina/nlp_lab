import os
import joblib
from sklearn.model_selection import GridSearchCV

class BaseModel:
    def __init__(self, checkpoint_path):
        self._checkpoint_path = checkpoint_path
        self.estimator = None
        self.param_grid = None

    def train(self, X, y):
        if not os.path.isfile(self._checkpoint_path):
            os.mknod(self._checkpoint_path)

        gridsearch = GridSearchCV(estimator=self.estimator,
                              param_grid=self.param_grid,
                              cv=2,
                              scoring=["f1_weighted", "accuracy"],
                              refit="f1_weighted",
                              verbose=2)
        gridsearch.fit(X, y)

        joblib.dump(gridsearch.best_estimator_, self._checkpoint_path)

        print("Best parameters from gridsearch: {}".format(gridsearch.best_params_))
        print("CV score=%0.3f" % gridsearch.best_score_)