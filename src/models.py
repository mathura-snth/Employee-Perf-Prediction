import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import SEED


class WeightedXGB(ClassifierMixin, BaseEstimator):
    """XGBoost with balanced sample weights computed inside fit()."""

    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 subsample=1.0, colsample_bytree=1.0, random_state=SEED, n_jobs=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state, n_jobs=self.n_jobs,
        )
        self.model_.fit(X, y_idx, sample_weight=compute_sample_weight("balanced", y_idx))
        return self

    def predict(self, X):
        return self.classes_[self.model_.predict(X)]
