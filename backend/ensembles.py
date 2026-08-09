"""Ensemble estimators that respect the chronological ordering of the data.

Lives in its own module because the fitted object gets pickled into model.pkl,
and unpickling has to be able to import the class by name at serve time.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class TemporalStackingClassifier(ClassifierMixin, BaseEstimator):
    """Stacking whose meta-learner is trained on a chronologically later block.

    scikit-learn's StackingClassifier builds its meta-features with
    ``cross_val_predict``, which requires the CV splitter to *partition* the
    data. ``TimeSeriesSplit`` does not partition -- the earliest block is never
    part of any test fold -- so it raises "cross_val_predict only works for
    partitions".

    Falling back to the default StratifiedKFold would fix the error by letting
    the meta-learner train on out-of-fold predictions drawn from the future,
    which is the same leakage this pipeline removes elsewhere. Instead:

    1. Fit the base models on the earliest ``1 - holdout`` of the training fold.
    2. Predict the final ``holdout`` block and fit the meta-learner on those
       predictions, so it only ever learns from forward-in-time predictions.
    3. Refit the base models on the whole training fold, so the deployed
       ensemble uses all available data.

    Step 3 makes the base models slightly stronger than the ones the
    meta-learner was calibrated against -- the same tradeoff sklearn's own
    ``StackingClassifier`` makes when ``passthrough=False``.
    """

    def __init__(self, estimators=None, final_estimator=None, holdout=0.25):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.holdout = holdout

    def _stack(self, fitted, X):
        return np.column_stack([est.predict_proba(X)[:, 1] for est in fitted])

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        self.n_features_in_ = X_arr.shape[1]

        cut = int(len(X_arr) * (1 - self.holdout))
        if cut <= 0 or cut >= len(X_arr):
            raise ValueError(f"holdout={self.holdout} leaves no data to split on")

        # 1 + 2: bases learn from the past, meta-learner from their future.
        early = [clone(est).fit(X_arr[:cut], y_arr[:cut]) for _, est in self.estimators]
        self.final_estimator_ = clone(self.final_estimator).fit(
            self._stack(early, X_arr[cut:]), y_arr[cut:]
        )

        # 3: refit on everything for deployment.
        self.estimators_ = [clone(est).fit(X_arr, y_arr) for _, est in self.estimators]
        self.named_estimators_ = dict(zip([n for n, _ in self.estimators], self.estimators_))
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        return self.final_estimator_.predict_proba(self._stack(self.estimators_, X_arr))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def meta_weights_(self) -> dict[str, float]:
        """How much the meta-learner leans on each base model."""
        coefs = np.ravel(self.final_estimator_.coef_)
        return {name: float(c) for (name, _), c in zip(self.estimators, coefs)}
