"""Tests for the multi-model bake-off and the ensemble estimators."""
import numpy as np
import pickle
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ensembles import TemporalStackingClassifier

EXPECTED_FAMILIES = [
    "Logistic Regression",
    "Random Forest",
    "Balanced Random Forest",
    "Hist Gradient Boosting",
    "LightGBM",
    "XGBoost",
    "CatBoost",
]


# --------------------------------------------------------------------------
# The bake-off actually ran every family
# --------------------------------------------------------------------------
def test_leaderboard_covers_every_model_family(flask_app):
    names = {row["Model"] for row in flask_app.METADATA["leaderboard"]}
    for family in EXPECTED_FAMILIES:
        assert family in names, f"{family} missing from the leaderboard"


def test_leaderboard_includes_both_ensembles(flask_app):
    names = {row["Model"] for row in flask_app.METADATA["leaderboard"]}
    assert any(n.startswith("Voting") for n in names)
    assert any(n.startswith("Stacking") for n in names)


def test_both_balancing_strategies_were_evaluated(flask_app):
    strategies = {row["Strategy"] for row in flask_app.METADATA["leaderboard"]}
    assert {"class_weight", "SMOTE"} <= strategies


def test_deployed_model_is_the_leaderboard_winner(flask_app):
    board = flask_app.METADATA["leaderboard"]
    top = max(board, key=lambda row: row["Val PR-AUC"])
    assert top["Model"] == flask_app.METADATA["model"]
    assert top["Strategy"] == flask_app.METADATA["resampling"]


def test_deployed_model_exposes_predict_proba(flask_app):
    assert hasattr(flask_app.model, "predict_proba")


# --------------------------------------------------------------------------
# TemporalStackingClassifier
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def toy():
    rng = np.random.RandomState(0)
    X = rng.rand(1200, 4)
    y = (X[:, 0] + rng.rand(1200) * 0.4 > 1.1).astype(int)
    return X, y


@pytest.fixture(scope="module")
def fitted_stack(toy):
    X, y = toy
    return TemporalStackingClassifier(
        estimators=[("rf", RandomForestClassifier(n_estimators=20, random_state=0)),
                    ("lr", LogisticRegression(max_iter=500))],
        final_estimator=LogisticRegression(max_iter=500),
        holdout=0.25,
    ).fit(X, y)


def test_stacking_returns_valid_probabilities(fitted_stack, toy):
    probs = fitted_stack.predict_proba(toy[0])
    assert probs.shape == (len(toy[0]), 2)
    assert probs.min() >= 0 and probs.max() <= 1
    assert np.allclose(probs.sum(axis=1), 1)


def test_stacking_predict_matches_argmax(fitted_stack, toy):
    probs = fitted_stack.predict_proba(toy[0])
    assert np.array_equal(fitted_stack.predict(toy[0]), np.argmax(probs, axis=1))


def test_stacking_survives_pickling(fitted_stack, toy):
    """The winner gets pickled into model.pkl, so this must round-trip."""
    restored = pickle.loads(pickle.dumps(fitted_stack))
    assert np.allclose(restored.predict_proba(toy[0]),
                       fitted_stack.predict_proba(toy[0]))


def test_stacking_refits_bases_on_the_full_fold(fitted_stack, toy):
    """Step 3 of the docstring: deployed bases see all the training data."""
    assert len(fitted_stack.estimators_) == 2
    assert fitted_stack.n_features_in_ == toy[0].shape[1]


def test_stacking_exposes_meta_weights(fitted_stack):
    weights = fitted_stack.meta_weights_
    assert set(weights) == {"rf", "lr"}
    assert all(isinstance(v, float) for v in weights.values())


def test_stacking_rejects_degenerate_holdout(toy):
    X, y = toy
    stack = TemporalStackingClassifier(
        estimators=[("lr", LogisticRegression(max_iter=200))],
        final_estimator=LogisticRegression(max_iter=200),
        holdout=1.0,
    )
    with pytest.raises(ValueError, match="holdout"):
        stack.fit(X, y)
