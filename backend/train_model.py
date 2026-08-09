
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ensembles import TemporalStackingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from data_loader import (
    CATEGORICAL_COLUMNS,
    DERIVED_FEATURES,
    NUMERIC_COLUMNS,
    SHARED_VOCABULARIES,
    TARGET,
    TIME_COL,
    load_all,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42

# Fraction of the timeline held out. Train is the earliest 70%, then validation,
# then test as the most recent 15%.
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# Tree count is env-tunable so a memory-constrained machine can dial it down.
N_ESTIMATORS = int(os.environ.get("RF_TREES", "100"))
# Bounded leaf size keeps forest memory in check on large datasets and
# regularises against the noise SMOTE introduces.
MIN_SAMPLES_LEAF = 20
N_JOBS = int(os.environ.get("N_JOBS", "3"))

UNSEEN_CODE = -1


def banner(text: str) -> None:
    print(f"\n{'=' * 62}\n{text}\n{'=' * 62}")


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model features. Runs BEFORE encoding so comparisons use raw values."""
    df = df.copy()
    df["amount_deviation"] = df["amount"] - df["avg_amount_user"]
    # Compared on raw values here; at serving time both columns share one
    # vocabulary, so the encoded comparison is equivalent.
    df["country_mismatch"] = (df["country"] != df["bin_country"]).astype(int)
    return df


# ----------------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------------
def fit_encoders(train_df: pd.DataFrame) -> dict[str, list[str]]:
    
    encoders: dict[str, list[str]] = {}

    shared_members = {c for cols in SHARED_VOCABULARIES.values() for c in cols}
    for cols in SHARED_VOCABULARIES.values():
        present = [c for c in cols if c in train_df.columns]
        vocab = sorted(set().union(*(train_df[c].astype(str).unique() for c in present)))
        for col in present:
            encoders[col] = vocab

    for col in CATEGORICAL_COLUMNS:
        if col in shared_members or col not in train_df.columns:
            continue
        encoders[col] = sorted(train_df[col].astype(str).unique().tolist())

    return encoders


def apply_encoders(df: pd.DataFrame, encoders: dict[str, list[str]]) -> pd.DataFrame:
    """Map categories to codes. Categories unseen in training become -1."""
    df = df.copy()
    for col, classes in encoders.items():
        mapping = {c: i for i, c in enumerate(classes)}
        df[col] = df[col].astype(str).map(mapping).fillna(UNSEEN_CODE).astype(int)
    return df


# ----------------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------------
def temporal_split(df: pd.DataFrame):
    """Split chronologically into train / validation / test."""
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    n = len(df)
    test_start = int(n * (1 - TEST_FRACTION))
    val_start = int(n * (1 - TEST_FRACTION - VAL_FRACTION))
    return df.iloc[:val_start], df.iloc[val_start:test_start], df.iloc[test_start:]


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(y_true, probs, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
    }


def best_f1_threshold(y_true, probs) -> float:
    """Pick the probability cutoff maximising F1 on the given fold."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    # precision/recall have one more element than thresholds.
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(thresholds, dtype=float),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    return float(thresholds[int(np.argmax(f1))])


# Boosting families, tracked so the ensemble can pick the best one rather than
# stacking four near-identical gradient boosters together.
GBM_NAMES = {"Hist Gradient Boosting", "LightGBM", "XGBoost", "CatBoost"}


def build_candidates(balanced: bool, pos_weight: float) -> dict:
    """Fresh model instances.

    *balanced* is False when SMOTE has already levelled the classes, so the
    model must not reweight them a second time. *pos_weight* is the
    negative:positive ratio XGBoost uses in place of class_weight.
    """
    cw = "balanced" if balanced else None
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight=cw,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=N_ESTIMATORS, min_samples_leaf=MIN_SAMPLES_LEAF,
            n_jobs=N_JOBS, random_state=RANDOM_STATE, class_weight=cw,
        ),
        # Undersamples the majority class inside every tree; balances itself,
        # so it ignores the strategy arm it is run in.
        "Balanced Random Forest": BalancedRandomForestClassifier(
            n_estimators=N_ESTIMATORS, min_samples_leaf=MIN_SAMPLES_LEAF,
            n_jobs=N_JOBS, random_state=RANDOM_STATE,
            sampling_strategy="all", replacement=True, bootstrap=False,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1,
            random_state=RANDOM_STATE, class_weight=cw,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=MIN_SAMPLES_LEAF, n_jobs=N_JOBS,
            random_state=RANDOM_STATE, class_weight=cw, verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=5,
            tree_method="hist", n_jobs=N_JOBS, random_state=RANDOM_STATE,
            scale_pos_weight=pos_weight if balanced else 1.0,
            eval_metric="aucpr", verbosity=0,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=6,
            random_seed=RANDOM_STATE, thread_count=N_JOBS,
            auto_class_weights="Balanced" if balanced else None,
            verbose=0, allow_writing_files=False,
        ),
    }


def build_ensembles(ranked_gbms: list[str], pos_weight: float) -> dict:
    """Combine deliberately DIFFERENT model families.

    Averaging four gradient boosters would just average four copies of the same
    inductive bias, so each ensemble pairs the two strongest boosters with a
    bagged forest -- and, for stacking only, a linear model.
    """
    base = build_candidates(balanced=True, pos_weight=pos_weight)
    top, second = ranked_gbms[0], ranked_gbms[1]

    # Soft voting weights every member equally, so it is dragged down by its
    # weakest one. Logistic Regression (validation PR-AUC ~0.60) is excluded
    # here for that reason. Stacking keeps it: the meta-learner can assign it a
    # small coefficient, so a weak-but-different model can still contribute.
    voting_members = [(top, base[top]), (second, base[second]),
                      ("Random Forest", base["Random Forest"])]
    stacking_members = voting_members + [("Logistic Regression",
                                          base["Logistic Regression"])]

    return {
        f"Voting ({top} + {second} + RF)": VotingClassifier(
            estimators=[(n, e) for n, e in voting_members], voting="soft",
        ),
        f"Stacking ({top} + {second} + RF + LR)": TemporalStackingClassifier(
            estimators=stacking_members,
            final_estimator=LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE,
            ),
            holdout=0.25,
        ),
    }


def main() -> None:
    banner("1. LOADING DATASETS")
    df = load_all()

    banner("2. FEATURE ENGINEERING")
    df = engineer_features(df)
    print(f"Derived: {', '.join(DERIVED_FEATURES)}")

    banner("3. TEMPORAL SPLIT")
    train_df, val_df, test_df = temporal_split(df)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name:<6} {len(part):>7,} rows  "
              f"{part[TIME_COL].min():%Y-%m-%d} -> {part[TIME_COL].max():%Y-%m-%d}  "
              f"fraud {part[TARGET].mean() * 100:.2f}%")

    banner("4. ENCODING (fit on train only)")
    encoders = fit_encoders(train_df)
    for col, classes in encoders.items():
        preview = classes if len(classes) <= 6 else classes[:6] + ["..."]
        print(f"{col:<20} {len(classes):>2} categories: {preview}")
    train_df, val_df, test_df = (apply_encoders(d, encoders) for d in (train_df, val_df, test_df))

    # Listed explicitly rather than "every column except the drops" so an extra
    # column in a new CSV can never silently become a model feature.
    feature_columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + DERIVED_FEATURES

    def xy(part):
        return part[feature_columns], part[TARGET].astype(int)

    X_train, y_train = xy(train_df)
    X_val, y_val = xy(val_df)
    X_test, y_test = xy(test_df)
    print(f"\n{len(feature_columns)} features: {feature_columns}")

    banner("5. SCALING (fit on train only)")
    scaler = StandardScaler().fit(X_train)

    def scale(X):
        # Keep it a DataFrame so the scaler sees feature names at fit and at
        # serve time -- otherwise sklearn warns on every prediction.
        return pd.DataFrame(scaler.transform(X), columns=feature_columns,
                            index=X.index).astype(np.float32)

    X_train_s, X_val_s, X_test_s = scale(X_train), scale(X_val), scale(X_test)
    print("StandardScaler fitted; matrices cast to float32")

    banner("6. RESAMPLING (train fold only)")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_s, y_train)
    print(f"class_weight strategy : {len(X_train_s):,} rows "
          f"(fraud {y_train.mean() * 100:.2f}%)")
    print(f"SMOTE strategy        : {len(X_train_smote):,} rows "
          f"(fraud {y_train_smote.mean() * 100:.2f}%)")

    banner("7. BASE MODEL SELECTION (ranked by validation PR-AUC)")
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"XGBoost scale_pos_weight = {pos_weight:.2f}\n")

    strategies = {
        "class_weight": (X_train_s, y_train, True),
        "SMOTE": (X_train_smote, y_train_smote, False),
    }

    results, best = [], None

    def record(name, strategy, model, val_probs):
        nonlocal best
        pr_auc = average_precision_score(y_val, val_probs)
        roc = roc_auc_score(y_val, val_probs)
        results.append({"Model": name, "Strategy": strategy,
                        "Val PR-AUC": pr_auc, "Val ROC-AUC": roc})
        print(f"  {name:<34} {strategy:<13} PR-AUC {pr_auc:.4f}   ROC-AUC {roc:.4f}")
        if best is None or pr_auc > best["pr_auc"]:
            best = {"model": model, "name": name, "strategy": strategy,
                    "pr_auc": pr_auc, "val_probs": val_probs}

    for strategy, (Xt, yt, balanced) in strategies.items():
        for name, model in build_candidates(balanced, pos_weight).items():
            model.fit(Xt, yt)
            record(name, strategy, model, model.predict_proba(X_val_s)[:, 1])

    ranked_gbms = [
        r["Model"] for r in sorted(
            (r for r in results
             if r["Model"] in GBM_NAMES and r["Strategy"] == "class_weight"),
            key=lambda r: r["Val PR-AUC"], reverse=True,
        )
    ]

    banner("8. ENSEMBLES")
    print(f"Boosters ranked: {' > '.join(ranked_gbms)}")
    print(f"Combining {ranked_gbms[0]} + {ranked_gbms[1]} + Random Forest "
          f"for family diversity (class_weight arm).\n")
    ensembles = build_ensembles(ranked_gbms, pos_weight)
    for name, ensemble in ensembles.items():
        ensemble.fit(X_train_s, y_train)
        record(name, "class_weight", ensemble, ensemble.predict_proba(X_val_s)[:, 1])
        if isinstance(ensemble, TemporalStackingClassifier):
            weights = ", ".join(f"{n}={w:+.3f}" for n, w in ensemble.meta_weights_.items())
            print(f"      meta-learner coefficients: {weights}")

    banner("9. LEADERBOARD")
    print(pd.DataFrame(results)
          .sort_values("Val PR-AUC", ascending=False)
          .to_string(index=False))

    banner("10. THRESHOLD TUNING (on validation)")
    threshold = best_f1_threshold(y_val, best["val_probs"])
    print(f"Winner: {best['name']} + {best['strategy']}")
    print(f"Tuned threshold: {threshold:.4f} (default would be 0.5000)")
    print("\nValidation @ tuned threshold:")
    for k, v in evaluate(y_val, best["val_probs"], threshold).items():
        print(f"  {k:<10} {v:.4f}")

    banner("11. FINAL TEST EVALUATION (held out, scored once)")
    test_probs = best["model"].predict_proba(X_test_s)[:, 1]
    at_default = evaluate(y_test, test_probs, 0.5)
    at_tuned = evaluate(y_test, test_probs, threshold)

    print(f"{'metric':<12}{'@0.50':>12}{'@tuned':>12}")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        print(f"{k:<12}{at_default[k]:>12.4f}{at_tuned[k]:>12.4f}")

    tn, fp, fn, tp = confusion_matrix(y_test, (test_probs >= threshold).astype(int)).ravel()
    print(f"\nConfusion matrix @ tuned threshold:")
    print(f"  true negative  {tn:>7,}    false positive {fp:>7,}")
    print(f"  false negative {fn:>7,}    true positive  {tp:>7,}")
    print(f"\n  caught {tp:,} of {tp + fn:,} frauds; "
          f"{fp:,} false alarms out of {tn + fp:,} legitimate transactions")

    banner("12. SAVING ARTIFACTS")
    artifacts = {
        "model.pkl": best["model"],
        "scaler.pkl": scaler,
        "features.pkl": feature_columns,
        "encoders.pkl": encoders,
    }
    for filename, obj in artifacts.items():
        with open(os.path.join(BASE_DIR, filename), "wb") as fh:
            pickle.dump(obj, fh)
        print(f"  saved {filename}")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "model": best["name"],
        "resampling": best["strategy"],
        "threshold": threshold,
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(X_train)),
        "features": feature_columns,
        "validation_pr_auc": float(best["pr_auc"]),
        "test_metrics_default_threshold": at_default,
        "test_metrics_tuned_threshold": at_tuned,
        "leaderboard": results,
    }
    with open(os.path.join(BASE_DIR, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print("  saved metadata.json")

    print(f"\nBest model: {best['name']} + {best['strategy']}")
    print(f"Test PR-AUC: {at_tuned['pr_auc']:.4f}  |  "
          f"precision {at_tuned['precision']:.4f}  recall {at_tuned['recall']:.4f}")


if __name__ == "__main__":
    main()
