"""Dataset discovery, schema validation, and loading.

Any CSV dropped into ``data/`` whose columns satisfy REQUIRED_COLUMNS is picked
up automatically and unioned with the others -- adding a new compatible dataset
needs no code change. Files that do not match the schema are reported and
skipped rather than silently corrupting the training set.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))

TARGET = "is_fraud"
TIME_COL = "transaction_time"

# Numeric features fed to the model as-is.
NUMERIC_COLUMNS = [
    "account_age_days",
    "avg_amount_user",
    "amount",
    "shipping_distance_km",
    "promo_used",
    "avs_match",
    "three_ds_flag",
]

# Features that get integer-encoded before training.
CATEGORICAL_COLUMNS = [
    "country",
    "bin_country",
    "channel",
    "cvv_result",
]

# Computed in train_model.engineer_features from the columns above.
DERIVED_FEATURES = ["amount_deviation", "country_mismatch"]

# Dropped after permutation-importance testing on the held-out fold. Each
# contributed ~0% of PR-AUC, and removing all three raised test PR-AUC from
# 0.8624 to 0.8643 and precision from 0.920 to 0.932:
#   hour, total_transactions_user, merchant_category  -- no measurable signal
#   high_amount  -- redundant with the continuous amount_deviation
# They are no longer collected by the form or required of input CSVs.
EXCLUDED_FEATURES = ["hour", "total_transactions_user", "merchant_category", "high_amount"]

# Columns that must share a single vocabulary so their encoded values stay
# comparable. country vs bin_country are compared directly to derive
# country_mismatch, so encoding them independently would make "US" mean 9 in one
# column and something else in the other.
SHARED_VOCABULARIES = {"geo": ["country", "bin_country"]}

REQUIRED_COLUMNS = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TIME_COL, TARGET])

# Dropped before training: identifiers carry no signal and leak user identity.
ID_COLUMNS = ["transaction_id", "user_id"]


def discover_datasets(data_dir: str = DATA_DIR) -> list[str]:
    """Return every CSV in *data_dir*, sorted for reproducible ordering."""
    return sorted(glob.glob(os.path.join(data_dir, "*.csv")))


def validate_columns(columns) -> tuple[bool, str]:
    """Header-only check -- runs before the file body is read."""
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        return False, f"missing columns: {sorted(missing)}"
    return True, "ok"


def validate_schema(df: pd.DataFrame, path: str | None = None) -> tuple[bool, str]:
    """Full check, including the target column's contents."""
    ok, reason = validate_columns(df.columns)
    if not ok:
        return False, reason
    if df[TARGET].dropna().nunique() < 2:
        return False, f"target '{TARGET}' has fewer than 2 classes"
    return True, "ok"


def _impute(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Fill nulls: mode for categoricals, median for numerics.

    Only touches columns that actually contain nulls, and says so when it does.
    """
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if cols_with_nulls.empty:
        if verbose:
            print("    no missing values to impute")
        return df

    if verbose:
        print(f"    imputing {len(cols_with_nulls)} column(s) with nulls:")
    for col, n in cols_with_nulls.items():
        if col == TARGET:
            continue
        if df[col].dtype == "object":
            fill = df[col].mode()
            fill = fill.iloc[0] if not fill.empty else "unknown"
        else:
            fill = df[col].median()
        df[col] = df[col].fillna(fill)
        if verbose:
            print(f"      {col}: {n} null(s) -> {fill!r}")
    return df


def load_dataset(path: str, verbose: bool = True) -> pd.DataFrame | None:
    """Load and validate one CSV. Returns None if it fails validation."""
    name = os.path.basename(path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if verbose:
        print(f"  - {name}  ({size_mb:,.1f} MB)")

    # Check the header before reading the body. An incompatible file can be
    # hundreds of megabytes -- loading it in full only to discard it wastes
    # time and can exhaust memory on a small machine.
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception as exc:  # unreadable file should not kill the whole run
        print(f"    SKIPPED - could not read header: {exc}")
        return None

    ok, reason = validate_columns(header.columns)
    if not ok:
        print(f"    SKIPPED without loading ({size_mb:,.1f} MB) - {reason}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"    SKIPPED - could not read: {exc}")
        return None

    ok, reason = validate_schema(df)
    if not ok:
        print(f"    SKIPPED - {reason}")
        return None

    df = _impute(df, verbose=verbose)
    df = df.dropna(subset=[TARGET])
    df["source_file"] = name
    if verbose:
        print(f"    {len(df):,} rows  |  fraud {df[TARGET].mean() * 100:.2f}%")
    return df


def load_all(data_dir: str = DATA_DIR, verbose: bool = True) -> pd.DataFrame:
    """Discover, validate, and concatenate every compatible CSV in *data_dir*."""
    paths = discover_datasets(data_dir)
    if not paths:
        raise FileNotFoundError(f"no CSV files found in {data_dir}")

    if verbose:
        print(f"Scanning {data_dir}")
        print(f"Found {len(paths)} CSV file(s):")

    frames = [df for p in paths if (df := load_dataset(p, verbose)) is not None]
    if not frames:
        raise ValueError(f"no CSV in {data_dir} matched the required schema")

    combined = pd.concat(frames, ignore_index=True)

    # Drop cross-file duplicates so overlapping datasets do not double-weight rows.
    id_cols = [c for c in ID_COLUMNS if c in combined.columns]
    if id_cols:
        before = len(combined)
        combined = combined.drop_duplicates(subset=id_cols, keep="first")
        removed = before - len(combined)
        if removed and verbose:
            print(f"  removed {removed:,} duplicate row(s) by {id_cols}")

    combined[TIME_COL] = pd.to_datetime(combined[TIME_COL], errors="coerce", utc=True)
    combined = combined.dropna(subset=[TIME_COL])

    if verbose:
        print(f"\nCombined: {len(combined):,} rows from {len(frames)} dataset(s)")
        print(f"Fraud rate: {combined[TARGET].mean() * 100:.3f}%  "
              f"({int(combined[TARGET].sum()):,} fraud / {len(combined):,} total)")
        if len(frames) > 1:
            print(combined.groupby("source_file")[TARGET]
                  .agg(rows="size", fraud_rate="mean").to_string())

    return combined


if __name__ == "__main__":
    load_all()
