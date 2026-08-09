"""Tests for multi-dataset discovery and schema validation."""
import pandas as pd
import pytest

import data_loader


def _valid_frame(n=40, start="2024-01-01"):
    return pd.DataFrame({
        "transaction_id": range(n),
        "user_id": range(n),
        "account_age_days": [100] * n,
        "total_transactions_user": [10] * n,
        "avg_amount_user": [50.0] * n,
        "amount": [60.0] * n,
        "country": ["DE"] * n,
        "bin_country": ["DE"] * n,
        "channel": ["web"] * n,
        "merchant_category": ["grocery"] * n,
        "promo_used": [0] * n,
        "avs_match": [1] * n,
        "cvv_result": [1] * n,
        "three_ds_flag": [1] * n,
        "transaction_time": pd.date_range(start, periods=n, freq="h").astype(str),
        "shipping_distance_km": [10.0] * n,
        "is_fraud": [0, 1] * (n // 2),
    })


def test_discovers_and_unions_multiple_csvs(tmp_path):
    _valid_frame(40).to_csv(tmp_path / "a.csv", index=False)
    second = _valid_frame(20, start="2024-06-01")
    second["transaction_id"] = range(1000, 1020)
    second["user_id"] = range(1000, 1020)
    second.to_csv(tmp_path / "b.csv", index=False)

    combined = data_loader.load_all(str(tmp_path), verbose=False)
    assert len(combined) == 60
    assert set(combined["source_file"]) == {"a.csv", "b.csv"}


def test_incompatible_csv_is_skipped_not_merged(tmp_path, capsys):
    _valid_frame(40).to_csv(tmp_path / "good.csv", index=False)
    # Mimics the ULB creditcard schema: shares only 'Amount', nothing else.
    pd.DataFrame({"Time": [0, 1], "V1": [0.1, 0.2], "Amount": [5.0, 6.0],
                  "Class": [0, 1]}).to_csv(tmp_path / "creditcard.csv", index=False)

    combined = data_loader.load_all(str(tmp_path), verbose=False)
    assert set(combined["source_file"]) == {"good.csv"}
    assert "SKIPPED" in capsys.readouterr().out


def test_paysim_schema_is_rejected_from_the_header_alone(tmp_path, capsys):
    """PaySim shares only 'amount' and spells the label 'isFraud'.

    The rejection must come from the header, without reading the body -- these
    files run to hundreds of megabytes.
    """
    _valid_frame(40).to_csv(tmp_path / "good.csv", index=False)
    pd.DataFrame({
        "step": [1, 1], "type": ["PAYMENT", "TRANSFER"], "amount": [9839.64, 1864.28],
        "nameOrig": ["C123", "C166"], "oldbalanceOrg": [170136.0, 21249.0],
        "newbalanceOrig": [160296.36, 19384.72], "nameDest": ["M197", "M204"],
        "oldbalanceDest": [0.0, 0.0], "newbalanceDest": [0.0, 0.0],
        "isFraud": [0, 1], "isFlaggedFraud": [0, 0],
    }).to_csv(tmp_path / "Dataset.csv", index=False)

    combined = data_loader.load_all(str(tmp_path), verbose=False)
    out = capsys.readouterr().out
    assert set(combined["source_file"]) == {"good.csv"}
    assert "SKIPPED without loading" in out
    assert "is_fraud" in out, "the reason should name the missing target column"


def test_validate_columns_needs_no_data(tmp_path):
    """Header-only validation must work on a body-less read."""
    _valid_frame(10).to_csv(tmp_path / "a.csv", index=False)
    header = pd.read_csv(tmp_path / "a.csv", nrows=0)
    assert len(header) == 0
    assert data_loader.validate_columns(header.columns)[0] is True
    assert data_loader.validate_columns(["step", "isFraud"])[0] is False


def test_duplicate_transactions_across_files_are_dropped(tmp_path):
    frame = _valid_frame(40)
    frame.to_csv(tmp_path / "a.csv", index=False)
    frame.to_csv(tmp_path / "b_duplicate.csv", index=False)

    combined = data_loader.load_all(str(tmp_path), verbose=False)
    assert len(combined) == 40


def test_nulls_are_imputed(tmp_path):
    frame = _valid_frame(40)
    frame.loc[0:4, "amount"] = None
    frame.loc[5:9, "country"] = None
    frame.to_csv(tmp_path / "holes.csv", index=False)

    combined = data_loader.load_all(str(tmp_path), verbose=False)
    assert combined["amount"].isnull().sum() == 0
    assert combined["country"].isnull().sum() == 0


def test_empty_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data_loader.load_all(str(tmp_path), verbose=False)


def test_all_files_invalid_raises(tmp_path):
    pd.DataFrame({"foo": [1, 2]}).to_csv(tmp_path / "junk.csv", index=False)
    with pytest.raises(ValueError):
        data_loader.load_all(str(tmp_path), verbose=False)
