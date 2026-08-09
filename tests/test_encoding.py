"""Regression tests for the encoding bug.

The original app hardcoded dropdown values (India=0, USA=1, UK=2) that had no
relationship to what LabelEncoder actually produced (DE=0, ES=1, FR=2). These
tests fail if the form's options ever drift from the model's vocabulary again.
"""
import pytest


def test_encoders_cover_every_categorical_column(flask_app):
    from data_loader import CATEGORICAL_COLUMNS

    assert set(flask_app.encoders) == set(CATEGORICAL_COLUMNS)


def test_country_and_bin_country_share_one_vocabulary(flask_app):
    """country_mismatch compares these two codes, so the vocabularies must match."""
    enc = flask_app.encoders
    assert enc["country"] == enc["bin_country"], (
        "country and bin_country must share a vocabulary, otherwise the same "
        "country encodes to different integers in each column"
    )


def test_form_options_match_encoder_vocabulary(flask_app):
    """Every dropdown value must be a valid code for that column."""
    choices = flask_app.form_choices()
    for column, classes in flask_app.encoders.items():
        codes = [opt["value"] for opt in choices[column]]
        assert codes == list(range(len(classes))), f"{column} options drifted"


def test_rendered_page_contains_all_ten_countries(client, flask_app):
    """The old UI exposed only 3 of 10 countries; all must be reachable."""
    html = client.get("/").get_data(as_text=True)
    assert len(flask_app.encoders["country"]) == 10
    for code in range(len(flask_app.encoders["country"])):
        assert f'value="{code}"' in html


def test_schema_endpoint_reports_real_categories(client, flask_app):
    body = client.get("/api/schema").get_json()
    countries = body["categorical_fields"]["country"]
    assert set(countries.values()) == set(flask_app.encoders["country"])
    # The dataset has no India; the old UI offered it as country 0.
    assert "IN" not in countries.values()


def test_country_mismatch_derived_from_codes(flask_app, legit_payload):
    same = flask_app.parse_input(legit_payload)
    assert same["country_mismatch"] == 0

    differing = dict(legit_payload)
    differing["bin_country"] = flask_app.encoders["bin_country"].index("US")
    assert flask_app.parse_input(differing)["country_mismatch"] == 1


def test_amount_deviation_derived(flask_app, legit_payload):
    values = flask_app.parse_input(dict(legit_payload, avg_amount_user=100, amount=250))
    assert values["amount_deviation"] == pytest.approx(150)

    values = flask_app.parse_input(dict(legit_payload, avg_amount_user=100, amount=60))
    assert values["amount_deviation"] == pytest.approx(-40)
