"""Endpoint, validation, and sanity tests."""
import re
import warnings

import pytest


# --------------------------------------------------------------------------
# Health and artifacts
# --------------------------------------------------------------------------
def test_health_ok(client):
    body = client.get("/health").get_json()
    assert body["status"] == "ok"
    assert body["n_features"] == 13
    assert 0 < body["threshold"] <= 1


def test_home_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Fraud Detection System" in resp.get_data(as_text=True)


def test_feature_order_matches_model(flask_app):
    assert len(flask_app.features_list) == 13
    assert flask_app.features_list[0] == "account_age_days"


def test_dropped_fields_are_gone(flask_app, client):
    """These three carried ~0% importance and were removed from the form."""
    from data_loader import EXCLUDED_FEATURES

    html = client.get("/").get_data(as_text=True)
    for field in EXCLUDED_FEATURES:
        assert field not in flask_app.features_list
        assert f'name="{field}"' not in html
    for label in ["Total Transactions", "Transaction Hour", "Merchant Category"]:
        assert label not in html


def test_dropped_fields_are_ignored_not_rejected(client, legit_payload):
    """An old client still sending the removed fields must keep working."""
    stale = dict(legit_payload, hour=3, total_transactions_user=99, merchant_category=1)
    assert client.post("/api/predict", json=stale).status_code == 200


# --------------------------------------------------------------------------
# Prediction behaviour
# --------------------------------------------------------------------------
def test_api_predict_returns_probability(client, legit_payload):
    body = client.post("/api/predict", json=legit_payload).get_json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_percent"] == pytest.approx(body["fraud_probability"] * 100, abs=0.01)
    assert isinstance(body["is_fraud"], bool)


def test_risky_profile_scores_higher_than_legit(client, legit_payload, risky_payload):
    legit = client.post("/api/predict", json=legit_payload).get_json()
    risky = client.post("/api/predict", json=risky_payload).get_json()
    assert risky["fraud_probability"] > legit["fraud_probability"], (
        f"risky profile scored {risky['fraud_probability']} but legit scored "
        f"{legit['fraud_probability']}"
    )


def test_is_fraud_follows_threshold(client, risky_payload):
    body = client.post("/api/predict", json=risky_payload).get_json()
    assert body["is_fraud"] == (body["fraud_probability"] >= body["threshold"])


def test_form_predict_renders_result(client, legit_payload):
    resp = client.post("/predict", data=legit_payload)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Fraud risk" in html
    assert "Transaction" in html


# --------------------------------------------------------------------------
# The form must not reset to defaults after submitting
# --------------------------------------------------------------------------
def test_numeric_inputs_keep_submitted_values(client, legit_payload):
    payload = dict(legit_payload, amount=1234.5, avg_amount_user=99,
                   account_age_days=7, shipping_distance_km=888)
    html = client.post("/predict", data=payload).get_data(as_text=True)
    for field, value in [("amount", "1234.5"), ("avg_amount_user", "99"),
                         ("account_age_days", "7"), ("shipping_distance_km", "888")]:
        assert f'name="{field}"' in html
        assert f'value="{value}"' in html, f"{field} reset instead of keeping {value}"


def _selected_option(html, name):
    """Return the value of the <option> marked selected inside <select name>."""
    block = re.search(rf'<select id="{name}".*?</select>', html, re.S)
    assert block, f"no <select> named {name}"
    chosen = re.search(r'<option value="(\d+)"\s+selected', block.group(0))
    return chosen.group(1) if chosen else None


def test_dropdowns_keep_submitted_selection(client, flask_app, legit_payload):
    us = flask_app.encoders["country"].index("US")
    gb = flask_app.encoders["bin_country"].index("GB")
    payload = dict(legit_payload, country=us, bin_country=gb,
                   avs_match=0, three_ds_flag=1, promo_used=1)
    html = client.post("/predict", data=payload).get_data(as_text=True)

    assert _selected_option(html, "country") == str(us)
    assert _selected_option(html, "bin_country") == str(gb)
    assert _selected_option(html, "avs_match") == "0"
    assert _selected_option(html, "three_ds_flag") == "1"
    assert _selected_option(html, "promo_used") == "1"


def test_every_dropdown_has_exactly_one_selection(client, flask_app, legit_payload):
    html = client.post("/predict", data=legit_payload).get_data(as_text=True)
    for name in list(flask_app.encoders) + flask_app.BINARY_FIELDS:
        block = re.search(rf'<select id="{name}".*?</select>', html, re.S).group(0)
        assert block.count(" selected") == 1, f"{name} has ambiguous selection"


def test_defaults_used_on_fresh_get(client, flask_app):
    html = client.get("/").get_data(as_text=True)
    assert f'value="{flask_app.FORM_DEFAULTS["account_age_days"]}"' in html
    assert f'value="{flask_app.FORM_DEFAULTS["amount"]}"' in html
    assert _selected_option(html, "country") == flask_app.FORM_DEFAULTS["country"]


def test_invalid_submission_preserves_inputs(client, legit_payload):
    """A validation error must not wipe the fields the user already filled in."""
    payload = dict(legit_payload, amount=4321, shipping_distance_km=999_999)
    resp = client.post("/predict", data=payload)
    assert resp.status_code == 400
    html = resp.get_data(as_text=True)
    assert 'value="4321"' in html, "valid field was reset by an unrelated error"
    assert 'value="999999"' in html, "the offending value must stay visible to correct"


def test_predict_emits_no_feature_name_warning(flask_app, legit_payload):
    """Passing a bare list to the scaler used to warn on every prediction."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        flask_app.predict_one(flask_app.parse_input(legit_payload))


# --------------------------------------------------------------------------
# Input validation -- all of these used to return 200 with a raw traceback
# --------------------------------------------------------------------------
@pytest.mark.parametrize("field,value,reason", [
    ("shipping_distance_km", 999_999, "distance above max"),
    ("account_age_days", -1, "negative account age"),
    ("amount", -50, "negative amount"),
    ("account_age_days", "abc", "non-numeric"),
    ("country", 999, "country code out of range"),
    ("country", -1, "negative country code"),
    ("avs_match", 7, "binary flag not 0/1"),
])
def test_invalid_input_returns_400(client, legit_payload, field, value, reason):
    payload = dict(legit_payload, **{field: value})
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 400, f"expected 400 for {reason}"
    assert "error" in resp.get_json()


def test_missing_field_returns_400(client, legit_payload):
    payload = dict(legit_payload)
    del payload["amount"]
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 400
    assert "amount" in resp.get_json()["error"]


def test_empty_body_returns_400(client):
    assert client.post("/api/predict", json={}).status_code == 400


def test_form_validation_error_renders_400_page(client, legit_payload):
    resp = client.post("/predict", data=dict(legit_payload, shipping_distance_km=999_999))
    assert resp.status_code == 400
    assert "must be between" in resp.get_data(as_text=True)


def test_error_message_does_not_leak_traceback(client, legit_payload):
    payload = dict(legit_payload, shipping_distance_km=999_999)
    body = client.post("/api/predict", json=payload).get_json()
    assert "Traceback" not in body["error"]
    assert "File \"" not in body["error"]
