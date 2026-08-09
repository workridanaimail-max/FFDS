import os
import sys

import pytest

BACKEND = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, BACKEND)


@pytest.fixture(scope="session")
def flask_app():
    import app as app_module

    app_module.app.config.update(TESTING=True)
    return app_module


@pytest.fixture(scope="session")
def client(flask_app):
    return flask_app.app.test_client()


@pytest.fixture
def legit_payload(flask_app):
    """A low-risk profile: established account, normal amount, everything matches."""
    enc = flask_app.encoders
    return {
        "account_age_days": 1200,
        "avg_amount_user": 80,
        "amount": 75,
        "shipping_distance_km": 8,
        "country": enc["country"].index("DE"),
        "bin_country": enc["bin_country"].index("DE"),
        "channel": enc["channel"].index("web"),
        "cvv_result": enc["cvv_result"].index("1"),
        "promo_used": 0,
        "avs_match": 1,
        "three_ds_flag": 1,
    }


@pytest.fixture
def risky_payload(flask_app):
    """A high-risk profile: new account, huge amount, country mismatch, CVV fail."""
    enc = flask_app.encoders
    return {
        "account_age_days": 1,
        "avg_amount_user": 40,
        "amount": 4000,
        "shipping_distance_km": 9000,
        "country": enc["country"].index("RO"),
        "bin_country": enc["bin_country"].index("US"),
        "channel": enc["channel"].index("app"),
        "cvv_result": enc["cvv_result"].index("0"),
        "promo_used": 1,
        "avs_match": 0,
        "three_ds_flag": 0,
    }
