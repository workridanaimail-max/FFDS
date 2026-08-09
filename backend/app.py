"""Flask web app and prediction API for the fraud detection model.

The form's dropdown options are generated from the encoder vocabularies saved at
training time, so the values the browser submits always mean what the model
thinks they mean. Hardcoding them is how the old version ended up sending
"India" for a category the model had learned as "DE".
"""
from __future__ import annotations

import json
import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Debug is opt-in. The Werkzeug debugger executes arbitrary code, so it must
# never default to on for something that might get exposed on a network.
DEBUG = os.environ.get("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "5000"))

UNSEEN_CODE = -1

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "..", "frontend", "templates"),
    static_folder=os.path.join(BASE_DIR, "..", "frontend", "static"),
)


def _load(filename: str):
    with open(os.path.join(BASE_DIR, filename), "rb") as fh:
        return pickle.load(fh)


try:
    model = _load("model.pkl")
    scaler = _load("scaler.pkl")
    features_list = _load("features.pkl")
    encoders = _load("encoders.pkl")
except FileNotFoundError as exc:
    raise SystemExit(
        f"Missing model artifact: {exc.filename}\n"
        "Run `python train_model.py` first to generate the model files."
    ) from exc

try:
    with open(os.path.join(BASE_DIR, "metadata.json"), encoding="utf-8") as fh:
        METADATA = json.load(fh)
except FileNotFoundError:
    METADATA = {}

# Cutoff tuned on the validation fold; beats a blind 0.5 for imbalanced data.
THRESHOLD = float(METADATA.get("threshold", 0.5))

# Display names for raw category codes, per column. Anything not listed falls
# back to the code itself, so a new category never breaks the form.
COUNTRY_NAMES = {
    "DE": "Germany", "ES": "Spain", "FR": "France", "GB": "United Kingdom",
    "IT": "Italy", "NL": "Netherlands", "PL": "Poland", "RO": "Romania",
    "TR": "Turkey", "US": "United States", "IN": "India",
}
DISPLAY_NAMES = {
    "country": COUNTRY_NAMES,
    "bin_country": COUNTRY_NAMES,
    "channel": {"app": "Mobile App", "web": "Web"},
    "cvv_result": {"0": "Fail", "1": "Pass"},
}

# Values the form starts with on a fresh GET. Anything the user submits
# overrides these, so a prediction does not wipe the inputs that produced it.
FORM_DEFAULTS = {
    "account_age_days": "365",
    "shipping_distance_km": "25",
    "avg_amount_user": "80",
    "amount": "120",
    "country": "0",
    "bin_country": "0",
    "channel": "0",
    "cvv_result": "0",
    "avs_match": "0",
    "three_ds_flag": "0",
    "promo_used": "0",
}

# Numeric inputs: (field, type, minimum, maximum). Bounds are enforced server
# side -- the HTML min/max attributes are trivially bypassed.
NUMERIC_FIELDS = [
    ("account_age_days", float, 0, 50_000),
    ("avg_amount_user", float, 0, 10_000_000),
    ("amount", float, 0, 10_000_000),
    ("shipping_distance_km", float, 0, 50_000),
]

# Binary flags that are not label-encoded; they are already 0/1 in the data.
BINARY_FIELDS = ["promo_used", "avs_match", "three_ds_flag"]


class ValidationError(ValueError):
    """Raised when submitted input is missing, malformed, or out of range."""


def _select_options(column: str) -> list[dict]:
    """Build dropdown options for *column* straight from the saved vocabulary."""
    names = DISPLAY_NAMES.get(column, {})
    return [
        {"value": code, "label": names.get(category, category)}
        for code, category in enumerate(encoders.get(column, []))
    ]


def form_choices() -> dict[str, list[dict]]:
    choices = {col: _select_options(col) for col in encoders}
    yes_no = [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]
    for field in BINARY_FIELDS:
        choices[field] = yes_no
    return choices


def form_state(submitted=None) -> dict[str, str]:
    """Defaults overlaid with whatever the user actually submitted.

    Values are kept as strings so they round-trip into the HTML unchanged, and
    invalid entries are preserved too -- the user needs to see what they typed
    in order to correct it.
    """
    state = dict(FORM_DEFAULTS)
    for field in FORM_DEFAULTS:
        if submitted and str(submitted.get(field, "")).strip() != "":
            state[field] = str(submitted[field]).strip()
    return state


def _require(data, field: str) -> str:
    if field not in data or str(data[field]).strip() == "":
        raise ValidationError(f"'{field}' is required")
    return str(data[field]).strip()


def parse_input(data) -> dict:
    """Validate and coerce a submitted payload into model features."""
    values = {}

    for field, caster, low, high in NUMERIC_FIELDS:
        raw = _require(data, field)
        try:
            value = caster(float(raw))
        except (TypeError, ValueError):
            raise ValidationError(f"'{field}' must be a number, got {raw!r}")
        if not low <= value <= high:
            raise ValidationError(f"'{field}' must be between {low} and {high}, got {value}")
        values[field] = value

    for field in BINARY_FIELDS:
        raw = _require(data, field)
        try:
            value = int(float(raw))
        except (TypeError, ValueError):
            raise ValidationError(f"'{field}' must be 0 or 1, got {raw!r}")
        if value not in (0, 1):
            raise ValidationError(f"'{field}' must be 0 or 1, got {value}")
        values[field] = value

    for column, classes in encoders.items():
        raw = _require(data, column)
        try:
            code = int(float(raw))
        except (TypeError, ValueError):
            raise ValidationError(f"'{column}' must be an integer code, got {raw!r}")
        if not 0 <= code < len(classes):
            raise ValidationError(
                f"'{column}' code {code} is out of range; valid codes are "
                f"0-{len(classes) - 1} ({', '.join(classes)})"
            )
        values[column] = code

    values["amount_deviation"] = values["amount"] - values["avg_amount_user"]
    # country and bin_country share one vocabulary, so comparing codes is
    # equivalent to comparing the raw country strings used in training.
    values["country_mismatch"] = int(values["country"] != values["bin_country"])
    return values


def predict_one(values: dict) -> dict:
    """Score one validated feature dict."""
    row = pd.DataFrame([[values[col] for col in features_list]], columns=features_list)
    scaled = pd.DataFrame(scaler.transform(row), columns=features_list)
    probability = float(model.predict_proba(scaled)[0][1])
    is_fraud = probability >= THRESHOLD
    return {
        "is_fraud": is_fraud,
        "fraud_probability": round(probability, 4),
        "risk_percent": round(probability * 100, 2),
        "threshold": round(THRESHOLD, 4),
        "label": "Fraudulent Transaction" if is_fraud else "Legitimate Transaction",
    }


@app.route("/")
def home():
    return render_template("index.html", choices=form_choices(),
                           metadata=METADATA, form=form_state())


@app.route("/predict", methods=["POST"])
def predict():
    # Echo the submission back either way, so the form keeps the values the
    # user entered instead of snapping back to defaults.
    state = form_state(request.form)
    try:
        result = predict_one(parse_input(request.form))
    except ValidationError as exc:
        return render_template("index.html", choices=form_choices(),
                               metadata=METADATA, form=state, error=str(exc)), 400
    return render_template("index.html", choices=form_choices(),
                           metadata=METADATA, form=state, result=result)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON prediction endpoint. Accepts a JSON body or form-encoded fields."""
    payload = request.get_json(silent=True) or request.form
    if not payload:
        return jsonify({"error": "request body must be JSON or form-encoded"}), 400
    try:
        result = predict_one(parse_input(payload))
    except ValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": METADATA.get("model", type(model).__name__),
        "resampling": METADATA.get("resampling"),
        "threshold": THRESHOLD,
        "n_features": len(features_list),
        "trained_at": METADATA.get("trained_at"),
        "test_pr_auc": METADATA.get("test_metrics_tuned_threshold", {}).get("pr_auc"),
    })


@app.route("/api/schema")
def api_schema():
    """Describe the accepted fields and the valid code for every category."""
    return jsonify({
        "numeric_fields": [
            {"name": f, "type": c.__name__, "min": lo, "max": hi}
            for f, c, lo, hi in NUMERIC_FIELDS
        ],
        "binary_fields": BINARY_FIELDS,
        "categorical_fields": {
            col: {str(i): c for i, c in enumerate(classes)}
            for col, classes in encoders.items()
        },
        "threshold": THRESHOLD,
    })


if __name__ == "__main__":
    print(f"Model: {METADATA.get('model', type(model).__name__)} | threshold {THRESHOLD:.4f}")
    print(f"Serving on http://{HOST}:{PORT}  (debug={DEBUG})")
    app.run(host=HOST, port=PORT, debug=DEBUG)
