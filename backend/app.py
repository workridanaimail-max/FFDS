from flask import Flask, request, render_template
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "../frontend/templates"),
    static_folder=os.path.join(BASE_DIR, "../frontend/static")
)

# Load model + scaler + features
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features_list = pickle.load(open("features.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        amount = float(data["amount"])
        avg_amount = float(data["avg_amount_user"])

        input_dict = {
            "account_age_days": float(data["account_age_days"]),
            "total_transactions_user": float(data["total_transactions_user"]),
            "avg_amount_user": avg_amount,
            "amount": amount,
            "country": int(data["country"]),
            "bin_country": int(data["bin_country"]),
            "channel": int(data["channel"]),
            "merchant_category": int(data["merchant_category"]),
            "promo_used": int(data["promo_used"]),
            "avs_match": int(data["avs_match"]),
            "cvv_result": int(data["cvv_result"]),
            "three_ds_flag": int(data["three_ds_flag"]),
            "shipping_distance_km": float(data["shipping_distance_km"]),
            "amount_deviation": amount - avg_amount,
            "high_amount": int(amount > avg_amount * 2),
            "country_mismatch": int(data["country"]) != int(data["bin_country"]),
            "hour": int(data["hour"])
        }

        # Maintain correct order
        input_data = [input_dict[col] for col in features_list]

        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)[0]

        result = "⚠️ Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)