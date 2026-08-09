import json
import os
import pickle
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="E-Commerce Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

# Custom CSS for Premium Design
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stAppHeader {
        background-color: transparent;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .fraud-alert {
        background-color: rgba(255, 75, 75, 0.15);
        border: 1px solid #ff4b4b;
        border-radius: 12px;
        padding: 20px;
        color: #ff4b4b;
        margin-top: 20px;
    }
    .legit-alert {
        background-color: rgba(33, 195, 84, 0.15);
        border: 1px solid #21c354;
        border-radius: 12px;
        padding: 20px;
        color: #21c354;
        margin-top: 20px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1f77b4, #00d2ff);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #00d2ff, #1f77b4);
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    with open(os.path.join(BACKEND_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BACKEND_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(BACKEND_DIR, "features.pkl"), "rb") as f:
        features_list = pickle.load(f)
    with open(os.path.join(BACKEND_DIR, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    with open(os.path.join(BACKEND_DIR, "metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    return model, scaler, features_list, encoders, metadata

try:
    model, scaler, features_list, encoders, metadata = load_artifacts()
    THRESHOLD = float(metadata.get("threshold", 0.9687))
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Country mapping
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

# Sidebar - Model Provenance
with st.sidebar:
    st.title("💳 ML Model Info")
    st.markdown(f"**Model Architecture:** `{metadata.get('model', 'Hist Gradient Boosting')}`")
    st.markdown(f"**Sampling Strategy:** `{metadata.get('resampling', 'class_weight')}`")
    st.markdown(f"**Tuned Threshold:** `{THRESHOLD:.4f}`")
    st.markdown(f"**Validation PR-AUC:** `{metadata.get('validation_pr_auc', 0.8721):.4f}`")
    
    st.markdown("---")
    st.subheader("📊 Dataset Statistics")
    st.write(f"- Total Training Rows: **{metadata.get('n_rows_train', 209786):,}**")
    st.write(f"- Features Evaluated: **{len(features_list)}**")
    st.write(f"- Test Precision: **{metadata.get('test_metrics_tuned_threshold', {}).get('precision', 0.9315):.2%}**")
    st.write(f"- Test Recall: **{metadata.get('test_metrics_tuned_threshold', {}).get('recall', 0.7513):.2%}**")

# Main Interface
st.title("💳 E-Commerce Real-Time Fraud Detection System")
st.markdown("Enter transaction details below to evaluate real-time fraud risk score calibrated by machine learning.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 User & Transaction Info")
    account_age_days = st.number_input("Account Age (Days)", min_value=0, max_value=50000, value=365)
    avg_amount_user = st.number_input("User Average Transaction Amount ($)", min_value=0.0, max_value=10000000.0, value=80.0)
    amount = st.number_input("Current Transaction Amount ($)", min_value=0.0, max_value=10000000.0, value=120.0)
    shipping_distance_km = st.number_input("Shipping Distance (km)", min_value=0.0, max_value=50000.0, value=25.0)

with col2:
    st.subheader("⚙️ Verification & Location")
    
    # Billing Country
    country_vocab = encoders.get("country", [])
    country_options = {i: DISPLAY_NAMES["country"].get(c, c) for i, c in enumerate(country_vocab)}
    selected_country = st.selectbox("Billing Country", options=list(country_options.keys()), format_func=lambda x: country_options[x])

    # BIN Country
    bin_vocab = encoders.get("bin_country", [])
    bin_options = {i: DISPLAY_NAMES["bin_country"].get(c, c) for i, c in enumerate(bin_vocab)}
    selected_bin_country = st.selectbox("Card BIN Country", options=list(bin_options.keys()), format_func=lambda x: bin_options[x])

    # Channel
    channel_vocab = encoders.get("channel", [])
    channel_options = {i: DISPLAY_NAMES["channel"].get(c, c) for i, c in enumerate(channel_vocab)}
    selected_channel = st.selectbox("Channel", options=list(channel_options.keys()), format_func=lambda x: channel_options[x])

    # Security Verification Flags
    cvv_vocab = encoders.get("cvv_result", [])
    cvv_options = {i: DISPLAY_NAMES["cvv_result"].get(c, c) for i, c in enumerate(cvv_vocab)}
    selected_cvv = st.selectbox("CVV Verification Result", options=list(cvv_options.keys()), format_func=lambda x: cvv_options[x])

    c_sub1, c_sub2, c_sub3 = st.columns(3)
    with c_sub1:
        avs_match = st.selectbox("AVS Match", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with c_sub2:
        three_ds_flag = st.selectbox("3D Secure", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with c_sub3:
        promo_used = st.selectbox("Promo Used", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.markdown("---")

if st.button("🔍 Assess Fraud Risk"):
    # Feature Engineering & Scaling
    amount_deviation = amount - avg_amount_user
    country_mismatch = int(selected_country != selected_bin_country)

    raw_values = {
        "account_age_days": account_age_days,
        "avg_amount_user": avg_amount_user,
        "amount": amount,
        "shipping_distance_km": shipping_distance_km,
        "promo_used": promo_used,
        "avs_match": avs_match,
        "three_ds_flag": three_ds_flag,
        "country": selected_country,
        "bin_country": selected_bin_country,
        "channel": selected_channel,
        "cvv_result": selected_cvv,
        "amount_deviation": amount_deviation,
        "country_mismatch": country_mismatch,
    }

    row = pd.DataFrame([[raw_values[col] for col in features_list]], columns=features_list)
    scaled = pd.DataFrame(scaler.transform(row), columns=features_list)
    probability = float(model.predict_proba(scaled)[0][1])
    is_fraud = probability >= THRESHOLD

    # Output Results
    st.subheader("🎯 Risk Assessment Result")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric(
            label="Calculated Fraud Probability",
            value=f"{probability * 100:.2f}%",
            delta=f"{'FLAGGED' if is_fraud else 'CLEARED'} (Cutoff: {THRESHOLD * 100:.1f}%)",
            delta_color="inverse" if is_fraud else "normal"
        )
    
    with res_col2:
        st.progress(probability)
        if is_fraud:
            st.markdown(f"""
                <div class="fraud-alert">
                    <h3>🚨 FRAUDULENT TRANSACTION DETECTED</h3>
                    <p>Risk Score: <b>{probability * 100:.2f}%</b> (Exceeds decision threshold of {THRESHOLD * 100:.2f}%).</p>
                    <p>Recommendation: Flag for manual review or decline authorization.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="legit-alert">
                    <h3>✅ LEGITIMATE TRANSACTION</h3>
                    <p>Risk Score: <b>{probability * 100:.2f}%</b> (Below decision threshold of {THRESHOLD * 100:.2f}%).</p>
                    <p>Recommendation: Safe to process.</p>
                </div>
            """, unsafe_allow_html=True)

    # Risk Drivers Summary
    st.markdown("#### 🔍 Derived Model Signals")
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.info(f"**Amount Deviation:** ${amount_deviation:+.2f} relative to user average")
    with d_col2:
        if country_mismatch:
            st.warning("⚠️ **Country Mismatch Detected:** Billing country differs from Card BIN country.")
        else:
            st.success("✅ **Country Match:** Billing country matches Card BIN country.")
