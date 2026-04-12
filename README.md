# 💳 E-Commerce Fraud Detection System

A machine learning web application that detects fraudulent e-commerce transactions in real time. Users can input transaction details through a clean web interface and instantly receive a fraud prediction powered by a trained ML model.

---

## 🚀 Demo

> Enter transaction details → Click **Detect Fraud** → Get instant prediction: ✅ Legitimate or ⚠️ Fraudulent

---

## 📁 Project Structure

```
ecommerce-fraud-detection/
│
├── data/
│   └── ecommerce_fraud.csv        # Dataset with transaction records
│
├── backend/
│   ├── train_model.py             # Model training pipeline
│   ├── app.py                     # Flask web server & prediction API
│   ├── model.pkl                  # Trained best model (saved)
│   ├── scaler.pkl                 # StandardScaler (saved)
│   └── features.pkl               # Feature column order (saved)
│
├── frontend/
│   ├── templates/
│   │   └── index.html             # Web UI form
│   └── static/
│       └── style.css              # Styling
│
└── requirements.txt               # Python dependencies
```

---

## 🧠 ML Pipeline

### 1. Data Loading
- Loads `ecommerce_fraud.csv` and samples 50,000 records for training.

### 2. Data Preprocessing
- Handles missing values: mode imputation for categorical columns, median for numerical.
- Encodes categorical features: `country`, `bin_country`, `channel`, `merchant_category`, `cvv_result` using `LabelEncoder`.

### 3. Feature Engineering
| Feature | Description |
|---|---|
| `amount_deviation` | Difference between transaction amount and user's average |
| `high_amount` | Flag: transaction > 2× user's average amount |
| `country_mismatch` | Flag: billing country ≠ card BIN country |
| `hour` | Hour extracted from transaction timestamp |

### 4. Class Imbalance Handling
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the fraud vs. non-fraud classes before training.

### 5. Model Training & Selection
Two models are trained and evaluated across two train/test splits (80:20, 75:25):
- **Logistic Regression**
- **Random Forest** (50 estimators)

Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.

The best model (by ROC-AUC) is automatically saved as `model.pkl`.

---

## 🌐 Web Application

Built with **Flask**, the app exposes:

- `GET /` — Renders the prediction form
- `POST /predict` — Accepts transaction input, scales it, runs the model, and returns the result

### Input Features (via form)
| Field | Type |
|---|---|
| Account Age (days) | Numeric |
| Total Transactions | Numeric |
| Average Amount | Numeric |
| Transaction Amount | Numeric |
| Shipping Distance (km) | Numeric |
| Transaction Hour | Numeric (0–23) |
| Country | Dropdown |
| BIN Country | Dropdown |
| Channel | Dropdown (Web / Mobile) |
| Merchant Category | Dropdown |
| Promo Used | Dropdown (Yes / No) |
| AVS Match | Dropdown (Yes / No) |
| CVV Result | Dropdown (Pass / Fail) |
| 3D Secure | Dropdown (Yes / No) |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/saibhargavi-rapolu/ecommerce-fraud-detection.git
cd ecommerce-fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
cd backend
python train_model.py
```
This generates `model.pkl`, `scaler.pkl`, and `features.pkl` inside the `backend/` folder.

### 4. Run the Flask app
```bash
python app.py
```

### 5. Open in browser
```
http://127.0.0.1:5000
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| ML Models | Scikit-learn (Logistic Regression, Random Forest) |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Web Framework | Flask |
| Frontend | HTML, CSS |
| Data Processing | Pandas, NumPy |
| Model Persistence | Pickle |

---

## 📊 Dataset

The dataset (`ecommerce_fraud.csv`) contains transaction-level records with the following columns:

`transaction_id`, `user_id`, `account_age_days`, `total_transactions_user`, `avg_amount_user`, `amount`, `country`, `bin_country`, `channel`, `merchant_category`, `promo_used`, `avs_match`, `cvv_result`, `three_ds_flag`, `transaction_time`, `shipping_distance_km`, `is_fraud`

Target column: `is_fraud` (0 = Legitimate, 1 = Fraudulent)

---

## 👩‍💻 Author

**Sai Bhargavi Rapolu**  
[GitHub](https://github.com/saibhargavi-rapolu)
