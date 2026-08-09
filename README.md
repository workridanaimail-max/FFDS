# 💳 E-Commerce Fraud Detection System

A machine learning web application that detects fraudulent e-commerce transactions in real time. Users enter transaction details through a web interface and receive a calibrated fraud risk score from a model trained on ~300,000 transactions.

---

## 📁 Project Structure

```
ecommerce-fraud-detection/
│
├── data/
│   └── ecommerce_fraud.csv        # Any compatible CSV here is picked up automatically
│
├── backend/
│   ├── data_loader.py             # Dataset discovery, schema validation, merging
│   ├── train_model.py             # Training pipeline (16-configuration bake-off)
│   ├── ensembles.py               # Leakage-safe temporal stacking estimator
│   ├── app.py                     # Flask web server + JSON API
│   ├── model.pkl                  # Trained model
│   ├── scaler.pkl                 # StandardScaler
│   ├── features.pkl               # Feature column order
│   ├── encoders.pkl               # Category vocabularies (drives the UI dropdowns)
│   └── metadata.json              # Metrics, tuned threshold, training provenance
│
├── frontend/
│   ├── templates/index.html       # Web UI (dropdowns generated from encoders.pkl)
│   └── static/style.css
│
├── tests/                         # pytest suite
└── requirements.txt
```

---

## 🧠 ML Pipeline

### 1. Data loading
`data_loader.py` scans `data/` and loads **every** CSV that satisfies the required schema, unions them, drops cross-file duplicates by `transaction_id`, and tags each row with its `source_file`. Files that do not match the schema are reported and skipped rather than silently corrupting the training set.

**To add another dataset, drop a compatible CSV into `data/` and retrain — no code changes.** Required columns:

```
account_age_days, avg_amount_user, amount, shipping_distance_km,
promo_used, avs_match, three_ds_flag,
country, bin_country, channel, cvv_result,
transaction_time, is_fraud
```

(`transaction_time` is required for the chronological split even though the hour itself is not a model feature.)

### 2. Feature engineering
| Feature | Description |
|---|---|
| `amount_deviation` | Transaction amount minus the user's average |
| `country_mismatch` | Flag: billing country ≠ card BIN country |

### 2b. Feature selection
Permutation importance on the held-out fold showed five features carrying 91% of the signal:

| Feature | Share of PR-AUC |
|---|---|
| `shipping_distance_km` | 35.5% |
| `amount` | 18.0% |
| `amount_deviation` | 13.8% |
| `country_mismatch` | 13.7% |
| `account_age_days` | 10.1% |

`hour`, `total_transactions_user`, and `merchant_category` each contributed ~0%, and `high_amount` was redundant with the continuous `amount_deviation`. Dropping all four **raised** test PR-AUC from 0.8624 to 0.8643 and precision from 0.920 to 0.932, so the form no longer collects them.

Note that `country`, `bin_country`, and `avg_amount_user` score near zero *individually* but are retained: they feed `country_mismatch` and `amount_deviation`, which together carry 27.5%. Cutting further is costly — an 8-input form loses 1.5pp of PR-AUC, and a 6-input form loses 6.7pp with recall falling to 0.68.

### 3. Temporal split
The data is split **chronologically**, not randomly — train on the earliest 70%, validate on the next 15%, test on the most recent 15%. Fraud patterns drift over time, so a random split lets the model peek at the future and overstates real-world performance.

### 4. Encoding
Category vocabularies are fit on the **training fold only** and saved to `encoders.pkl`. `country` and `bin_country` share a single vocabulary so their codes stay directly comparable. Categories unseen during training encode to `-1`.

The web form's dropdown options are generated from `encoders.pkl` at request time, which makes it structurally impossible for the UI to send a code that means something different to the model.

### 5. Class imbalance
The dataset is ~2.2% fraud. Two strategies are evaluated head to head:
- **`class_weight="balanced"`** — reweight the loss
- **SMOTE** — synthetic oversampling, applied to the **training fold only**, after the split

Applying SMOTE before splitting leaks synthetic neighbours of test rows into training and inflates every metric. On this dataset `class_weight` beat SMOTE for all three model families.

### 6. Model selection
**Sixteen configurations** are ranked by **validation** PR-AUC — seven model families × two balancing strategies, plus two ensembles:

| Family | Library |
|---|---|
| Logistic Regression | scikit-learn |
| Random Forest | scikit-learn |
| Balanced Random Forest | imbalanced-learn |
| HistGradientBoosting | scikit-learn |
| LightGBM | lightgbm |
| XGBoost | xgboost |
| CatBoost | catboost |

Plus a soft-voting ensemble and a stacking ensemble (see below). The test fold is scored exactly once, at the end, so its numbers stay honest.

At a 2.2% fraud rate, accuracy and ROC-AUC both flatter a weak model — predicting "never fraud" scores 97.8% accuracy. PR-AUC is the headline metric.

### 6b. Ensembles
Two ensembles are built from the two strongest boosters plus a Random Forest, chosen for family diversity — averaging four gradient boosters would just average four copies of the same inductive bias.

- **Soft voting** — equal-weight probability average. Logistic Regression is deliberately excluded: an early run including it scored 0.8198, well below the best single model, because equal weighting let a 0.60 model drag the mean down.
- **Stacking** — a Logistic Regression meta-learner over base model probabilities, which learns the weights instead of assuming them.

`TemporalStackingClassifier` in [`ensembles.py`](backend/ensembles.py) exists because scikit-learn's `StackingClassifier` builds meta-features with `cross_val_predict`, which requires a *partitioning* CV splitter. `TimeSeriesSplit` does not partition — the earliest block is never in a test fold — so it raises `cross_val_predict only works for partitions`. Falling back to the default `StratifiedKFold` would fix the error by letting the meta-learner train on out-of-fold predictions drawn from the future, reintroducing exactly the leakage this pipeline removes. Instead it fits base models on the earliest 75% of the training fold, fits the meta-learner on their predictions over the final 25%, then refits the bases on the whole fold for deployment.

### 7. Threshold tuning
The decision cutoff is tuned on validation to maximise F1 instead of defaulting to 0.5, then saved to `metadata.json` and applied at serving time. On this model that moves test precision from **0.317 → 0.920**.

---

## 📊 Results

Trained on 299,695 transactions. Held-out test fold = the most recent 44,955 transactions.

Full leaderboard, all 16 configurations:

| Rank | Model | Strategy | Val PR-AUC |
|---|---|---|---|
| **1** | **HistGradientBoosting** | **class_weight** | **0.8721** |
| 2 | CatBoost | class_weight | 0.8672 |
| 3 | Voting (HGB + CatBoost + RF) | class_weight | 0.8668 |
| 4 | LightGBM | class_weight | 0.8649 |
| 5 | Stacking (HGB + CatBoost + RF + LR) | class_weight | 0.8648 |
| 6 | XGBoost | class_weight | 0.8642 |
| 7 | XGBoost | SMOTE | 0.8619 |
| 8 | CatBoost | SMOTE | 0.8578 |
| 9 | LightGBM | SMOTE | 0.8570 |
| 10 | HistGradientBoosting | SMOTE | 0.8562 |
| 11 | Random Forest | class_weight | 0.8510 |
| 12 | Balanced Random Forest | SMOTE | 0.8388 |
| 13 | Random Forest | SMOTE | 0.8371 |
| 14 | Balanced Random Forest | class_weight | 0.8293 |
| 15 | Logistic Regression | SMOTE | 0.6034 |
| 16 | Logistic Regression | class_weight | 0.5998 |

Three findings worth recording:

- **Neither ensemble beat the best single model.** Voting landed 3rd (−0.005) and stacking 5th (−0.007). The stacking meta-learner's coefficients (HGB +2.38, CatBoost +3.17, RF +2.24, LR +0.22) show it correctly downweighted Logistic Regression almost to zero, but the four boosters agree too closely for combining them to add information.
- **`class_weight` beat SMOTE for six of seven families.** XGBoost was the sole exception, and only marginally.
- **The newer boosting libraries did not beat scikit-learn's built-in.** CatBoost, LightGBM, and XGBoost all landed within 0.008 of HistGradientBoosting without winning — at defaults, on 13 features, they are effectively the same model.

Test metrics for the winner (HistGradientBoosting + class_weight, 13 features):

| Metric | @ 0.50 | @ tuned (0.9687) |
|---|---|---|
| Accuracy | 0.9556 | 0.9934 |
| Precision | 0.3172 | **0.9315** |
| Recall | 0.9069 | 0.7513 |
| F1 | 0.4700 | **0.8317** |
| PR-AUC | 0.8643 | 0.8643 |
| ROC-AUC | 0.9777 | 0.9777 |

Caught **734 of 977** frauds with only **54 false alarms** across 43,978 legitimate transactions.

---

## 🌐 Endpoints

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Prediction form |
| `POST` | `/predict` | Form submission → rendered result (400 on invalid input) |
| `POST` | `/api/predict` | JSON in, JSON out |
| `GET` | `/api/schema` | Accepted fields and the valid code for every category |
| `GET` | `/health` | Model name, threshold, training provenance |

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"account_age_days":1,"avg_amount_user":40,"amount":4000,
       "shipping_distance_km":9000,"country":7,"bin_country":9,"channel":0,
       "cvv_result":0,"promo_used":1,"avs_match":0,"three_ds_flag":0}'
```

```json
{"fraud_probability": 0.999, "is_fraud": true, "label": "Fraudulent Transaction",
 "risk_percent": 99.9, "threshold": 0.9687}
```

Every input is validated server side — out-of-range values, bad category codes, and missing fields return **400** with a readable message instead of a 200 carrying a stack trace.

---

## ⚙️ Setup

### Prerequisites
Python 3.9+

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Train (optional — artifacts are committed)
```bash
python backend/train_model.py
```
Generates `model.pkl`, `scaler.pkl`, `features.pkl`, `encoders.pkl`, and `metadata.json`.

Tunables for constrained machines:
```bash
RF_TREES=50 N_JOBS=2 python backend/train_model.py
```

### 3. Run
```bash
python backend/app.py
```
Open <http://127.0.0.1:5000>.

Configuration is environment-driven:

| Variable | Default | Notes |
|---|---|---|
| `FLASK_DEBUG` | `0` | Off by default — the Werkzeug debugger executes arbitrary code |
| `HOST` | `127.0.0.1` | |
| `PORT` | `5000` | |

### 4. Test
```bash
python -m pytest tests -q
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| ML | scikit-learn, LightGBM, XGBoost, CatBoost |
| Ensembles | Soft voting, temporal stacking |
| Imbalance | imbalanced-learn (SMOTE, BalancedRandomForest), class weighting |
| Web | Flask |
| Frontend | HTML, CSS, Jinja2 |
| Data | Pandas, NumPy |
| Tests | pytest |

---

## 📊 Dataset

`ecommerce_fraud.csv` — 299,695 transactions spanning 2024-01-01 to 2024-10-31, 6,612 of them fraudulent (2.206%).

Categorical values used by the model:
- **country / bin_country**: DE, ES, FR, GB, IT, NL, PL, RO, TR, US
- **channel**: app, web
- **cvv_result**: 0 (fail), 1 (pass)

The CSV also carries `merchant_category`, `total_transactions_user`, and `transaction_id`/`user_id`; the first two were dropped as uninformative (see Feature selection) and identifiers are never used as features.

### A note on adding external datasets
Public fraud datasets (ULB creditcard, PaySim, IEEE-CIS) share essentially no features with this schema — ULB's are PCA-anonymised `V1…V28`, and none carry `bin_country`, `avs_match`, `cvv_result`, or `three_ds_flag`. Merging them would produce a table that is mostly missing values with inconsistent fraud semantics, degrading the model rather than improving it. The loader validates schemas and skips incompatible files for exactly this reason.

---

## 👩‍💻 Author

**Sai Bhargavi Rapolu**
[GitHub](https://github.com/saibhargavi-rapolu)
