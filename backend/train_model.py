import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from imblearn.over_sampling import SMOTE

# =========================
# 1. LOAD DATA (SAFE PATH)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "../data/ecommerce_fraud.csv")

df = pd.read_csv(file_path)
df = df.sample(50000, random_state=42)

print("\n========== DATASET LOADED ==========")
print("Shape:", df.shape)

# =========================
# 2. INITIAL CLASS DISTRIBUTION
# =========================
print("\n========== INITIAL CLASS DISTRIBUTION ==========")
print(df["is_fraud"].value_counts())

# =========================
# 3. MISSING VALUES
# =========================
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# FIXED (NO inplace warning)
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

print("\nAfter Handling Missing Values:")
print(df.isnull().sum())

# =========================
# 4. FEATURE ENGINEERING
# =========================
df["amount_deviation"] = df["amount"] - df["avg_amount_user"]
df["high_amount"] = (df["amount"] > df["avg_amount_user"] * 2).astype(int)
df["country_mismatch"] = (df["country"] != df["bin_country"]).astype(int)

df["transaction_time"] = pd.to_datetime(df["transaction_time"])
df["hour"] = df["transaction_time"].dt.hour

print("\n========== FEATURE ENGINEERING DONE ==========")

# =========================
# 5. ENCODING
# =========================
le = LabelEncoder()

for col in ["country", "bin_country", "channel", "merchant_category", "cvv_result"]:
    df[col] = le.fit_transform(df[col])

print("\n========== ENCODING DONE ==========")

# =========================
# 6. DROP UNUSED
# =========================
df.drop(["transaction_id", "user_id", "transaction_time"], axis=1, inplace=True)

# =========================
# 7. SPLIT FEATURES
# =========================
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]
pickle.dump(X.columns.tolist(), open("features.pkl", "wb"))

print("\n========== FEATURE LIST SAVED ==========")
print(X.columns.tolist())
# =========================
# 8. SMOTE
# =========================
print("\n========== BEFORE SMOTE ==========")
print(y.value_counts())

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

print("\n========== AFTER SMOTE ==========")
print(y.value_counts())

# =========================
# 9. MODEL TRAINING
# =========================
splits = [0.2, 0.25]

best_score = 0
best_model = None
best_scaler = None
best_config = ""

results = []

print("\n========== MODEL TRAINING ==========")

for split in splits:

    split_name = f"{int((1-split)*100)}:{int(split*100)}"
    print(f"\n\n===== TRAIN TEST SPLIT: {split_name} =====")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=50, n_jobs=-1)
    }

    for name, model in models.items():

        model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc = roc_auc_score(y_test, probs)

        print(f"\n{name}")
        print(f"Accuracy   : {acc:.4f}")
        print(f"Precision  : {prec:.4f}")
        print(f"Recall     : {rec:.4f}")
        print(f"F1 Score   : {f1:.4f}")
        print(f"ROC-AUC    : {roc:.4f}")

        results.append([split_name, name, acc, prec, rec, f1, roc])

        if roc > best_score:
            best_score = roc
            best_model = model
            best_scaler = scaler
            best_config = f"{name} with {split_name} split"

# =========================
# 10. FINAL RESULTS TABLE
# =========================
print("\n========== FINAL RESULTS SUMMARY ==========")

results_df = pd.DataFrame(results, columns=[
    "Split", "Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"
])

print(results_df)

# =========================
# 11. SAVE BEST MODEL
# =========================
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(best_scaler, open("scaler.pkl", "wb"))

print("\n========== FINAL BEST MODEL ==========")
print("Best Model:", best_config)
print("Best ROC-AUC:", round(best_score, 4))
print("Model saved successfully ✅")