import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# ============================================
# PCOSence - Machine Learning Model Training
# ============================================

print("=" * 50)
print("PCOSence - Model Training")
print("=" * 50)

# --------------------------------------------
# Load Dataset
# --------------------------------------------

dataset_path = os.path.join("..", "data", "PCOSData.xlsx")

print("\nLoading Dataset...")

df = pd.read_excel(dataset_path)

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# --------------------------------------------
# Data Cleaning
# --------------------------------------------

print("\nCleaning Dataset...")

# Remove unwanted column if present
if "Unnamed: 44" in df.columns:
    df.drop(columns=["Unnamed: 44"], inplace=True)

# Remove ID columns
drop_cols = [
    "Sl. No",
    "Patient File No."
]

for col in drop_cols:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Convert all columns except target to numeric
for col in df.columns:

    if col != "PCOS (Y/N)":

        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[^0-9.-]", "", regex=True)
        )

        df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

print("Cleaning Complete!")

# --------------------------------------------
# Features and Target
# --------------------------------------------

X = df.drop("PCOS (Y/N)", axis=1)

y = df["PCOS (Y/N)"]

print("\nNumber of Features:", X.shape[1])

# --------------------------------------------
# Save Feature Names
# --------------------------------------------

os.makedirs("model", exist_ok=True)

joblib.dump(list(X.columns), "model/feature_names.pkl")

# --------------------------------------------
# Train Test Split
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# --------------------------------------------
# Feature Scaling
# --------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")

# ============================================
# Logistic Regression (Primary Model)
# ============================================

print("\n==============================")
print("Training Logistic Regression")
print("==============================")

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)

log_acc = accuracy_score(y_test, log_pred)

print("\nAccuracy: {:.2f}%".format(log_acc * 100))

print("\nClassification Report\n")

print(classification_report(y_test, log_pred))

joblib.dump(log_model, "model/pcos_model.pkl")

# ============================================
# Decision Tree
# ============================================

print("\n==============================")
print("Training Decision Tree")
print("==============================")

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)

print("Accuracy: {:.2f}%".format(dt_acc * 100))

# ============================================
# Naive Bayes
# ============================================

print("\n==============================")
print("Training Naive Bayes")
print("==============================")

nb_model = GaussianNB()

nb_model.fit(X_train_scaled, y_train)

nb_pred = nb_model.predict(X_test_scaled)

nb_acc = accuracy_score(y_test, nb_pred)

print("Accuracy: {:.2f}%".format(nb_acc * 100))

# ============================================
# Accuracy Comparison
# ============================================

print("\n======================================")
print("MODEL COMPARISON")
print("======================================")

print(f"Logistic Regression : {log_acc*100:.2f}%")
print(f"Decision Tree       : {dt_acc*100:.2f}%")
print(f"Naive Bayes         : {nb_acc*100:.2f}%")

print("\nModel Files Saved Successfully!")

print("\nSaved Files:")

print("✔ model/pcos_model.pkl")
print("✔ model/scaler.pkl")
print("✔ model/feature_names.pkl")

print("\nPCOSence Training Completed Successfully!")