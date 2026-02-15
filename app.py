import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# -----------------------------
# Load and preprocess dataset
# -----------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
columns = ["ID", "Diagnosis"] + [f"feature_{i}" for i in range(1, 31)]
df = pd.read_csv(url, header=None, names=columns)

# Convert target to numeric (M=1, B=0)
df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})
df = df.drop("ID", axis=1)

X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

# Scale features for better convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# Define models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, solver="lbfgs"),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Breast Cancer Classification App")

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# File uploader for test data
uploaded_file = st.file_uploader("Upload Test CSV (optional)", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    # Ensure column names match training schema
    expected_cols = [f"feature_{i}" for i in range(1, 31)] + ["Diagnosis"]
    test_df.columns = test_df.columns.str.strip()

    if not set(expected_cols).issubset(set(test_df.columns)):
        st.error("Uploaded file columns do not match training schema.")
        st.stop()

    # Handle Diagnosis column safely
    if test_df['Diagnosis'].dtype == object:
        test_df['Diagnosis'] = (
            test_df['Diagnosis']
            .astype(str).str.strip().str.upper()
            .map({'M': 1, 'B': 0})
        )
    else:
        test_df['Diagnosis'] = test_df['Diagnosis'].astype(int)

    # Drop rows with missing target values
    test_df = test_df.dropna(subset=['Diagnosis'])

    # Apply the same scaler used during training
    test_X = scaler.transform(test_df.drop("Diagnosis", axis=1))
    test_y = test_df["Diagnosis"].astype(int)
else:
    test_X, test_y = X_test, y_test

# -----------------------------
# Provide full RAW test CSV download
# -----------------------------
full_test_raw = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(1, 31)])
full_test_raw["Diagnosis"] = y_test.values

csv_bytes = full_test_raw.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Test CSV",
    data=csv_bytes,
    file_name="full_test_raw.csv",
    mime="text/csv"
)

# -----------------------------
# Model selection and execution
# -----------------------------
model_choice = st.selectbox("Choose Model", list(models.keys()))

if st.button("Run Model"):
    model = models[model_choice]
    model.fit(X_train, y_train)
    y_pred = model.predict(test_X)

    # Save trained model and scaler
    joblib.dump(model, f"model/{model_choice.replace(' ', '_')}.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    # Display metrics
    st.write("Accuracy:", accuracy_score(test_y, y_pred))
    st.write("Precision:", precision_score(test_y, y_pred))
    st.write("Recall:", recall_score(test_y, y_pred))
    st.write("F1 Score:", f1_score(test_y, y_pred))
    st.write("MCC:", matthews_corrcoef(test_y, y_pred))

    # AUC Score (binary classification)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(test_X)[:, 1]
        st.write("AUC:", roc_auc_score(test_y, y_prob))

    # Confusion Matrix
    cm = confusion_matrix(test_y, y_pred)
    st.write("Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)
