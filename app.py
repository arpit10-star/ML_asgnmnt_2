import streamlit as st
import pandas as pd
import seaborn as sns
import joblib
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


st.set_page_config(page_title="ML Model Deployment", layout="wide")

st.title("📊 Heart Disease - Multi Model Prediction Platform")


# ==============================
# Load Models
# ==============================
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("lr_pipeline3.pkl"),
        "Random Forest": joblib.load("RF_pipeline3.pkl"),
        "XGBoost": joblib.load("XGB_pipeline3.pkl"),
        "Decision Tree": joblib.load("tree_pipeline3.pkl"),
        "KNN": joblib.load("knn_pipeline3.pkl"),
        "Gaussian Naive Bayes": joblib.load("GNB_pipeline3.pkl")
    }
    return models


models = load_models()


# ==============================
# Model Selection
# ==============================
selected_model_name = st.selectbox("Select a model", list(models.keys()))
model = models[selected_model_name]

st.success(f"Selected Model: {selected_model_name}")


# ==============================
# File Upload
# ==============================
uploaded_file = st.file_uploader("Upload CSV for prediction", type=["csv"])


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ==============================
    # Split features & target
    # ==============================
    TARGET_COL = "HeartDisease"

    if TARGET_COL in df.columns:
        X = df.drop(columns=[TARGET_COL])
        y_true = df[TARGET_COL].map({"Yes":1, "No":0})
    else:
        X = df.copy()
        y_true = None

    # ======================

    # ==============================
    # Prediction
    # ==============================
    if st.button("Run Prediction"):
        y_pred = model.predict(X)



        df["Prediction"] = y_pred

        st.subheader("Predictions")
        st.dataframe(df.head())

        # ==============================
        # Metrics
        # ==============================
        if y_true is not None:
            st.subheader("Evaluation Metrics")

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average="weighted")
            rec = recall_score(y_true, y_pred, average="weighted")
            f1 = f1_score(y_true, y_pred, average="weighted")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{acc:.3f}")
            col2.metric("Precision", f"{prec:.3f}")
            col3.metric("Recall", f"{rec:.3f}")
            col4.metric("F1 Score", f"{f1:.3f}")

            # ==============================
            # Confusion Matrix
            # ==============================
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots(figsize=(4, 4))
            sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            st.pyplot(fig,use_container_width=False)

        else:
            st.warning("Target column not found. Metrics cannot be computed.")
