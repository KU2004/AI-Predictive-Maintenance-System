import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Predictive Maintenance", layout="wide")

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/model.pkl")

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- HEADER ----------------
st.title("⚙️ AI Predictive Maintenance System")
st.markdown("### 🚀 Smart Machine Failure Detection Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.radio("Input Mode", ["Manual", "Auto Simulation"])
threshold = st.sidebar.slider("Failure Threshold", 0.1, 0.9, 0.5)
normalize = st.sidebar.checkbox("Normalize Data")

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📂 File Prediction", "📈 Analytics"])

# =========================================================
# ---------------- DASHBOARD -------------------------------
# =========================================================
with tab1:

    st.subheader("📡 Sensor Input Panel")

    sensor_values = []
    cols = st.columns(3)

    for i in range(1, 22):
        with cols[(i-1) % 3]:
            if mode == "Auto Simulation":
                val = np.random.uniform(0, 100)
                st.write(f"s{i}: {val:.2f}")
            else:
                val = st.slider(f"Sensor s{i}", 0.0, 100.0, 50.0)
            sensor_values.append(val)

    if normalize:
        sensor_values = (np.array(sensor_values) - np.mean(sensor_values)) / np.std(sensor_values)

    st.markdown("---")

    if st.button("🔮 Predict Machine Health"):

        data = np.array(sensor_values).reshape(1, -1)
        prob = model.predict_proba(data)[0][1]
        prediction = 1 if prob > threshold else 0

        health = int((1 - prob) * 100)

        if prob > 0.7:
            risk = "HIGH"
        elif prob > 0.4:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        st.subheader("📊 Prediction Result")

        if prediction:
            st.error(f"⚠️ HIGH FAILURE RISK ({prob*100:.2f}%)")
        else:
            st.success("✅ MACHINE HEALTHY")

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Health Score", f"{health}")
        col2.metric("Failure Probability", f"{prob*100:.2f}%")
        col3.metric("Risk Level", risk)

        st.progress(health)

        # ---------------- ANOMALY ----------------
        st.markdown("### 🚨 Anomaly Detection")

        z_scores = np.abs(zscore(sensor_values))
        anomalies = np.where(z_scores > 2)[0]

        if len(anomalies):
            st.warning(f"Anomalies detected in sensors: {anomalies+1}")
        else:
            st.success("No anomalies detected")

        # ---------------- FEATURE IMPORTANCE ----------------
        st.markdown("### 🔍 Top Influential Sensors")

        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:10]

        fig, ax = plt.subplots()
        ax.bar([f"s{i+1}" for i in sorted_idx], importances[sorted_idx])
        ax.set_title("Top 10 Important Sensors")
        st.pyplot(fig)

        # Save history
        st.session_state.history.append({
            "Health": health,
            "Probability": prob,
            "Prediction": prediction
        })

# =========================================================
# ---------------- FILE PREDICTION -------------------------
# =========================================================
with tab2:

    st.subheader("📂 Upload Dataset (.csv or .txt)")

    file = st.file_uploader("Upload file", type=["csv", "txt"])

    if file:

        try:
            # ---------------- LOAD ----------------
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)

            else:
                df = pd.read_csv(file, sep=" ", header=None)
                df = df.dropna(axis=1)

                cols = ["unit","time"] + \
                       [f"op{i}" for i in range(1,4)] + \
                       [f"s{i}" for i in range(1,22)]

                df.columns = cols
                df = df[[f"s{i}" for i in range(1,22)]]

            st.markdown("### 📊 Data Preview")
            st.dataframe(df.head())

            # ---------------- VALIDATION ----------------
            if df.shape[1] != 21:
                st.error("Dataset must contain 21 sensor columns")
            else:

                # ---------------- PREDICTION ----------------
                preds = model.predict(df)
                probs = model.predict_proba(df)[:,1]

                df["Prediction"] = preds
                df["Probability"] = probs

                total = len(df)
                failures = preds.sum()

                col1, col2 = st.columns(2)
                col1.metric("Total Records", total)
                col2.metric("Predicted Failures", failures)

                st.markdown("### 📋 Results")
                st.dataframe(df.head())

                # Download
                st.download_button(
                    "📥 Download Results",
                    df.to_csv(index=False),
                    "predictions.csv"
                )

                # Save for analytics
                st.session_state.history.extend([
                    {"Health": int((1-p)*100), "Probability": p, "Prediction": pr}
                    for p, pr in zip(probs, preds)
                ])

        except Exception as e:
            st.error(f"Error processing file: {e}")

# =========================================================
# ---------------- ANALYTICS -------------------------------
# =========================================================
with tab3:

    st.subheader("📈 Advanced Analytics")

    if len(st.session_state.history) > 0:

        hist = pd.DataFrame(st.session_state.history)

        # ---------------- DISTRIBUTION ----------------
        st.markdown("### 📊 Prediction Distribution")
        st.bar_chart(hist["Prediction"].value_counts())

        # ---------------- PROBABILITY HIST ----------------
        st.markdown("### 📊 Failure Probability Histogram")

        fig, ax = plt.subplots()
        ax.hist(hist["Probability"], bins=20)
        ax.set_title("Probability Distribution")
        st.pyplot(fig)

        # ---------------- HEALTH TREND ----------------
        st.markdown("### 📈 Health Score Trend")

        fig2, ax2 = plt.subplots()
        ax2.plot(hist["Health"])
        ax2.set_title("Health Over Time")
        st.pyplot(fig2)

        # ---------------- HIGH RISK CASES ----------------
        st.markdown("### ⚠️ High Risk Records")

        high_risk = hist[hist["Probability"] > 0.7]
        st.dataframe(high_risk.head())

    else:
        st.info("Run predictions to see analytics")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 AI Predictive Maintenance | Final Industry Version")