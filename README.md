# ⚙️ AI Predictive Maintenance System

🚀 An industry-level machine learning project that predicts equipment failures using sensor data and provides real-time insights through an interactive dashboard.

---

## 📌 Overview

This project simulates a real-world **Predictive Maintenance System** used in industries like manufacturing, aviation, and energy.

It uses historical sensor data to:
- Predict machine failures
- Analyze risk levels
- Detect anomalies
- Provide actionable insights

---

## 🎯 Problem Statement

Traditional maintenance approaches:
- ❌ Reactive (after failure)
- ❌ Preventive (fixed schedule)

👉 This project uses **AI-based predictive maintenance** to:
- Reduce downtime
- Save costs
- Improve efficiency

---

## 🧠 Features

### 🔹 Core Features
- Machine failure prediction (ML model)
- Health score calculation
- Risk level classification (Low / Medium / High)

### 🔹 Advanced Features
- Anomaly detection (Z-score based)
- Feature importance analysis
- Sensor-based predictions (21 sensors)

### 🔹 Dashboard (Streamlit)
- Interactive UI
- Manual + auto sensor input
- Real-time prediction results

### 🔹 File Processing
- Upload `.csv` and `.txt` datasets
- NASA dataset compatibility
- Batch prediction support

### 🔹 Analytics
- Prediction distribution
- Probability histogram
- Health trend analysis
- High-risk case identification

---

## 🏗️ Project Structure
AI-Predictive-Maintenance-System/
│
├── data/
│ └── train_FD001.txt
│
├── models/
│ └── model.pkl
│
├── src/
│ ├── data_preprocessing.py
│ ├── model.py
│ ├── evaluate.py
│ └── visualize.py
│
├── outputs/
├── images/
│
├── app.py
├── main.py
├── train_model.py
├── requirements.txt
└── README.md


---

## ⚙️ Tech Stack

- Python 🐍
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit

---

## 📊 Dataset

- NASA Turbofan Engine Dataset (C-MAPSS)
- Contains:
  - Sensor readings
  - Time-based degradation
  - Machine lifecycle data

---

## 🔄 Workflow
Dataset → Preprocessing → Feature Engineering → Model Training → Prediction → Dashboard


---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
