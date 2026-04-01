import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="AI Growth Engine", layout="wide")

st.title("🚀 AI Customer Growth Engine")
st.write("Upload your CSV to predict churn + business insights")

# Upload file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # -------------------------------
    # 🧠 HANDLE FEATURE MISMATCH
    # -------------------------------

    expected_features = model.feature_names_in_

    # Add missing columns
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    # Keep only required columns in correct order
    X = df[expected_features]

    # -------------------------------
    # 💰 ADD BUSINESS FEATURES
    # -------------------------------

    if 'MonthlyCharges' not in df.columns:
        df['MonthlyCharges'] = np.random.randint(1000, 5000, len(df))

    if 'tenure' not in df.columns:
        df['tenure'] = np.random.randint(1, 72, len(df))

    df['CLV'] = df['MonthlyCharges'] * df['tenure']

    # -------------------------------
    # 🤖 PREDICTIONS
    # -------------------------------

    df['Churn'] = model.predict(X)
    df['Churn Probability'] = model.predict_proba(X)[:, 1]

    # -------------------------------
    # 🎯 SMART SEGMENTATION
    # -------------------------------

    def segment(row):
        if row['Churn Probability'] > 0.7 and row['CLV'] > df['CLV'].median():
            return "🔥 High Risk - High Value"
        elif row['Churn Probability'] > 0.7:
            return "⚠️ High Risk"
        else:
            return "✅ Safe"

    df['Segment'] = df.apply(segment, axis=1)

    # -------------------------------
    # 📈 SHOW RESULTS
    # -------------------------------

    st.subheader("📈 Predictions")
    st.dataframe(df)

    # -------------------------------
    # 💼 BUSINESS INSIGHTS
    # -------------------------------

    high_risk = df[df['Churn Probability'] > 0.7]
    high_value_risk = df[
        (df['Churn Probability'] > 0.7) &
        (df['CLV'] > df['CLV'].median())
    ]

    st.subheader("💼 Business Insights")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df))
    col2.metric("High Risk Customers", len(high_risk))
    col3.metric("🔥 High Value at Risk", len(high_value_risk))

    # Chart
    st.subheader("📊 Churn Probability Distribution")
    st.bar_chart(df['Churn Probability'])

    st.success("✅ Analysis Complete!")