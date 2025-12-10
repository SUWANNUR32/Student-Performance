import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================================
# CEK FILE
# ================================
st.write("📂 File dalam folder:", os.listdir())

# Load model & encoder
model = joblib.load("linear_regression_model.joblib")
encoders = joblib.load("encoders (1).joblib")

# ================================
# STREAMLIT PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Prediksi Student Performance",
    layout="wide",
    page_icon="🎓"
)

# ================================
# HEADER
# ================================
st.title("🎓 Student Performance Prediction App")
st.markdown("""
Aplikasi ini menggunakan model **Linear Regression** untuk memprediksi nilai akhir siswa.
Silakan masukkan data pada panel sebelah kiri.
""")

# ================================
# INPUT SIDEBAR
# ================================
st.sidebar.header("📥 Masukkan Data Siswa")

def safe_select(label, options):
    try:
        return st.sidebar.selectbox(label, options)
    except:
        return st.sidebar.text_input(label)

gender = safe_select("Gender", ["male", "female"])
race = safe_select("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
lunch = safe_select("Lunch", ["standard", "free/reduced"])
prep = safe_select("Test Preparation", ["none", "completed"])

math = st.sidebar.number_input("Math Score", min_value=0, max_value=100, value=50)
reading = st.sidebar.number_input("Reading Score", min_value=0, max_value=100, value=50)

# ================================
# INPUT DATAFRAME
# ================================
input_df = pd.DataFrame({
    "gender": [gender],
    "race/ethnicity": [race],
    "lunch": [lunch],
    "test preparation course": [prep],
    "math score": [math],
    "reading score": [reading]
})

# ================================
# APPLY ENCODER KE INPUT
# ================================
for col in encoders:
    if col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])

# ================================
# COCOKKAN KOLOM DENGAN MODEL
# ================================
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# ================================
# PREDIKSI
# ================================
st.subheader("🔍 Hasil Prediksi")

if st.sidebar.button("🔮 Prediksi Score"):

    prediction = model.predict(input_df)
    prediction_value = float(prediction[0])

    st.success(f"🎯 **Prediksi Final Score: {prediction_value:.2f}**")

    # ================================
    # 1) Grafik Prediksi (Bar)
    # ================================
    fig, ax = plt.subplots()
    ax.bar(["Predicted Score"], [prediction_value])
    ax.set_ylabel("Score")
    ax.set_title("Grafik Prediksi Nilai")
    st.pyplot(fig)

    # ================================
    # 2) Distribusi Prediksi (Histogram)
    # ================================
    st.subheader("📈 Distribusi Nilai Prediksi Model")

    # Buat sample_df dinamis sesuai encoder (ANTI ERROR)
    sample_df = pd.DataFrame()

    for col in encoders.keys():
        classes = encoders[col].classes_

        if col in ["math score", "reading score"]:
            sample_df[col] = np.random.randint(0, 100, 200)
        else:
            sample_df[col] = np.random.choice(classes, 200)

        # encode kolom kategorikal
        if col in encoders:
            sample_df[col] = encoders[col].transform(sample_df[col])

    # Reindex agar kolom sama dengan model
    sample_df = sample_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Prediksi distribusi
    preds = model.predict(sample_df)

    # Plot histogram
    fig3, ax3 = plt.subplots()
    ax3.hist(preds, bins=20, alpha=0.6)
    ax3.axvline(prediction_value, color='red', linestyle='--', linewidth=2,
                label=f"Prediksi Anda: {prediction_value:.2f}")
    ax3.set_title("Distribusi Prediksi Model")
    ax3.set_xlabel("Score")
    ax3.set_ylabel("Frekuensi")
    ax3.legend()

    st.pyplot(fig3)

# ================================
# SCATTER PLOT
# ================================
st.subheader("📊 Grafik Hubungan Math vs Reading (Input Anda)")

fig2, ax2 = plt.subplots()
ax2.scatter([math], [reading])
ax2.set_xlabel("Math Score")
ax2.set_ylabel("Reading Score")
ax2.set_title("Scatter Plot Math vs Reading")
st.pyplot(fig2)
