import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ================================
# LOAD MODEL
# ================================
try:
    model = joblib.load("linear_regression_model.joblib")
    encoders = joblib.load("encoders (1).joblib")
except Exception as e:
    st.error(f"Gagal memuat model atau encoder: {e}")
    st.stop()

# ================================
# CONFIG
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
# SIDEBAR INPUT
# ================================
st.sidebar.header("📥 Masukkan Data Siswa")

def safe_select(label, options, key):
    return st.sidebar.selectbox(label, options, key=key)

gender = safe_select("Gender", ["male", "female"], key="gender")
race = safe_select("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"], key="race")
lunch = safe_select("Lunch", ["standard", "free/reduced"], key="lunch")
prep = safe_select("Test Preparation", ["none", "completed"], key="prep")

math = st.sidebar.number_input("Math Score", 0, 100, 50, step=1)
reading = st.sidebar.number_input("Reading Score", 0, 100, 50, step=1)

# ================================
# TAMPILKAN INPUT
# ================================
st.subheader("📝 Data Input Siswa")

input_data_display = pd.DataFrame({
    "Gender": [gender],
    "Race/Ethnicity": [race],
    "Lunch": [lunch],
    "Test Preparation": [prep],
    "Math Score": [math],
    "Reading Score": [reading]
})
st.dataframe(input_data_display.T, use_container_width=True)

# Dataframe untuk prediksi
input_df = pd.DataFrame({
    "gender": [gender],
    "race/ethnicity": [race],
    "lunch": [lunch],
    "test preparation course": [prep],
    "math score": [math],
    "reading score": [reading]
})

# ================================
# PREDIKSI BUTTON
# ================================
if st.sidebar.button("🔮 Prediksi Score"):

    with st.spinner("Model sedang memproses..."):

        # Copy input
        input_df_encoded = input_df.copy()

        # ================================
        # ENCODE INPUT
        # ================================
        for col in encoders:
            if col in input_df_encoded.columns:
                try:
                    input_df_encoded[col] = encoders[col].transform(input_df_encoded[col])
                except:
                    st.warning(f"Nilai baru pada '{col}' tidak dikenali encoder → otomatis di-set 0.")
                    input_df_encoded[col] = 0

        # ================================
        # SESUAIKAN KOLOM DENGAN MODEL
        # ================================
        input_df_final = input_df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

        # ================================
        # PREDIKSI
        # ================================
        prediction = float(model.predict(input_df_final)[0])

    st.success(f"🎯 **Prediksi Final Score: {prediction:.2f}**")

    st.markdown("---")

    # ===================================
    # TABS
    # ===================================
    tab1, tab2, tab3 = st.tabs(["🚀 Prediksi Score", "📈 Distribusi Model", "🔍 Math vs Reading"])

    # ---------------------------------------------------
    # TAB 1 — BAR CHART
    # ---------------------------------------------------
    with tab1:
        st.subheader("🚀 Predicted Final Score")

        fig_bar = go.Figure([
            go.Bar(x=["Predicted Score"], y=[prediction], marker_color="#1f77b4")
        ])
        fig_bar.update_layout(
            title="Prediksi Nilai Akhir",
            yaxis_title="Score",
            yaxis_range=[0, 100],
            height=400
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------------------------------------------
    # TAB 2 — DISTRIBUSI MODEL
    # ---------------------------------------------------
    with tab2:
        st.subheader("📈 Distribusi Prediksi Model")

        sample_size = 500
        sample_df = pd.DataFrame()

        for col in encoders:
            if col in ["math score", "reading score"]:
                sample_df[col] = np.random.randint(0, 100, sample_size)
            else:
                sample_df[col] = np.random.choice(encoders[col].classes_, sample_size)

        # Encode sample
        for col in encoders:
            try:
                sample_df[col] = encoders[col].transform(sample_df[col])
            except:
                sample_df[col] = 0

        sample_df_final = sample_df.reindex(columns=model.feature_names_in_, fill_value=0)
        preds = model.predict(sample_df_final)

        fig_hist = px.histogram(
            preds, nbins=25, title="Distribusi Prediksi Nilai", height=450
        )

        fig_hist.add_vline(
            x=prediction,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Prediksi Anda: {prediction:.2f}"
        )

        fig_hist.update_layout(xaxis_range=[0, 100])
        st.plotly_chart(fig_hist, use_container_width=True)

    # ---------------------------------------------------
    # TAB 3 — SCATTER MATH VS READING
    # ---------------------------------------------------
    with tab3:
        st.subheader("🔍 Math vs Reading Input Anda")

        fig_scatter = go.Figure([
            go.Scatter(x=[math], y=[reading], mode="markers", marker=dict(size=12, color="red"))
        ])
        fig_scatter.update_layout(
            title="Scatter Plot Math vs Reading",
            xaxis_title="Math Score",
            yaxis_title="Reading Score",
            xaxis_range=[0, 100],
            yaxis_range=[0, 100]
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
