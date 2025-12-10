import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
# Import Plotly untuk visualisasi interaktif
import plotly.express as px
import plotly.graph_objects as go

# ================================
# CEK FILE (dapat dihapus di aplikasi final)
# ================================
# st.write("📂 File dalam folder:", os.listdir()) 

# Load model & encoder
try:
    model = joblib.load("linear_regression_model.joblib")
    encoders = joblib.load("encoders (1).joblib")
except FileNotFoundError:
    st.error("Model atau encoder tidak ditemukan. Pastikan file 'linear_regression_model.joblib' dan 'encoders (1).joblib' ada di direktori yang sama.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat file: {e}")
    st.stop()


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

def safe_select(label, options, key):
    """Fungsi untuk selectbox dengan key unik."""
    try:
        return st.sidebar.selectbox(label, options, key=key)
    except:
        return st.sidebar.text_input(label, key=key)

gender = safe_select("Gender", ["male", "female"], key="gender_select")
race = safe_select("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"], key="race_select")
lunch = safe_select("Lunch", ["standard", "free/reduced"], key="lunch_select")
prep = safe_select("Test Preparation", ["none", "completed"], key="prep_select")

# Mengatur step=1 agar input lebih nyaman
math = st.sidebar.number_input("Math Score", min_value=0, max_value=100, value=50, step=1)
reading = st.sidebar.number_input("Reading Score", min_value=0, max_value=100, value=50, step=1)

# ================================
# TAMPILAN DATA INPUT
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
st.dataframe(input_data_display.T, use_container_width=True) # Transpose agar lebih rapi

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
# PREDIKSI
# ================================
if st.sidebar.button("🔮 Prediksi Score"):
    with st.spinner('Model sedang memprediksi...'):
        
        # Buat salinan data untuk proses encoding
        input_df_encoded = input_df.copy()

        # ================================
        # APPLY ENCODER KE INPUT
        # ================================
        for col in encoders:
            if col in input_df_encoded.columns:
                try:
                    # Menggunakan .transform() pada array 1D
                    input_df_encoded[col] = encoders[col].transform(input_df_encoded[col].to_numpy().reshape(-1, 1))
                except ValueError as e:
                    # Tangani jika data baru tidak dikenal oleh encoder
                    st.warning(f"Nilai pada kolom '{col}' mungkin tidak ada dalam data training. Encoding gagal.")
                    # Jika encoding gagal, set nilai ke 0 atau gunakan nilai aman lainnya
                    input_df_encoded[col] = 0
                except Exception as e:
                    st.error(f"Kesalahan saat encoding kolom {col}: {e}")
                    input_df_encoded[col] = 0


        # ================================
        # COCOKKAN KOLOM DENGAN MODEL
        # ================================
        # Memastikan kolom sesuai urutan dan nama fitur model, isi yang hilang dengan 0
        try:
            input_df_final = input_df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
        except AttributeError:
             st.error("Model tidak memiliki atribut 'feature_names_in_'. Pastikan model dilatih dengan scikit-learn terbaru.")
             st.stop()

        # ================================
        # PREDIKSI
        # ================================
        prediction = model.predict(input_df_final)
        prediction_value = float(prediction[0])

    st.success(f"🎯 **Prediksi Final Score: {prediction_value:.2f}**")
    
    st.markdown("---")

    # Menggunakan tabs untuk memisahkan visualisasi
    tab1, tab2, tab3 = st.tabs(["🚀 Prediksi Score", "📈 Distribusi Model", "🔍 Math vs Reading"])

    with tab1:
        # ================================
        # 1) Grafik Prediksi (Bar Plotly)
        # ================================
        st.subheader("🚀 Predicted Final Score")
        
        fig_bar = go.Figure(data=[
            go.Bar(x=["Predicted Score"], y=[prediction_value], marker_color='#1f77b4')
        ])
        fig_bar.update_layout(
            title_text='Prediksi Nilai Akhir Siswa', 
            yaxis_title='Score',
            yaxis_range=[0, 100], # Memastikan skala 0-100
            height=400
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        # ================================
        # 2) Distribusi Prediksi (Histogram Plotly)
        # ================================
        st.subheader("📈 Distribusi Nilai Prediksi Model")

        # Buat sample_df dinamis sesuai encoder
        sample_size = 500 # Tingkatkan sampel untuk distribusi yang lebih halus
        sample_df_raw = pd.DataFrame()

        for col in encoders.keys():
            classes = encoders[col].classes_

            if col in ["math score", "reading score"]:
                sample_df_raw[col] = np.random.randint(0, 100, sample_size)
            else:
                sample_df_raw[col] = np.random.choice(classes, sample_size)
        
        # Buat salinan untuk proses encoding
        sample_df_encoded = sample_df_raw.copy()

        # encode kolom kategorikal
        for col in encoders:
            if col in sample_df_encoded.columns:
                try:
                    # Menggunakan .transform()
                    sample_df_encoded[col] = encoders[col].transform(sample_df_encoded[col].to_numpy().reshape(-1, 1))
                except Exception:
                    sample_df_encoded[col] = 0 # Fallback jika ada error

        # Reindex agar kolom sama dengan model
        sample_df_final = sample_df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

        # Prediksi distribusi
        preds = model.predict(sample_df_final)

        # Plot histogram Plotly
        preds_df = pd.DataFrame({'Predicted Score': preds})
        fig_hist = px.histogram(
            preds_df, 
            x="Predicted Score", 
            nbins=25, 
            title="Distribusi Prediksi Nilai Akhir",
            height=450
        )
        
        # Tambahkan garis prediksi Anda
        fig_hist.add_vline(
            x=prediction_value, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"Prediksi Anda: {prediction_value:.2f}",
            annotation_position="top right"
        )
        fig_hist.update_layout(xaxis_range=[0, 100])
        
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        # ================================
        # 3) SCATTER PLOT (Plotly)
        # ================================
        st.subheader("🔍 Hubungan Nilai Math vs Reading")
        
        # Scatter Plot Plotly
        fig_scatter = go.Figure(data=[
            go.Scatter(
                x=[math], 
                y=[reading], 
                mode='markers', 
                marker=dict(size=12, color='red'),
                name='Input Anda'
            )
        ])

        fig_scatter.update_layout(
            title_text='Scatter Plot Math Score vs Reading Score (Input Anda)',
            xaxis_title='Math Score',
            yaxis_title='Reading Score',
            xaxis_range=[0, 100],
            yaxis_range=[0, 100],
            height=450
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
