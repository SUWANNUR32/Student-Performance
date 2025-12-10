import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_array

st.set_page_config(page_title="Prediksi Student Performance", layout="wide", page_icon="🎓")

st.title("🎓 Student Performance Prediction App")
st.write("Aplikasi ini memuat model & encoder dari file .joblib dan memprediksi nilai akhir siswa.")

# -----------------------
# Tampilkan isi folder (debug)
# -----------------------
st.sidebar.header("🔎 Debug: File di folder")
try:
    files = os.listdir(".")
except Exception as e:
    files = []
st.sidebar.write(files)

# -----------------------
# LOAD MODEL & ENCODER (ganti nama file jika diperlukan)
# -----------------------
MODEL_FILE = "linear_regression_model.joblib"
ENC_FILE = "encoders (1).joblib"

model = None
encoders = None

try:
    model = joblib.load(MODEL_FILE)
    st.sidebar.success(f"✅ Model loaded: {MODEL_FILE}")
except Exception as e:
    st.sidebar.error(f"❌ Gagal load model '{MODEL_FILE}': {e}")

try:
    encoders = joblib.load(ENC_FILE)
    st.sidebar.success(f"✅ Encoders loaded: {ENC_FILE}")
except Exception as e:
    st.sidebar.error(f"❌ Gagal load encoders '{ENC_FILE}': {e}")

# Kalau model belum ter-load, stop
if model is None:
    st.error("Model belum berhasil dimuat — periksa nama file model dan coba lagi.")
    st.stop()

# Ambil list feature yang diharapkan model (sklearn >= 1.0 biasanya punya attribute ini)
try:
    model_features = list(model.feature_names_in_)
except Exception:
    # fallback: kalau model tidak punya feature_names_in_
    model_features = None

st.sidebar.header("ℹ️ Info model")
st.sidebar.write("Feature names expected by model:")
st.sidebar.write(model_features if model_features is not None else "model.feature_names_in_ not available")

# -----------------------
# INPUT SIDEBAR (user)
# -----------------------
st.sidebar.header("📥 Masukkan Data Siswa")

def safe_select(label, options):
    try:
        return st.sidebar.selectbox(label, options)
    except Exception:
        return st.sidebar.text_input(label)

# -- contoh input sesuai Student Performance dataset umum --
gender = safe_select("Gender", ["male", "female"])
race = safe_select("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
lunch = safe_select("Lunch", ["standard", "free/reduced"])
prep = safe_select("Test Preparation", ["none", "completed"])

math = st.sidebar.number_input("Math Score", min_value=0, max_value=100, value=50)
reading = st.sidebar.number_input("Reading Score", min_value=0, max_value=100, value=50)
# tambahkan writing input jika model mengharapkan
writing = st.sidebar.number_input("Writing Score", min_value=0, max_value=100, value=50)

# Buat raw dataframe
raw_df = pd.DataFrame({
    "gender": [gender],
    "race/ethnicity": [race],
    "lunch": [lunch],
    "test preparation course": [prep],
    "math score": [math],
    "reading score": [reading],
    "writing score": [writing]
})

st.subheader("📌 Data Input (mentah)")
st.write(raw_df)

# -----------------------
# APPLY ENCODERS (robust)
# encoders expected: dict {col_name: fitted_encoder}
# handles LabelEncoder and OneHotEncoder style (with get_feature_names_out)
# -----------------------
input_df = raw_df.copy()

if encoders is None:
    st.warning("Encoders tidak ditemukan — akan mencoba prediksi dengan nilai mentah (bisa gagal jika model butuh encoding).")
else:
    # encoders may be dict-like
    if isinstance(encoders, dict):
        for col_name, enc in encoders.items():
            # if the encoder's column exists in our input, transform it
            if col_name in input_df.columns:
                try:
                    # try transform with 2D input (OneHotEncoder expects 2D)
                    transformed = enc.transform(input_df[[col_name]])
                except Exception:
                    # fallback: transform 1D
                    transformed = enc.transform(input_df[col_name])
                # convert sparse to dense if necessary
                if hasattr(transformed, "toarray"):
                    transformed = transformed.toarray()
                # get output feature names if available
                try:
                    if hasattr(enc, "get_feature_names_out"):
                        out_names = list(enc.get_feature_names_out([col_name]))
                    else:
                        # if transform returned 1d, keep original col name
                        if np.ndim(transformed) == 1 or (np.ndim(transformed) == 2 and transformed.shape[1] == 1):
                            out_names = [col_name]
                        else:
                            # create generic names
                            out_names = [f"{col_name}_{i}" for i in range(transformed.shape[1])]
                except Exception:
                    out_names = [col_name] if np.ndim(transformed) <= 1 else [f"{col_name}_{i}" for i in range(transformed.shape[1])]

                # write transformed columns into input_df
                if np.ndim(transformed) == 1:
                    input_df[out_names[0]] = transformed
                    if out_names[0] != col_name:
                        # if original still exists, drop it to avoid duplication
                        if col_name in input_df.columns:
                            input_df.drop(columns=[col_name], inplace=True, errors='ignore')
                else:
                    for i, name in enumerate(out_names):
                        input_df[name] = transformed[:, i]
                    # drop original col
                    input_df.drop(columns=[col_name], inplace=True, errors='ignore')
    else:
        st.warning("Encoders bukan dict — tidak melakukan transformasi otomatis. Pastikan encoders berupa dict {col: encoder}.")

st.subheader("🔧 Data setelah encoding (siap untuk disesuaikan dengan model)")
st.write(input_df)

# -----------------------
# Align columns with model expectations
# -----------------------
if model_features is not None:
    # add missing columns as 0, reorder
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    # keep only those columns and order them
    input_df = input_df[model_features]
    st.sidebar.write("Input dataframe columns (after align):")
    st.sidebar.write(list(input_df.columns))
else:
    st.sidebar.write("Tidak bisa menyesuaikan kolom karena model tidak menyediakan feature_names_in_.")
    # keep input_df as-is (may fail on predict)
    st.sidebar.write(list(input_df.columns))

# -----------------------
# Predict button
# -----------------------
if st.sidebar.button("🔮 Prediksi Score"):
    try:
        # show shapes/columns for debug
        st.write("🔎 Kolom yang dikirim ke model:")
        st.write(list(input_df.columns))
        if hasattr(model, "feature_names_in_"):
            st.write("🔎 Feature names in model:")
            st.write(list(model.feature_names_in_))

        # convert to numeric array if needed
        X = input_df.values.astype(float)

        pred = model.predict(X)[0]
        st.success(f"🎯 Prediksi Final Score: {pred:.2f}")

        # plot
        fig, ax = plt.subplots()
        ax.bar(["Predicted Score"], [pred])
        ax.set_ylabel("Score")
        ax.set_ylim(0, 100)
        ax.set_title("Gr_
