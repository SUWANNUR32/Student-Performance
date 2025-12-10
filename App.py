import streamlit as st
import joblib
import os

st.write("📂 File dalam folder:", os.listdir())

# Load model dan encoder sesuai nama file
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

# Contoh input — GANTI sesuai dataset Anda
gender = st.sidebar.selectbox("Gender", ["male", "female"])
race = st.sidebar.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
lunch = st.sidebar.selectbox("Lunch", ["standard", "free/reduced"])
prep = st.sidebar.selectbox("Test Preparation", ["none", "completed"])
math = st.sidebar.number_input("Math Score", 0, 100, 50)
reading = st.sidebar.number_input("Reading Score", 0, 100, 50)

# Bentuk dataframe input
input_df = pd.DataFrame({
    "gender": [gender],
    "race/ethnicity": [race],
    "lunch": [lunch],
    "test preparation course": [prep],
    "math score": [math],
    "reading score": [reading]
})

st.subheader("📌 Data Input")
st.write(input_df)

# ================================
# APPLY ENCODER
# ================================
for col in encoders:
    if col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])

# ================================
# PREDICT
# ================================
if st.sidebar.button("🔮 Prediksi Score"):
    prediction = model.predict(input_df)[0]

    st.success(f"🎯 **Prediksi Final Score: {prediction:.2f}**")

    # Grafik hasil prediksi
    fig, ax = plt.subplots()
    ax.bar(["Predicted Score"], [prediction])
    ax.set_ylabel("Score")
    ax.set_title("Grafik Prediksi Nilai")
    st.pyplot(fig)

# ================================
# ADDITIONAL CHART
# ================================
st.subheader("📊 Grafik Hubungan Math vs Reading (Input Anda)")
fig2, ax2 = plt.subplots()
ax2.scatter([math], [reading])
ax2.set_xlabel("Math Score")
ax2.set_ylabel("Reading Score")
ax2.set_title("Scatter Plot Math vs Reading")
st.pyplot(fig2)
