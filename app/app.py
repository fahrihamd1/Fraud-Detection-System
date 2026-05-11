import os
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import pickle
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ================================
# LOAD MODEL, SCALER, DAN EXPLAINER
# ================================
@st.cache_resource
def load_all():
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path  = os.path.join(base_dir, 'models', 'xgboost_model_fix.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')

    # Load pakai pickle — menyimpan semua informasi model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    explainer = shap.TreeExplainer(model)

    return model, scaler, explainer

model, scaler, explainer = load_all()

# ================================
# LOAD CONTOH TRANSAKSI
# ================================
@st.cache_data
def load_fraud_sample():
    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'creditcard.csv')
    df        = pd.read_csv(data_path)
    return df[df['Class'] == 1].iloc[0]

@st.cache_data
def load_normal_sample():
    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'creditcard.csv')
    df        = pd.read_csv(data_path)
    return df[df['Class'] == 0].iloc[0]

# ================================
# INISIALISASI SESSION STATE
# ================================
if 'amount' not in st.session_state:
    st.session_state['amount'] = 100.0
if 'time' not in st.session_state:
    st.session_state['time'] = 50000.0
for i in range(1, 29):
    if f'V{i}' not in st.session_state:
        st.session_state[f'V{i}'] = 0.0

# ================================
# HEADER
# ================================
st.title("🔍 Fraud Detection System")
st.markdown("Sistem deteksi transaksi kartu kredit mencurigakan menggunakan Machine Learning")
st.divider()

# ================================
# TOMBOL CONTOH
# ================================
st.subheader("Masukkan Data Transaksi")

col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    if st.button("🚨 Isi Contoh Transaksi FRAUD", use_container_width=True):
        fraud = load_fraud_sample()
        st.session_state['amount'] = float(fraud['Amount'])
        st.session_state['time']   = float(fraud['Time'])
        for i in range(1, 29):
            st.session_state[f'V{i}'] = float(fraud[f'V{i}'])
        st.rerun()

with col_btn2:
    if st.button("✅ Isi Contoh Transaksi NORMAL", use_container_width=True):
        normal = load_normal_sample()
        st.session_state['amount'] = float(normal['Amount'])
        st.session_state['time']   = float(normal['Time'])
        for i in range(1, 29):
            st.session_state[f'V{i}'] = float(normal[f'V{i}'])
        st.rerun()

with col_btn3:
    if st.button("🔄 Reset ke Default", use_container_width=True):
        st.session_state['amount'] = 100.0
        st.session_state['time']   = 50000.0
        for i in range(1, 29):
            st.session_state[f'V{i}'] = 0.0
        st.rerun()

st.divider()

# ================================
# INPUT FORM
# ================================
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "Nilai Transaksi (€)",
        min_value=0.0,
        max_value=30000.0,
        value=st.session_state['amount'],
        key='amount'
    )

with col2:
    time = st.number_input(
        "Waktu (detik sejak transaksi pertama)",
        min_value=0.0,
        value=st.session_state['time'],
        key='time'
    )

with st.expander("⚙️ Nilai V1 - V28 (terisi otomatis saat klik contoh)"):
    cols     = st.columns(4)
    v_values = {}
    for i in range(1, 29):
        with cols[(i-1) % 4]:
            v_values[f'V{i}'] = st.number_input(
                f'V{i}',
                value=st.session_state[f'V{i}'],
                format="%.4f",
                key=f'V{i}'
            )

# ================================
# TOMBOL DETEKSI
# ================================
st.divider()

if st.button("🔎 Deteksi Sekarang", type="primary", use_container_width=True):

    # Susun data input
    input_data = {'Time': time, 'Amount': amount}
    input_data.update(v_values)
    input_df = pd.DataFrame([input_data])

    # Urutkan kolom
    column_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    input_df     = input_df[column_order]

    # Scaling
    input_scaled = scaler.transform(input_df.values)

    # Prediksi
    prediksi     = model.predict(input_scaled)[0]
    probabilitas = model.predict_proba(input_scaled)[0]

    st.divider()
    st.subheader("📊 Hasil Deteksi")

    if prediksi == 1:
        st.error("⚠️ TRANSAKSI INI TERDETEKSI FRAUD!")
        st.metric(
            label="Tingkat Keyakinan Model",
            value=f"{probabilitas[1]*100:.1f}%"
        )
    else:
        st.success("✅ TRANSAKSI INI NORMAL")
        st.metric(
            label="Tingkat Keyakinan Model",
            value=f"{probabilitas[0]*100:.1f}%"
        )

    # ================================
    # SHAP EXPLANATION
    # ================================
    st.divider()
    st.subheader("🧠 Kenapa Model Memutuskan Ini?")
    st.caption("Grafik ini menjelaskan kolom mana yang paling mempengaruhi keputusan model")

    input_df_named = pd.DataFrame(input_scaled, columns=column_order)
    shap_vals      = explainer.shap_values(input_df_named)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            data=input_df_named.iloc[0],
            feature_names=column_order
        ),
        show=False
    )
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.caption("Fraud Detection System — dibuat menggunakan XGBoost + SHAP")